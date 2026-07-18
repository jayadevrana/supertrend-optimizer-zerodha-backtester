from __future__ import annotations

import multiprocessing as mp
from datetime import datetime, time, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.data_loader import (
    TIMEFRAME_CONFIG,
    TIMEFRAME_OPTIONS,
    build_kite_client,
    fetch_historical_ohlcv,
    instrument_token,
    resample_ohlcv,
    validate_connection,
)
from app.optimizer import optimize_supertrend
from app.supertrend_engine import strategy_artifacts
from app.utils import build_float_range, build_integer_range, combo_count, format_combo_size


def _inject_css() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background: radial-gradient(1200px 800px at 10% -10%, #1a2438 0%, #090f1a 45%, #05080f 100%);
        }
        .metric-card {
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 12px;
            padding: 12px 14px;
        }
        .section-card {
            background: rgba(5, 12, 24, 0.75);
            border: 1px solid rgba(145, 196, 255, 0.18);
            border-radius: 14px;
            padding: 14px;
            margin-bottom: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(ttl=1200, show_spinner=False)
def _cached_instrument_token(api_key: str, access_token: str, exchange: str, symbol: str) -> int:
    kite = build_kite_client(api_key=api_key, access_token=access_token)
    return instrument_token(kite=kite, exchange=exchange, trading_symbol=symbol)


@st.cache_data(ttl=1200, show_spinner=False)
def _cached_history(
    api_key: str,
    access_token: str,
    token: int,
    from_iso: str,
    to_iso: str,
    fetch_interval: str,
    resample_rule: str,
) -> pd.DataFrame:
    kite = build_kite_client(api_key=api_key, access_token=access_token)
    df = fetch_historical_ohlcv(
        kite=kite,
        instrument_token=token,
        from_dt=datetime.fromisoformat(from_iso),
        to_dt=datetime.fromisoformat(to_iso),
        interval=fetch_interval,
    )
    if resample_rule:
        return resample_ohlcv(df, resample_rule)
    return df


def _format_results(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Multiplier"] = out["Multiplier"].round(4)
    out["Net Profit %"] = out["Net Profit %"].round(4)
    out["Win Rate %"] = out["Win Rate %"].round(2)
    out["Max Drawdown %"] = out["Max Drawdown %"].round(2)
    return out


def _selection_rows(selection_event) -> list[int]:
    if selection_event is None:
        return []
    if isinstance(selection_event, dict):
        return selection_event.get("selection", {}).get("rows", [])
    try:
        return selection_event.selection.rows
    except Exception:
        return []


def _downsample_axis(values: np.ndarray, max_points: int = 200) -> np.ndarray:
    if values.size <= max_points:
        return values
    idx = np.linspace(0, values.size - 1, max_points, dtype=int)
    return values[idx]


def run() -> None:
    st.set_page_config(page_title="Supertrend Optimizer", layout="wide")
    _inject_css()

    if "auth_ok" not in st.session_state:
        st.session_state["auth_ok"] = False
    if "results_df" not in st.session_state:
        st.session_state["results_df"] = None
    if "price_df" not in st.session_state:
        st.session_state["price_df"] = None
    if "selected_row" not in st.session_state:
        st.session_state["selected_row"] = 0

    st.title("Supertrend Backtesting + Parameter Optimizer")
    st.caption("Localhost Python app with Zerodha historical data, vectorized Supertrend, and parallel optimization.")

    with st.sidebar:
        st.header("1) Zerodha Login")
        api_key = st.text_input("API Key", type="password")
        api_secret = st.text_input("API Secret", type="password")
        access_token = st.text_input("Access Token", type="password")

        if st.button("Validate Kite Session", use_container_width=True):
            ok, message = validate_connection(
                api_key=api_key.strip(),
                api_secret=api_secret.strip(),
                access_token=access_token.strip(),
            )
            if ok:
                st.session_state["auth_ok"] = True
                st.session_state["kite_creds"] = {
                    "api_key": api_key.strip(),
                    "api_secret": api_secret.strip(),
                    "access_token": access_token.strip(),
                }
                st.success(message)
            else:
                st.session_state["auth_ok"] = False
                st.error(message)

        st.markdown("---")
        st.header("2) Instrument")
        exchange = st.selectbox("Exchange", options=["NSE", "BSE", "NFO", "MCX", "CDS"], index=0)
        trading_symbol = st.text_input("Trading Symbol", value="INFY")
        timeframe = st.selectbox("Timeframe", options=TIMEFRAME_OPTIONS, index=4)

        today = datetime.now().date()
        from_date = st.date_input("From Date", value=today - timedelta(days=120))
        to_date = st.date_input("To Date", value=today)

        st.markdown("---")
        st.header("3) Parameter Grid")
        atr_start = st.number_input("ATR Start", min_value=1, value=7, step=1)
        atr_end = st.number_input("ATR End", min_value=1, value=50, step=1)
        atr_step = st.number_input("ATR Step", min_value=1, value=1, step=1)

        mult_start = st.number_input("Multiplier Start", min_value=0.1, value=1.0, step=0.1, format="%.4f")
        mult_end = st.number_input("Multiplier End", min_value=0.1, value=5.0, step=0.1, format="%.4f")
        mult_step = st.number_input("Multiplier Step", min_value=0.01, value=0.1, step=0.01, format="%.4f")

        cpu_default = max(1, mp.cpu_count() - 1)
        n_jobs = st.number_input("Parallel Jobs", min_value=1, max_value=max(1, mp.cpu_count()), value=cpu_default, step=1)

        try:
            atr_values_preview = build_integer_range(int(atr_start), int(atr_end), int(atr_step))
            mult_values_preview = build_float_range(float(mult_start), float(mult_end), float(mult_step))
            total_combos = combo_count(atr_values_preview, mult_values_preview)
            st.info(
                f"ATR values: {atr_values_preview.size} | Multiplier values: {mult_values_preview.size} | "
                f"Total combos: {format_combo_size(total_combos)}"
            )
        except Exception as exc:
            total_combos = 0
            st.error(f"Parameter range error: {exc}")

        fetch_data_clicked = st.button("Fetch Historical Data", use_container_width=True)
        optimize_clicked = st.button("Run Optimization", use_container_width=True)

    if fetch_data_clicked:
        if not st.session_state.get("auth_ok"):
            st.error("Validate Zerodha session first.")
        elif not trading_symbol.strip():
            st.error("Trading symbol is required.")
        elif from_date > to_date:
            st.error("From date must be <= To date.")
        else:
            creds = st.session_state["kite_creds"]
            cfg = TIMEFRAME_CONFIG[timeframe]
            fetch_interval = str(cfg["fetch_interval"])
            resample_rule = cfg["resample_rule"] or ""
            from_dt = datetime.combine(from_date, time(hour=0, minute=0))
            to_dt = datetime.combine(to_date, time(hour=23, minute=59))
            with st.spinner("Fetching instrument token and historical candles from Zerodha..."):
                try:
                    token = _cached_instrument_token(
                        creds["api_key"],
                        creds["access_token"],
                        exchange,
                        trading_symbol.strip().upper(),
                    )
                    price_df = _cached_history(
                        creds["api_key"],
                        creds["access_token"],
                        token,
                        from_dt.isoformat(),
                        to_dt.isoformat(),
                        fetch_interval,
                        resample_rule,
                    )
                    st.session_state["price_df"] = price_df
                    st.session_state["results_df"] = None
                    st.success(f"Loaded {len(price_df):,} candles for {exchange}:{trading_symbol.strip().upper()}.")
                except Exception as exc:
                    st.error(f"Historical fetch failed: {exc}")

    price_df = st.session_state.get("price_df")
    if price_df is not None:
        with st.container(border=True):
            c1, c2, c3 = st.columns(3)
            c1.metric("Candles", f"{len(price_df):,}")
            c2.metric("Start", str(price_df["datetime"].iloc[0]))
            c3.metric("End", str(price_df["datetime"].iloc[-1]))

        preview_fig = go.Figure(
            data=[
                go.Candlestick(
                    x=price_df["datetime"],
                    open=price_df["open"],
                    high=price_df["high"],
                    low=price_df["low"],
                    close=price_df["close"],
                    name="OHLC",
                )
            ]
        )
        preview_fig.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10), xaxis_rangeslider_visible=False)
        st.plotly_chart(preview_fig, use_container_width=True)

    if optimize_clicked:
        if price_df is None:
            st.error("Load historical data first.")
        elif total_combos <= 0:
            st.error("Fix parameter ranges first.")
        else:
            atr_values = build_integer_range(int(atr_start), int(atr_end), int(atr_step))
            multiplier_values = build_float_range(float(mult_start), float(mult_end), float(mult_step))

            progress = st.progress(0.0, text="Preparing optimization...")
            status = st.empty()

            def _on_progress(done: int, total: int) -> None:
                pct = done / total
                progress.progress(pct, text=f"Optimizing ATR periods: {done}/{total}")
                status.caption(f"Completed {done:,}/{total:,} ATR buckets ({(pct * 100):.1f}%).")

            try:
                results_df = optimize_supertrend(
                    df=price_df,
                    atr_values=atr_values,
                    multiplier_values=multiplier_values,
                    n_jobs=int(n_jobs),
                    progress_callback=_on_progress,
                )
                st.session_state["results_df"] = results_df
                st.session_state["selected_row"] = 0
                progress.progress(1.0, text="Optimization complete")
                status.caption(f"Finished {len(results_df):,} combinations.")
                st.success("Optimization completed.")
            except Exception as exc:
                st.error(f"Optimization failed: {exc}")

    results_df = st.session_state.get("results_df")
    if results_df is None or results_df.empty:
        return

    best = results_df.iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best ATR", f"{int(best['ATR'])}")
    c2.metric("Best Multiplier", f"{best['Multiplier']:.4f}")
    c3.metric("Best Net Profit %", f"{best['Net Profit %']:.2f}")
    c4.metric("Best Max DD %", f"{best['Max Drawdown %']:.2f}")

    st.subheader("Optimization Results (Sorted by Net Profit %)" )
    max_rows = int(min(len(results_df), 200000))
    default_rows = min(max_rows, 5000)
    step_rows = 100 if max_rows >= 100 else 1
    show_rows = st.slider("Rows to display", min_value=1, max_value=max_rows, value=default_rows, step=step_rows)
    display_df = _format_results(results_df.head(show_rows))

    left, right = st.columns([2.2, 1.2], gap="large")

    with left:
        selection_event = st.dataframe(
            display_df,
            hide_index=True,
            use_container_width=True,
            height=580,
            on_select="rerun",
            selection_mode="single-row",
        )
        rows = _selection_rows(selection_event)
        if rows:
            st.session_state["selected_row"] = int(rows[0])

    selected_idx = int(st.session_state.get("selected_row", 0))
    selected_idx = max(0, min(selected_idx, len(display_df) - 1))
    selected_row = display_df.iloc[selected_idx]

    with right:
        with st.expander("Trade Detail Panel", expanded=True):
            st.write(
                f"Selected: ATR **{int(selected_row['ATR'])}**, Multiplier **{selected_row['Multiplier']:.4f}**"
            )
            artifacts = strategy_artifacts(
                df=price_df,
                atr_period=int(selected_row["ATR"]),
                multiplier=float(selected_row["Multiplier"]),
            )
            trade_df = artifacts.trade_df.copy()
            if trade_df.empty:
                st.warning("No trades found for selected parameters.")
            else:
                trade_df["Entry Price"] = trade_df["Entry Price"].round(4)
                trade_df["Exit Price"] = trade_df["Exit Price"].round(4)
                trade_df["PnL %"] = trade_df["PnL %"].round(4)
                st.dataframe(trade_df, hide_index=True, use_container_width=True, height=360)

            equity_series = pd.Series(artifacts.equity_curve, index=pd.to_datetime(price_df["datetime"]))
            eq_fig = go.Figure(
                data=[go.Scatter(x=equity_series.index, y=equity_series.values, mode="lines", name="Equity")]
            )
            eq_fig.update_layout(height=280, margin=dict(l=10, r=10, t=20, b=10), title="Equity Curve")
            st.plotly_chart(eq_fig, use_container_width=True)

    st.subheader("Parameter Profit Heatmap")
    pivot = results_df.pivot(index="ATR", columns="Multiplier", values="Net Profit %")
    atr_axis = pivot.index.to_numpy()
    mult_axis = pivot.columns.to_numpy()

    sampled_atr = _downsample_axis(atr_axis, max_points=200)
    sampled_mult = _downsample_axis(mult_axis, max_points=200)
    pivot_sampled = pivot.loc[sampled_atr, sampled_mult]

    heatmap = px.imshow(
        pivot_sampled.values,
        x=sampled_mult,
        y=sampled_atr,
        aspect="auto",
        color_continuous_scale="RdYlGn",
        labels={"x": "Multiplier", "y": "ATR", "color": "Net Profit %"},
    )
    heatmap.add_scatter(
        x=[best["Multiplier"]],
        y=[best["ATR"]],
        mode="markers",
        marker=dict(color="#1f77b4", size=10, symbol="x"),
        name="Best",
    )
    heatmap.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(heatmap, use_container_width=True)

    st.subheader("3D Surface (ATR vs Multiplier vs Net Profit %)")
    surface = go.Figure(
        data=[
            go.Surface(
                z=pivot_sampled.values,
                x=sampled_mult,
                y=sampled_atr,
                colorscale="Viridis",
                colorbar=dict(title="Net Profit %"),
            )
        ]
    )
    surface.add_trace(
        go.Scatter3d(
            x=[best["Multiplier"]],
            y=[best["ATR"]],
            z=[best["Net Profit %"]],
            mode="markers",
            marker=dict(size=6, color="red"),
            name="Best",
        )
    )
    surface.update_layout(
        height=520,
        margin=dict(l=0, r=0, t=30, b=0),
        scene=dict(xaxis_title="Multiplier", yaxis_title="ATR", zaxis_title="Net Profit %"),
    )
    st.plotly_chart(surface, use_container_width=True)


if __name__ == "__main__":
    run()
