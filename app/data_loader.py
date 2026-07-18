from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

import pandas as pd
from kiteconnect import KiteConnect

TIMEFRAME_CONFIG: Dict[str, Dict[str, Optional[str]]] = {
    "1 minute": {"fetch_interval": "minute", "resample_rule": None},
    "3 minutes": {"fetch_interval": "3minute", "resample_rule": None},
    "5 minutes": {"fetch_interval": "5minute", "resample_rule": None},
    "10 minutes": {"fetch_interval": "10minute", "resample_rule": None},
    "15 minutes": {"fetch_interval": "15minute", "resample_rule": None},
    "30 minutes": {"fetch_interval": "30minute", "resample_rule": None},
    "1 hour": {"fetch_interval": "60minute", "resample_rule": None},
    "2 hours": {"fetch_interval": "60minute", "resample_rule": "120min"},
    "3 hours": {"fetch_interval": "60minute", "resample_rule": "180min"},
    "4 hours": {"fetch_interval": "60minute", "resample_rule": "240min"},
    "1 day": {"fetch_interval": "day", "resample_rule": None},
    "1 week": {"fetch_interval": "day", "resample_rule": "W"},
    "1 month": {"fetch_interval": "day", "resample_rule": "ME"},
    "3 months": {"fetch_interval": "day", "resample_rule": "3ME"},
    "6 months": {"fetch_interval": "day", "resample_rule": "6ME"},
    "12 months": {"fetch_interval": "day", "resample_rule": "12ME"},
}

TIMEFRAME_OPTIONS = list(TIMEFRAME_CONFIG.keys())


def build_kite_client(api_key: str, access_token: str) -> KiteConnect:
    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access_token)
    return kite


def validate_connection(api_key: str, api_secret: str, access_token: str) -> Tuple[bool, str]:
    if not api_key or not api_secret or not access_token:
        return False, "API key, API secret, and access token are required"

    try:
        kite = build_kite_client(api_key=api_key, access_token=access_token)
        profile = kite.profile()
        user_id = profile.get("user_id", "unknown")
        return True, f"Connected as {user_id}"
    except Exception as exc:
        return False, f"Kite validation failed: {exc}"


def instrument_token(kite: KiteConnect, exchange: str, trading_symbol: str) -> int:
    instruments = kite.instruments(exchange=exchange)
    symbol_upper = trading_symbol.strip().upper()

    for item in instruments:
        if item.get("tradingsymbol", "").upper() == symbol_upper:
            return int(item["instrument_token"])

    raise ValueError(f"Trading symbol '{trading_symbol}' not found in {exchange}")


def _chunk_days(interval: str) -> int:
    if interval in {"minute", "3minute"}:
        return 10
    if interval == "5minute":
        return 20
    if interval == "10minute":
        return 45
    if interval == "15minute":
        return 75
    if interval in {"30minute", "60minute", "day"}:
        return 200
    return 30


def _interval_delta(interval: str) -> timedelta:
    if interval == "minute":
        return timedelta(minutes=1)
    if interval == "3minute":
        return timedelta(minutes=3)
    if interval == "5minute":
        return timedelta(minutes=5)
    if interval == "10minute":
        return timedelta(minutes=10)
    if interval == "15minute":
        return timedelta(minutes=15)
    if interval == "30minute":
        return timedelta(minutes=30)
    if interval == "60minute":
        return timedelta(hours=1)
    if interval == "day":
        return timedelta(days=1)
    return timedelta(minutes=1)


def fetch_historical_ohlcv(
    kite: KiteConnect,
    instrument_token: int,
    from_dt: datetime,
    to_dt: datetime,
    interval: str,
) -> pd.DataFrame:
    if from_dt >= to_dt:
        raise ValueError("From date must be before To date")

    out = []
    cursor = from_dt
    step = timedelta(days=_chunk_days(interval))
    interval_step = _interval_delta(interval)

    while cursor < to_dt:
        end_chunk = min(cursor + step, to_dt)
        candles = kite.historical_data(
            instrument_token=instrument_token,
            from_date=cursor,
            to_date=end_chunk,
            interval=interval,
            continuous=False,
            oi=False,
        )
        if candles:
            out.extend(candles)
            last_ts = pd.to_datetime(candles[-1]["date"])
            if getattr(last_ts, "tzinfo", None) is not None:
                last_ts = last_ts.tz_convert("Asia/Kolkata").tz_localize(None)
            cursor = max(last_ts.to_pydatetime() + interval_step, cursor + interval_step)
        else:
            cursor = end_chunk + interval_step

    if not out:
        raise ValueError("No historical candles returned for selected range")

    df = pd.DataFrame(out)
    df = df.rename(columns={"date": "datetime"})
    df = df[["datetime", "open", "high", "low", "close", "volume"]].copy()
    df["datetime"] = pd.to_datetime(df["datetime"], utc=False)
    df = df.drop_duplicates(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)

    return df


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    frame = df.copy()
    frame["datetime"] = pd.to_datetime(frame["datetime"], utc=False)
    frame.set_index("datetime", inplace=True)

    agg = (
        frame.resample(rule)
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        .dropna(subset=["open", "high", "low", "close"])
    )

    agg = agg.reset_index()
    return agg[["datetime", "open", "high", "low", "close", "volume"]]
