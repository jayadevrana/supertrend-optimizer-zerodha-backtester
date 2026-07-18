from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd


@dataclass
class StrategyArtifacts:
    direction: np.ndarray
    equity_curve: np.ndarray
    trade_df: pd.DataFrame


def true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum.reduce(
        [
            high - low,
            np.abs(high - prev_close),
            np.abs(low - prev_close),
        ]
    )
    return tr.astype(np.float64, copy=False)


def ema_atr(tr: np.ndarray, period: int) -> np.ndarray:
    if period <= 0:
        raise ValueError("ATR period must be > 0")
    alpha = 2.0 / (float(period) + 1.0)
    atr = np.empty_like(tr, dtype=np.float64)
    atr[0] = tr[0]
    for i in range(1, tr.size):
        atr[i] = alpha * tr[i] + (1.0 - alpha) * atr[i - 1]
    return atr


def supertrend_direction(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    atr: np.ndarray,
    multiplier: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = close.size
    hl2 = (high + low) / 2.0
    upper_basic = hl2 + multiplier * atr
    lower_basic = hl2 - multiplier * atr

    final_upper = np.empty(n, dtype=np.float64)
    final_lower = np.empty(n, dtype=np.float64)
    direction = np.empty(n, dtype=np.int8)

    final_upper[0] = upper_basic[0]
    final_lower[0] = lower_basic[0]
    direction[0] = 1

    for i in range(1, n):
        prev_upper = final_upper[i - 1]
        prev_lower = final_lower[i - 1]

        cur_upper = upper_basic[i]
        cur_lower = lower_basic[i]

        final_upper[i] = (
            cur_upper if (cur_upper < prev_upper or close[i - 1] > prev_upper) else prev_upper
        )
        final_lower[i] = (
            cur_lower if (cur_lower > prev_lower or close[i - 1] < prev_lower) else prev_lower
        )

        if direction[i - 1] == 1:
            direction[i] = 1 if close[i] >= final_lower[i] else -1
        else:
            direction[i] = -1 if close[i] <= final_upper[i] else 1

    return direction, final_upper, final_lower


def strategy_metrics_from_atr(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    atr: np.ndarray,
    multiplier: float,
) -> Dict[str, float]:
    direction, _, _ = supertrend_direction(high=high, low=low, close=close, atr=atr, multiplier=multiplier)
    equity = compute_equity_curve(close=close, direction=direction)
    net_profit_pct = (equity[-1] - 1.0) * 100.0

    running_max = np.maximum.accumulate(equity)
    drawdown = (equity / running_max) - 1.0
    max_drawdown_pct = drawdown.min() * 100.0

    flip_idx = np.flatnonzero(direction[1:] != direction[:-1]) + 1
    entries = np.concatenate(([0], flip_idx))
    exits = np.concatenate((flip_idx, [close.size - 1]))

    entry_prices = close[entries]
    exit_prices = close[exits]
    trade_dir = direction[entries]
    trade_pnl_pct = trade_dir * ((exit_prices - entry_prices) / entry_prices) * 100.0

    trades = int(trade_pnl_pct.size)
    win_rate_pct = (float((trade_pnl_pct > 0.0).mean()) * 100.0) if trades else 0.0

    return {
        "net_profit_pct": float(net_profit_pct),
        "trades": float(trades),
        "win_rate_pct": float(win_rate_pct),
        "max_drawdown_pct": float(max_drawdown_pct),
    }


def compute_equity_curve(close: np.ndarray, direction: np.ndarray) -> np.ndarray:
    returns = np.diff(close) / close[:-1]
    strategy_returns = returns * direction[:-1]

    equity = np.empty(close.size, dtype=np.float64)
    equity[0] = 1.0
    if strategy_returns.size:
        equity[1:] = np.cumprod(1.0 + strategy_returns)
    return equity


def strategy_artifacts(
    df: pd.DataFrame,
    atr_period: int,
    multiplier: float,
) -> StrategyArtifacts:
    close = df["close"].to_numpy(dtype=np.float64, copy=False)
    high = df["high"].to_numpy(dtype=np.float64, copy=False)
    low = df["low"].to_numpy(dtype=np.float64, copy=False)
    ts = pd.to_datetime(df["datetime"], utc=False)

    tr = true_range(high=high, low=low, close=close)
    atr = ema_atr(tr=tr, period=atr_period)
    direction, _, _ = supertrend_direction(
        high=high,
        low=low,
        close=close,
        atr=atr,
        multiplier=multiplier,
    )
    equity = compute_equity_curve(close=close, direction=direction)

    flip_idx = np.flatnonzero(direction[1:] != direction[:-1]) + 1
    entries = np.concatenate(([0], flip_idx))
    exits = np.concatenate((flip_idx, [close.size - 1]))

    entry_prices = close[entries]
    exit_prices = close[exits]
    trade_dir = direction[entries]
    trade_pnl_pct = trade_dir * ((exit_prices - entry_prices) / entry_prices) * 100.0

    trade_df = pd.DataFrame(
        {
            "Trade #": np.arange(1, trade_pnl_pct.size + 1),
            "Direction": np.where(trade_dir > 0, "Long", "Short"),
            "Entry Time": ts.iloc[entries].to_numpy(),
            "Entry Price": entry_prices,
            "Exit Time": ts.iloc[exits].to_numpy(),
            "Exit Price": exit_prices,
            "PnL %": trade_pnl_pct,
        }
    )
    if not trade_df.empty:
        trade_df["Cumulative Equity"] = (1.0 + trade_df["PnL %"] / 100.0).cumprod()

    return StrategyArtifacts(direction=direction, equity_curve=equity, trade_df=trade_df)
