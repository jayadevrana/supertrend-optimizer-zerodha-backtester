from __future__ import annotations

import multiprocessing as mp
from typing import Callable, Iterable, Optional

import numpy as np
import pandas as pd

from app.supertrend_engine import ema_atr, strategy_metrics_from_atr, true_range

ProgressCallback = Optional[Callable[[int, int], None]]

_WORKER_CLOSE: np.ndarray | None = None
_WORKER_HIGH: np.ndarray | None = None
_WORKER_LOW: np.ndarray | None = None
_WORKER_TR: np.ndarray | None = None
_WORKER_MULTIPLIERS: np.ndarray | None = None


def _init_worker(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    tr: np.ndarray,
    multipliers: np.ndarray,
) -> None:
    global _WORKER_CLOSE, _WORKER_HIGH, _WORKER_LOW, _WORKER_TR, _WORKER_MULTIPLIERS
    _WORKER_CLOSE = close
    _WORKER_HIGH = high
    _WORKER_LOW = low
    _WORKER_TR = tr
    _WORKER_MULTIPLIERS = multipliers


def _evaluate_period_worker(period: int) -> np.ndarray:
    if any(v is None for v in (_WORKER_CLOSE, _WORKER_HIGH, _WORKER_LOW, _WORKER_TR, _WORKER_MULTIPLIERS)):
        raise RuntimeError("Worker state was not initialized")

    atr = ema_atr(_WORKER_TR, int(period))
    out = np.empty((_WORKER_MULTIPLIERS.size, 6), dtype=np.float64)

    for i, multiplier in enumerate(_WORKER_MULTIPLIERS):
        metrics = strategy_metrics_from_atr(
            close=_WORKER_CLOSE,
            high=_WORKER_HIGH,
            low=_WORKER_LOW,
            atr=atr,
            multiplier=float(multiplier),
        )
        out[i, 0] = int(period)
        out[i, 1] = float(multiplier)
        out[i, 2] = metrics["net_profit_pct"]
        out[i, 3] = metrics["trades"]
        out[i, 4] = metrics["win_rate_pct"]
        out[i, 5] = metrics["max_drawdown_pct"]

    return out


def _evaluate_period_local(
    period: int,
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    tr: np.ndarray,
    multipliers: np.ndarray,
) -> np.ndarray:
    atr = ema_atr(tr, int(period))
    out = np.empty((multipliers.size, 6), dtype=np.float64)

    for i, multiplier in enumerate(multipliers):
        metrics = strategy_metrics_from_atr(
            close=close,
            high=high,
            low=low,
            atr=atr,
            multiplier=float(multiplier),
        )
        out[i, 0] = int(period)
        out[i, 1] = float(multiplier)
        out[i, 2] = metrics["net_profit_pct"]
        out[i, 3] = metrics["trades"]
        out[i, 4] = metrics["win_rate_pct"]
        out[i, 5] = metrics["max_drawdown_pct"]

    return out


def optimize_supertrend(
    df: pd.DataFrame,
    atr_values: np.ndarray,
    multiplier_values: np.ndarray,
    n_jobs: int,
    progress_callback: ProgressCallback = None,
) -> pd.DataFrame:
    close = df["close"].to_numpy(dtype=np.float64, copy=False)
    high = df["high"].to_numpy(dtype=np.float64, copy=False)
    low = df["low"].to_numpy(dtype=np.float64, copy=False)

    tr = true_range(high=high, low=low, close=close)
    periods = [int(x) for x in atr_values.tolist()]

    chunks: list[np.ndarray] = []
    total_periods = len(periods)

    if n_jobs <= 1:
        for i, period in enumerate(periods, start=1):
            chunks.append(
                _evaluate_period_local(
                    period=period,
                    close=close,
                    high=high,
                    low=low,
                    tr=tr,
                    multipliers=multiplier_values,
                )
            )
            if progress_callback:
                progress_callback(i, total_periods)
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(
            processes=n_jobs,
            initializer=_init_worker,
            initargs=(close, high, low, tr, multiplier_values),
        ) as pool:
            for i, block in enumerate(pool.imap_unordered(_evaluate_period_worker, periods, chunksize=1), start=1):
                chunks.append(block)
                if progress_callback:
                    progress_callback(i, total_periods)

    if not chunks:
        return pd.DataFrame(columns=["ATR", "Multiplier", "Net Profit %", "Trades", "Win Rate %", "Max Drawdown %"])

    matrix = np.concatenate(chunks, axis=0)
    results = pd.DataFrame(
        matrix,
        columns=["ATR", "Multiplier", "Net Profit %", "Trades", "Win Rate %", "Max Drawdown %"],
    )
    results["ATR"] = results["ATR"].astype(int)
    results["Trades"] = results["Trades"].astype(int)
    results.sort_values(by="Net Profit %", ascending=False, inplace=True, kind="mergesort")
    results.reset_index(drop=True, inplace=True)
    return results
