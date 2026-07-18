from __future__ import annotations

from decimal import Decimal, ROUND_FLOOR, getcontext
from typing import Tuple

import numpy as np


def build_integer_range(start: int, end: int, step: int) -> np.ndarray:
    if step <= 0:
        raise ValueError("Step must be positive")
    if end < start:
        raise ValueError("End must be >= start")
    values = np.arange(start, end + 1, step, dtype=np.int32)
    if values.size == 0:
        raise ValueError("Generated empty ATR range")
    return values


def build_float_range(start: float, end: float, step: float) -> np.ndarray:
    if step <= 0:
        raise ValueError("Step must be positive")
    if end < start:
        raise ValueError("End must be >= start")

    getcontext().prec = 28
    d_start = Decimal(str(start))
    d_end = Decimal(str(end))
    d_step = Decimal(str(step))

    steps = int(((d_end - d_start) / d_step).to_integral_value(rounding=ROUND_FLOOR))
    values = [float(d_start + i * d_step) for i in range(steps + 1)]
    if not values:
        raise ValueError("Generated empty multiplier range")
    return np.asarray(values, dtype=np.float64)


def combo_count(atr_values: np.ndarray, multiplier_values: np.ndarray) -> int:
    return int(atr_values.size * multiplier_values.size)


def format_combo_size(total: int) -> str:
    if total >= 1_000_000:
        return f"{total / 1_000_000:.2f}M"
    if total >= 1_000:
        return f"{total / 1_000:.2f}K"
    return str(total)


def max_jobs(default: int = 8) -> int:
    cpu = max(1, (default if default > 0 else 1))
    return cpu


def ensure_date_range(from_date, to_date) -> Tuple[object, object]:
    if from_date > to_date:
        raise ValueError("From date must be before To date")
    return from_date, to_date
