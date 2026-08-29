import numpy as np
import math

class UserStrategy:
    """AI-generated trading strategy."""

    def __init__(self):
        self.indicators = [
            {"name": "EMA", "period": 10, "source": "close"},
            {"name": "EMA", "period": 21, "source": "close"},
        ]

    def should_enter(self, idx: int, row: dict, indicators: dict) -> bool:
        """Return True to open a LONG position."""
        if idx < 25:
            return False
        ema_fast = indicators.get("EMA_10")
        ema_slow = indicators.get("EMA_21")
        if ema_fast is None or ema_slow is None:
            return False

        # Fast EMA crosses above Slow EMA
        return (
            ema_fast[idx-1] <= ema_slow[idx-1]
            and ema_fast[idx] > ema_slow[idx]
        )

    def should_exit(self, idx: int, row: dict, indicators: dict) -> bool:
        """Return True to close the current position."""
        ema_fast = indicators.get("EMA_10")
        ema_slow = indicators.get("EMA_21")
        if ema_fast is None or ema_slow is None:
            return False

        # Fast EMA crosses below Slow EMA
        return (
            ema_fast[idx-1] >= ema_slow[idx-1]
            and ema_fast[idx] < ema_slow[idx]
        )
