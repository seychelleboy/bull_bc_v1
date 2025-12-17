"""
Risk Management Module - Position sizing and stop management for LONG trades

Components:
- PositionSizer: Risk-based position sizing
- StopCalculator: ATR-based stop loss management

For LONG positions:
- Stop Loss is BELOW entry
- Take Profit is ABOVE entry
- Position size based on risk amount / stop distance
"""

from .position_sizer import PositionSizer, PositionSize
from .stop_calculator import StopCalculator, StopLevels

__all__ = [
    'PositionSizer',
    'PositionSize',
    'StopCalculator',
    'StopLevels'
]
