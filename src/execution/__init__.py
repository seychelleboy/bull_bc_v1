"""
Execution Module - Trade execution handlers

Provides MT5 connector for live trading with Pepperstone.
"""
from .mt5_connector import MT5Connector, OrderResult

__all__ = ['MT5Connector', 'OrderResult']
