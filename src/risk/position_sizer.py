"""
Position Sizer - Risk-based position sizing for LONG trades

Calculates position size based on:
- Account balance
- Risk per trade (default 2%)
- Stop loss distance
- Signal confidence

For LONG positions:
- Risk = Entry Price - Stop Loss
- Position Size = Risk Amount / Risk Distance
"""

import logging
from typing import Dict, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PositionSize:
    """Position sizing result."""
    size: float              # Position size in BTC
    size_usd: float          # Position value in USD
    risk_amount: float       # Dollar amount at risk
    risk_percent: float      # Percent of account at risk
    leverage: float          # Effective leverage
    stop_distance_pct: float # Stop distance as percentage


class PositionSizer:
    """
    Risk-based position sizing for LONG trades.

    Formula: position_size = (account_balance * risk_percent) / stop_distance

    Example:
        sizer = PositionSizer(account_balance=10000, risk_percent=2.0)
        pos = sizer.calculate(
            entry_price=50000,
            stop_loss=49000,
            confidence=85
        )
        print(f"Position: {pos.size:.4f} BTC (${pos.size_usd:,.2f})")
    """

    def __init__(
        self,
        account_balance: float = 10000.0,
        risk_percent: float = 2.0,
        max_position_percent: float = 25.0,
        max_leverage: float = 3.0,
        min_confidence: float = 60.0
    ):
        """
        Initialize position sizer.

        Args:
            account_balance: Account balance in USD
            risk_percent: Percent of account to risk per trade
            max_position_percent: Maximum position size as % of account
            max_leverage: Maximum leverage allowed
            min_confidence: Minimum confidence to take full position
        """
        self.account_balance = account_balance
        self.risk_percent = risk_percent
        self.max_position_percent = max_position_percent
        self.max_leverage = max_leverage
        self.min_confidence = min_confidence

    def calculate(
        self,
        entry_price: float,
        stop_loss: float,
        confidence: float = 80.0,
        take_profit: Optional[float] = None
    ) -> PositionSize:
        """
        Calculate position size for a LONG trade.

        Args:
            entry_price: Entry price
            stop_loss: Stop loss price (BELOW entry for LONG)
            confidence: Signal confidence (0-100)
            take_profit: Optional take profit price

        Returns:
            PositionSize with all calculated values
        """
        # Validate for LONG position
        if stop_loss >= entry_price:
            logger.warning(f"Invalid LONG: stop {stop_loss} >= entry {entry_price}")
            return PositionSize(
                size=0, size_usd=0, risk_amount=0,
                risk_percent=0, leverage=0, stop_distance_pct=0
            )

        # Calculate risk distance
        risk_distance = entry_price - stop_loss
        stop_distance_pct = (risk_distance / entry_price) * 100

        # Base risk amount
        risk_amount = self.account_balance * (self.risk_percent / 100)

        # Adjust for confidence
        if confidence < self.min_confidence:
            # Reduce position for low confidence
            confidence_factor = confidence / self.min_confidence
            risk_amount *= confidence_factor
            logger.debug(f"Reduced risk for low confidence: {confidence_factor:.2f}x")

        # Calculate position size
        position_size_btc = risk_amount / risk_distance
        position_size_usd = position_size_btc * entry_price

        # Apply maximum position limit
        max_position_usd = self.account_balance * (self.max_position_percent / 100)
        if position_size_usd > max_position_usd:
            position_size_usd = max_position_usd
            position_size_btc = position_size_usd / entry_price
            # Recalculate actual risk
            risk_amount = position_size_btc * risk_distance
            logger.debug(f"Position capped at max: ${max_position_usd:,.2f}")

        # Check leverage
        leverage = position_size_usd / self.account_balance
        if leverage > self.max_leverage:
            # Reduce to max leverage
            position_size_usd = self.account_balance * self.max_leverage
            position_size_btc = position_size_usd / entry_price
            risk_amount = position_size_btc * risk_distance
            leverage = self.max_leverage
            logger.debug(f"Position reduced for max leverage: {self.max_leverage}x")

        actual_risk_percent = (risk_amount / self.account_balance) * 100

        logger.info(
            f"Position sized: {position_size_btc:.4f} BTC (${position_size_usd:,.2f}), "
            f"risk=${risk_amount:.2f} ({actual_risk_percent:.1f}%), "
            f"leverage={leverage:.1f}x"
        )

        return PositionSize(
            size=position_size_btc,
            size_usd=position_size_usd,
            risk_amount=risk_amount,
            risk_percent=actual_risk_percent,
            leverage=leverage,
            stop_distance_pct=stop_distance_pct
        )

    def update_balance(self, new_balance: float) -> None:
        """Update account balance."""
        self.account_balance = new_balance
        logger.info(f"Account balance updated to ${new_balance:,.2f}")

    def get_max_position(self, entry_price: float) -> Dict:
        """
        Get maximum allowed position at given price.

        Args:
            entry_price: Entry price

        Returns:
            Dictionary with max position details
        """
        max_usd = self.account_balance * (self.max_position_percent / 100)
        max_btc = max_usd / entry_price
        max_leverage_usd = self.account_balance * self.max_leverage
        max_leverage_btc = max_leverage_usd / entry_price

        return {
            'max_position_usd': min(max_usd, max_leverage_usd),
            'max_position_btc': min(max_btc, max_leverage_btc),
            'max_risk_usd': self.account_balance * (self.risk_percent / 100)
        }
