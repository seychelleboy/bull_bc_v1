"""
Stop Calculator - ATR-based stop loss management for LONG trades

Calculates and manages stop losses:
- Initial stop based on ATR
- Trailing stop as position moves into profit
- Take profit levels

For LONG positions:
- Stop Loss is BELOW entry
- Take Profit is ABOVE entry
- Trailing stop moves UP as price rises
"""

import logging
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class StopLevels:
    """Stop loss and take profit levels."""
    entry: float
    stop_loss: float
    take_profit: float
    risk: float           # Risk distance
    reward: float         # Reward distance
    risk_reward: float    # R:R ratio
    atr: float            # ATR used for calculation


class StopCalculator:
    """
    ATR-based stop loss calculator for LONG trades.

    For LONG positions:
    - Stop Loss: entry_price - (atr * multiplier) [BELOW entry]
    - Take Profit: entry_price + (risk * risk_reward_ratio) [ABOVE entry]

    Example:
        calc = StopCalculator(atr_multiplier=2.0, risk_reward=2.0)
        levels = calc.calculate(
            entry_price=50000,
            atr=1000,
            support=48000
        )
        print(f"SL: ${levels.stop_loss:,.0f}, TP: ${levels.take_profit:,.0f}")
    """

    def __init__(
        self,
        atr_multiplier: float = 2.0,
        risk_reward: float = 2.0,
        min_stop_pct: float = 0.5,
        max_stop_pct: float = 5.0,
        trailing_activation: float = 1.0,  # Activate after 1R profit
        trailing_step: float = 0.5         # Trail by 0.5R
    ):
        """
        Initialize stop calculator.

        Args:
            atr_multiplier: ATR multiplier for stop distance
            risk_reward: Minimum risk:reward ratio
            min_stop_pct: Minimum stop distance as % of entry
            max_stop_pct: Maximum stop distance as % of entry
            trailing_activation: Profit (in R) to activate trailing
            trailing_step: How much to trail (in R)
        """
        self.atr_multiplier = atr_multiplier
        self.risk_reward = risk_reward
        self.min_stop_pct = min_stop_pct
        self.max_stop_pct = max_stop_pct
        self.trailing_activation = trailing_activation
        self.trailing_step = trailing_step

    def calculate(
        self,
        entry_price: float,
        atr: float,
        support: Optional[float] = None,
        resistance: Optional[float] = None
    ) -> StopLevels:
        """
        Calculate stop loss and take profit for LONG trade.

        Args:
            entry_price: Entry price
            atr: Average True Range
            support: Optional support level for stop placement
            resistance: Optional resistance level for target

        Returns:
            StopLevels with all calculated values
        """
        # Calculate ATR-based stop
        atr_stop_distance = atr * self.atr_multiplier

        # Apply min/max constraints
        min_stop_distance = entry_price * (self.min_stop_pct / 100)
        max_stop_distance = entry_price * (self.max_stop_pct / 100)

        stop_distance = max(min_stop_distance, min(max_stop_distance, atr_stop_distance))

        # For LONG: Stop is BELOW entry
        stop_loss = entry_price - stop_distance

        # Use support if provided and closer
        if support and support < entry_price:
            support_stop = support - (atr * 0.5)  # Small buffer below support
            if support_stop > stop_loss:
                stop_loss = support_stop
                stop_distance = entry_price - stop_loss

        risk = stop_distance

        # Calculate take profit
        # For LONG: TP is ABOVE entry
        reward = risk * self.risk_reward
        take_profit = entry_price + reward

        # Use resistance if provided and reasonable
        if resistance and resistance > entry_price:
            # If resistance gives better R:R, use it
            resistance_reward = resistance - entry_price
            if resistance_reward >= reward:
                take_profit = resistance
                reward = resistance_reward

        risk_reward_actual = reward / risk if risk > 0 else 0

        logger.debug(
            f"LONG stops: Entry=${entry_price:,.0f}, "
            f"SL=${stop_loss:,.0f} (-{(risk/entry_price)*100:.1f}%), "
            f"TP=${take_profit:,.0f} (+{(reward/entry_price)*100:.1f}%), "
            f"R:R={risk_reward_actual:.1f}"
        )

        return StopLevels(
            entry=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk=risk,
            reward=reward,
            risk_reward=risk_reward_actual,
            atr=atr
        )

    def calculate_trailing_stop(
        self,
        entry_price: float,
        current_price: float,
        current_stop: float,
        initial_risk: float
    ) -> Tuple[float, bool]:
        """
        Calculate trailing stop for LONG position.

        For LONG: Stop trails UP as price rises.

        Args:
            entry_price: Original entry price
            current_price: Current market price
            current_stop: Current stop loss level
            initial_risk: Initial risk (entry - original stop)

        Returns:
            Tuple of (new_stop, was_updated)
        """
        # Calculate current profit in R
        profit = current_price - entry_price
        profit_in_r = profit / initial_risk if initial_risk > 0 else 0

        # Check if trailing should activate
        if profit_in_r < self.trailing_activation:
            return current_stop, False

        # Calculate new trailing stop
        # Trail behind current price by trailing_step * R
        trail_distance = initial_risk * self.trailing_step
        new_stop = current_price - trail_distance

        # Only update if new stop is higher than current
        # (For LONG, we move stop UP to lock in profits)
        if new_stop > current_stop:
            logger.info(
                f"Trailing stop updated: ${current_stop:,.0f} → ${new_stop:,.0f} "
                f"(profit={profit_in_r:.1f}R)"
            )
            return new_stop, True

        return current_stop, False

    def is_stop_hit(
        self,
        current_price: float,
        stop_loss: float,
        is_long: bool = True
    ) -> bool:
        """
        Check if stop loss is hit.

        Args:
            current_price: Current market price
            stop_loss: Stop loss level
            is_long: True for long position

        Returns:
            True if stop is hit
        """
        if is_long:
            # LONG: Stop hit when price falls to/below stop
            return current_price <= stop_loss
        else:
            # SHORT: Stop hit when price rises to/above stop
            return current_price >= stop_loss

    def is_target_hit(
        self,
        current_price: float,
        take_profit: float,
        is_long: bool = True
    ) -> bool:
        """
        Check if take profit is hit.

        Args:
            current_price: Current market price
            take_profit: Take profit level
            is_long: True for long position

        Returns:
            True if target is hit
        """
        if is_long:
            # LONG: Target hit when price rises to/above target
            return current_price >= take_profit
        else:
            # SHORT: Target hit when price falls to/below target
            return current_price <= take_profit
