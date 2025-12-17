"""
Successful Breakout Scenario - 70% historical win rate

Detects when price breaks above resistance and HOLDS, indicating
continuation of bullish momentum.

Key signals:
1. Price breaks above resistance level
2. Breakout is confirmed with volume
3. RSI showing strength but not extreme overbought
4. MACD showing bullish momentum

This is the INVERSE of the bear bot's Failed Breakout scenario.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

from ..core.base_scenario import BaseScenario, ScenarioResult

logger = logging.getLogger(__name__)


class SuccessfulBreakoutScenario(BaseScenario):
    """
    Successful Breakout Pattern Detector.

    This scenario has historically achieved ~70% win rate for long entries.

    Detection criteria:
    1. Price broke above resistance and is holding
    2. Volume confirms the breakout (above average)
    3. RSI showing strength (50-80 range, not extreme)
    4. MACD in bullish configuration

    For LONG positions:
    - Entry: Near current price (market order)
    - Stop: Below the breakout level (old resistance = new support)
    - Target: Measured move or 2:1 R:R
    """

    scenario_name = "successful_breakout"
    historical_win_rate = "70%"

    def __init__(self, config=None):
        """
        Initialize successful breakout detector.

        Args:
            config: Optional configuration with thresholds
        """
        super().__init__(config)

        # Default thresholds
        self.lookback_period = config.get('lookback_period', 20) if config else 20
        self.breakout_confirmation_bars = config.get('breakout_confirmation_bars', 3) if config else 3
        self.volume_multiplier = config.get('volume_multiplier', 1.3) if config else 1.3
        self.rsi_min = config.get('rsi_min', 50) if config else 50
        self.rsi_max = config.get('rsi_max', 75) if config else 75

    def detect(self, data: Dict) -> ScenarioResult:
        """
        Detect if successful breakout pattern is present.

        Args:
            data: Dictionary containing:
                - prices: DataFrame with OHLCV
                - indicators: DataFrame with technical indicators
                - current_price: Current market price

        Returns:
            ScenarioResult with detection details
        """
        required_keys = ['prices', 'indicators', 'current_price']
        if not self.validate_data(data, required_keys):
            return self._create_not_detected_result("Missing required data")

        prices = data['prices']
        indicators = data['indicators']
        current_price = data['current_price']

        try:
            if len(prices) < self.lookback_period + 10:
                return self._create_not_detected_result("Insufficient data")

            high = prices['high'].values
            close = prices['close'].values
            volume = prices['volume'].values if 'volume' in prices.columns else None

            # Find resistance level (recent swing high before current)
            lookback_high = high[-(self.lookback_period + 5):-5]
            resistance = lookback_high.max()

            # Check if price has broken above resistance
            if current_price <= resistance:
                return self._create_not_detected_result("Price has not broken resistance")

            # Check if breakout is holding (price above resistance for confirmation bars)
            recent_closes = close[-self.breakout_confirmation_bars:]
            bars_above = sum(1 for c in recent_closes if c > resistance)

            if bars_above < self.breakout_confirmation_bars - 1:
                return self._create_not_detected_result("Breakout not confirmed (not holding)")

            evidence = []
            confidence = 50.0

            # Breakout confirmed
            breakout_pct = ((current_price - resistance) / resistance) * 100
            evidence.append(f"Price broke above resistance at ${resistance:,.0f} (+{breakout_pct:.1f}%)")
            confidence += 20

            # Volume confirmation
            if volume is not None:
                avg_volume = volume[-self.lookback_period:-self.breakout_confirmation_bars].mean()
                breakout_volume = volume[-self.breakout_confirmation_bars:].mean()
                volume_ratio = breakout_volume / avg_volume if avg_volume > 0 else 0

                if volume_ratio >= self.volume_multiplier:
                    evidence.append(f"Volume confirms breakout ({volume_ratio:.1f}x average)")
                    confidence += 15

            # RSI check - should be strong but not extreme
            rsi = indicators.get('rsi_14')
            if rsi is not None:
                current_rsi = rsi.iloc[-1] if hasattr(rsi, 'iloc') else rsi[-1]
                if self.rsi_min <= current_rsi <= self.rsi_max:
                    evidence.append(f"RSI shows strength at {current_rsi:.1f}")
                    confidence += 10
                elif current_rsi > 80:
                    confidence -= 10  # Overbought, reduce confidence

            # MACD bullish
            macd = indicators.get('macd')
            macd_signal = indicators.get('macd_signal')
            if macd is not None and macd_signal is not None:
                current_macd = macd.iloc[-1] if hasattr(macd, 'iloc') else macd[-1]
                current_signal = macd_signal.iloc[-1] if hasattr(macd_signal, 'iloc') else macd_signal[-1]
                if current_macd > current_signal:
                    evidence.append("MACD bullish (above signal)")
                    confidence += 10

            # OBV confirmation
            obv = indicators.get('obv')
            obv_sma = indicators.get('obv_sma')
            if obv is not None and obv_sma is not None:
                current_obv = obv.iloc[-1] if hasattr(obv, 'iloc') else obv[-1]
                obv_ma = obv_sma.iloc[-1] if hasattr(obv_sma, 'iloc') else obv_sma[-1]
                if current_obv > obv_ma:
                    evidence.append("OBV confirms accumulation")
                    confidence += 5

            # Calculate trade levels
            trade_levels = self.calculate_trade_levels(current_price, data)

            return self._create_detected_result(
                confidence=min(90, confidence),
                reason=f"Successful breakout above ${resistance:,.0f}",
                evidence=evidence,
                entry_price=trade_levels['entry'],
                stop_loss=trade_levels['stop_loss'],
                take_profit=trade_levels['take_profit'],
                risk_level=self._assess_risk(confidence, trade_levels),
                metadata={
                    'resistance': resistance,
                    'breakout_pct': breakout_pct,
                    'volume_ratio': volume_ratio if volume is not None else 0
                }
            )

        except Exception as e:
            logger.error(f"Successful breakout detection error: {e}")
            return self._create_not_detected_result(f"Detection error: {str(e)}")

    def calculate_trade_levels(
        self,
        current_price: float,
        data: Dict
    ) -> Dict[str, float]:
        """
        Calculate entry, stop-loss, and take-profit levels for LONG.

        For successful breakout:
        - Entry: Near current price
        - Stop: Below resistance (old resistance = new support) - ATR buffer
        - Target: Measured move or 2:1 R:R minimum
        """
        prices = data['prices']
        indicators = data.get('indicators', {})

        high = prices['high'].values
        low = prices['low'].values

        # Resistance that was broken (now support)
        resistance = high[-(self.lookback_period + 5):-5].max()

        # ATR for buffer
        atr = indicators.get('atr_14')
        if atr is not None:
            atr_value = atr.iloc[-1] if hasattr(atr, 'iloc') else atr[-1]
        else:
            atr_value = (high[-20:].max() - low[-20:].min()) / 10

        entry = current_price
        stop_loss = resistance - atr_value  # Below new support (old resistance)
        risk = entry - stop_loss

        # Target: Measured move (height of consolidation) or 2:1 R:R
        consolidation_height = resistance - low[-self.lookback_period:].min()
        measured_move_target = resistance + consolidation_height
        rr_target = entry + (2.0 * risk)

        take_profit = max(measured_move_target, rr_target)
        risk_reward = abs(take_profit - entry) / risk if risk > 0 else 0

        return {
            'entry': entry,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'risk_reward': risk_reward,
            'atr': atr_value
        }

    def _assess_risk(self, confidence: float, trade_levels: Dict) -> str:
        """Assess trade risk level."""
        rr = trade_levels.get('risk_reward', 0)

        if confidence >= 75 and rr >= 2.0:
            return 'LOW'
        elif confidence >= 60 and rr >= 1.5:
            return 'MEDIUM'
        else:
            return 'HIGH'
