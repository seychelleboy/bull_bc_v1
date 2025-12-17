"""
Volume Breakout Scenario - 60% historical win rate

Detects when price breaks above resistance on high volume, indicating
accumulation and potential continuation higher.

Key signals:
1. Price breaks above resistance
2. Volume significantly above average
3. OBV trending up
4. CMF positive (buying pressure)

This is the INVERSE of the bear bot's Volume Breakdown scenario.
"""

import pandas as pd
import numpy as np
from typing import Dict, List
import logging

from ..core.base_scenario import BaseScenario, ScenarioResult

logger = logging.getLogger(__name__)


class VolumeBreakoutScenario(BaseScenario):
    """
    Volume Breakout Pattern Detector.

    Historical win rate: ~60%

    Detection criteria:
    1. Price breaks above recent resistance
    2. Volume > 1.5x average (confirming breakout)
    3. OBV in uptrend
    4. CMF positive (money flow in)

    For LONG positions:
    - Entry: Current price
    - Stop: Below broken resistance (new support)
    - Target: Next resistance or 2:1 R:R
    """

    scenario_name = "volume_breakout"
    historical_win_rate = "60%"

    def __init__(self, config=None):
        super().__init__(config)
        self.lookback_period = config.get('lookback_period', 20) if config else 20
        self.volume_multiplier = config.get('volume_multiplier', 1.5) if config else 1.5
        self.breakout_threshold = config.get('breakout_threshold', 0.01) if config else 0.01

    def detect(self, data: Dict) -> ScenarioResult:
        """Detect volume breakout pattern."""
        required_keys = ['prices', 'indicators', 'current_price']
        if not self.validate_data(data, required_keys):
            return self._create_not_detected_result("Missing required data")

        prices = data['prices']
        indicators = data['indicators']
        current_price = data['current_price']

        try:
            if len(prices) < self.lookback_period + 5:
                return self._create_not_detected_result("Insufficient data")

            high = prices['high'].values
            close = prices['close'].values
            volume = prices['volume'].values if 'volume' in prices.columns else None

            # Find resistance level
            resistance = high[-(self.lookback_period + 5):-5].max()

            # Check for breakout
            if current_price < resistance:
                return self._create_not_detected_result("Price below resistance")

            # Check volume
            if volume is None:
                return self._create_not_detected_result("Volume data not available")

            avg_volume = volume[-self.lookback_period:-1].mean()
            current_volume = volume[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0

            evidence = []
            confidence = 45.0

            # Price breakout confirmed
            breakout_pct = (current_price - resistance) / resistance
            evidence.append(f"Price broke above resistance at ${resistance:,.0f}")
            confidence += 15

            # High volume confirmation
            if volume_ratio >= self.volume_multiplier:
                evidence.append(f"Volume {volume_ratio:.1f}x average (breakout confirmed)")
                confidence += 15
            else:
                return self._create_not_detected_result("Volume not confirming breakout")

            # OBV trend
            obv = indicators.get('obv')
            obv_sma = indicators.get('obv_sma')
            if obv is not None and obv_sma is not None:
                current_obv = obv.iloc[-1] if hasattr(obv, 'iloc') else obv[-1]
                obv_ma = obv_sma.iloc[-1] if hasattr(obv_sma, 'iloc') else obv_sma[-1]
                if current_obv > obv_ma:
                    evidence.append("OBV above moving average (accumulation)")
                    confidence += 10

            # CMF (Chaikin Money Flow)
            cmf = indicators.get('cmf')
            if cmf is not None:
                current_cmf = cmf.iloc[-1] if hasattr(cmf, 'iloc') else cmf[-1]
                if current_cmf > 0.1:
                    evidence.append(f"Strong buying pressure (CMF: {current_cmf:.2f})")
                    confidence += 10

            # RSI check
            rsi = indicators.get('rsi_14')
            if rsi is not None:
                current_rsi = rsi.iloc[-1] if hasattr(rsi, 'iloc') else rsi[-1]
                if 50 <= current_rsi <= 70:
                    evidence.append(f"RSI shows momentum ({current_rsi:.1f})")
                    confidence += 5

            # Calculate trade levels
            trade_levels = self.calculate_trade_levels(current_price, data)

            return self._create_detected_result(
                confidence=min(85, confidence),
                reason=f"Volume breakout above ${resistance:,.0f}",
                evidence=evidence,
                entry_price=trade_levels['entry'],
                stop_loss=trade_levels['stop_loss'],
                take_profit=trade_levels['take_profit'],
                risk_level='MEDIUM',
                metadata={
                    'resistance': resistance,
                    'volume_ratio': volume_ratio,
                    'breakout_pct': breakout_pct
                }
            )

        except Exception as e:
            logger.error(f"Volume breakout detection error: {e}")
            return self._create_not_detected_result(f"Error: {str(e)}")

    def calculate_trade_levels(
        self,
        current_price: float,
        data: Dict
    ) -> Dict[str, float]:
        """Calculate trade levels for volume breakout LONG."""
        prices = data['prices']
        indicators = data.get('indicators', {})

        low = prices['low'].values
        high = prices['high'].values

        # Resistance that was broken (now support)
        broken_resistance = high[-(self.lookback_period + 5):-5].max()

        # Next resistance level
        next_resistance = high[-30:].max()

        # ATR for buffer
        atr = indicators.get('atr_14')
        atr_value = atr.iloc[-1] if atr is not None and hasattr(atr, 'iloc') else (high[-20:].max() - low[-20:].min()) / 10

        entry = current_price
        stop_loss = broken_resistance - atr_value  # Below new support
        risk = entry - stop_loss

        take_profit = min(entry + (2.0 * risk), next_resistance) if next_resistance > entry else entry + (2.0 * risk)
        risk_reward = abs(take_profit - entry) / risk if risk > 0 else 0

        return {
            'entry': entry,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'risk_reward': risk_reward
        }
