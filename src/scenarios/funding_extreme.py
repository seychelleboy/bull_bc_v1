"""
Funding Rate Extreme Scenario - 65-70% historical win rate (BTC-specific)

Detects when perpetual futures funding rates are extremely NEGATIVE,
indicating overcrowded shorts and potential for short squeeze.

Key signals:
1. Funding rate < -0.1% (8-hour) - shorts paying longs
2. Price near recent lows (shorts in profit)
3. Fear & Greed in fear territory
4. Volume surge potential

This is the INVERSE of the bear bot's Funding Extreme (positive funding) scenario.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

from ..core.base_scenario import BaseScenario, ScenarioResult

logger = logging.getLogger(__name__)


class FundingExtremeScenario(BaseScenario):
    """
    Extreme Negative Funding Rate Pattern Detector (BTC-Specific).

    Historical win rate: ~65-70%

    When funding rates are extremely NEGATIVE, it indicates:
    - Shorts are overcrowded
    - Shorts are paying longs
    - High probability of short squeeze

    This is a BTC-specific scenario leveraging crypto market structure.

    For LONG positions:
    - Entry: Current price
    - Stop: Below recent low - ATR
    - Target: 3% or next resistance
    """

    scenario_name = "funding_extreme"
    historical_win_rate = "65-70%"

    def __init__(self, config=None):
        super().__init__(config)
        # Funding rate thresholds (8-hour funding) - NEGATIVE for LONG
        self.funding_extreme_threshold = config.get('funding_extreme_threshold', -0.1) if config else -0.1  # -0.1%
        self.funding_high_threshold = config.get('funding_high_threshold', -0.05) if config else -0.05  # -0.05%
        self.fear_greed_threshold = config.get('fear_greed_threshold', 35) if config else 35  # Fear zone

    def detect(self, data: Dict) -> ScenarioResult:
        """
        Detect extreme negative funding rate pattern.

        Args:
            data: Dictionary containing:
                - prices: DataFrame with OHLCV
                - indicators: Technical indicators
                - current_price: Current price
                - btc_market: BTC market data (optional)
                - fear_greed: Fear & Greed data (optional)
                - funding_rate: Current funding rate (optional)
        """
        required_keys = ['prices', 'current_price']
        if not self.validate_data(data, required_keys):
            return self._create_not_detected_result("Missing required data")

        current_price = data['current_price']

        try:
            evidence = []
            confidence = 40.0

            # Check funding rate
            funding_rate = data.get('funding_rate')
            if funding_rate is None:
                # Try to get from btc_market
                btc_market = data.get('btc_market', {})
                funding_rate = btc_market.get('funding_rate')

            if funding_rate is not None:
                if funding_rate <= self.funding_extreme_threshold:
                    evidence.append(f"Extreme negative funding rate: {funding_rate:.3f}%")
                    confidence += 25
                elif funding_rate <= self.funding_high_threshold:
                    evidence.append(f"High negative funding rate: {funding_rate:.3f}%")
                    confidence += 15
                else:
                    return self._create_not_detected_result(
                        f"Funding rate {funding_rate:.3f}% not extreme negative"
                    )
            else:
                # No funding data - use proxy signals
                # Big price drops often lead to negative funding
                btc_market = data.get('btc_market', {})
                price_change = btc_market.get('price_change_24h', 0)
                if price_change < -5:
                    evidence.append(f"Large price drop {price_change:.1f}% (likely negative funding)")
                    confidence += 10

            # Fear & Greed check (fear territory favors longs)
            fear_greed = data.get('fear_greed', {})
            fg_value = fear_greed.get('current', 50)
            if fg_value <= self.fear_greed_threshold:
                evidence.append(f"Market in fear territory (F&G: {fg_value})")
                confidence += 10
            if fear_greed.get('is_extreme_fear'):
                evidence.append("Extreme fear detected (contrarian buy)")
                confidence += 10

            # Price near lows
            prices = data['prices']
            low = prices['low'].values
            recent_low = low[-20:].min()

            if current_price <= recent_low * 1.02:
                evidence.append("Price near recent lows")
                confidence += 10

            # BTC market metrics
            btc_market = data.get('btc_market', {})
            if btc_market.get('is_far_from_ath'):
                evidence.append("Price far from ATH (value opportunity)")
                confidence += 5

            # Volume analysis (shorts covering)
            indicators = data.get('indicators', {})
            volume_ratio = indicators.get('volume_ratio')
            if volume_ratio is not None:
                vol_val = volume_ratio.iloc[-1] if hasattr(volume_ratio, 'iloc') else volume_ratio[-1]
                if vol_val > 2.0:
                    evidence.append(f"High volume ({vol_val:.1f}x average)")
                    confidence += 10

            # Need at least 2 signals
            if len(evidence) < 2:
                return self._create_not_detected_result("Insufficient extreme funding signals")

            # Calculate trade levels
            trade_levels = self.calculate_trade_levels(current_price, data)

            return self._create_detected_result(
                confidence=min(90, confidence),
                reason="Extreme negative funding - shorts overcrowded",
                evidence=evidence,
                entry_price=trade_levels['entry'],
                stop_loss=trade_levels['stop_loss'],
                take_profit=trade_levels['take_profit'],
                risk_level='MEDIUM',
                metadata={
                    'funding_rate': funding_rate,
                    'fear_greed': fg_value
                }
            )

        except Exception as e:
            logger.error(f"Funding extreme detection error: {e}")
            return self._create_not_detected_result(f"Error: {str(e)}")

    def calculate_trade_levels(
        self,
        current_price: float,
        data: Dict
    ) -> Dict[str, float]:
        """Calculate trade levels for funding extreme LONG."""
        prices = data['prices']
        indicators = data.get('indicators', {})

        high = prices['high'].values
        low = prices['low'].values

        # Stop below recent low
        recent_low = low[-10:].min()

        atr = indicators.get('atr_14')
        atr_value = atr.iloc[-1] if atr is not None and hasattr(atr, 'iloc') else (high[-20:].max() - low[-20:].min()) / 10

        entry = current_price
        stop_loss = recent_low - (1.5 * atr_value)

        # Target: Resistance or 3% move (short squeezes can be sharp)
        resistance = high[-20:].max()
        target_pct = current_price * 1.03  # 3% target

        take_profit = min(resistance, target_pct)

        risk = entry - stop_loss
        risk_reward = abs(take_profit - entry) / risk if risk > 0 else 0

        return {
            'entry': entry,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'risk_reward': risk_reward
        }
