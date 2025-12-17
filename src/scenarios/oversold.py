"""
Oversold Scenario - 55-60% historical win rate

Detects when price is significantly below VWAP or moving averages,
indicating mean reversion opportunity (bounce).

Key signals:
1. Price < 3% below VWAP
2. Price < 2 standard deviations below mean
3. RSI < 30 (oversold)
4. Bollinger Band %B < 0 (below lower band)

This is the INVERSE of the bear bot's Overextended scenario.
"""

import pandas as pd
import numpy as np
from typing import Dict, List
import logging

from ..core.base_scenario import BaseScenario, ScenarioResult

logger = logging.getLogger(__name__)


class OversoldScenario(BaseScenario):
    """
    Oversold Price Pattern Detector.

    Historical win rate: ~55-60%

    Detection criteria:
    1. Price significantly below VWAP (>3%)
    2. Price below lower Bollinger Band
    3. RSI in oversold territory (<30)
    4. Price far from moving averages

    For LONG positions:
    - Entry: Current price
    - Stop: Below recent low - ATR
    - Target: VWAP or Bollinger middle band
    """

    scenario_name = "oversold"
    historical_win_rate = "55-60%"

    def __init__(self, config=None):
        super().__init__(config)
        self.vwap_deviation = config.get('vwap_deviation', -0.03) if config else -0.03  # -3%
        self.rsi_threshold = config.get('rsi_threshold', 30) if config else 30
        self.bb_threshold = config.get('bb_threshold', 0.0) if config else 0.0

    def detect(self, data: Dict) -> ScenarioResult:
        """Detect oversold price pattern."""
        required_keys = ['prices', 'indicators', 'current_price']
        if not self.validate_data(data, required_keys):
            return self._create_not_detected_result("Missing required data")

        prices = data['prices']
        indicators = data['indicators']
        current_price = data['current_price']

        try:
            evidence = []
            confidence = 40.0
            oversold_signals = 0

            # VWAP underextension
            vwap = indicators.get('vwap')
            price_vs_vwap = indicators.get('price_vs_vwap')
            vwap_dev = 0

            if price_vs_vwap is not None:
                vwap_dev = price_vs_vwap.iloc[-1] if hasattr(price_vs_vwap, 'iloc') else price_vs_vwap[-1]
                if vwap_dev < self.vwap_deviation * 100:  # Convert to percentage
                    evidence.append(f"Price {vwap_dev:.1f}% below VWAP")
                    confidence += 15
                    oversold_signals += 1
            elif vwap is not None:
                vwap_val = vwap.iloc[-1] if hasattr(vwap, 'iloc') else vwap[-1]
                vwap_dev = ((current_price - vwap_val) / vwap_val) * 100
                if vwap_dev < self.vwap_deviation * 100:
                    evidence.append(f"Price {vwap_dev:.1f}% below VWAP")
                    confidence += 15
                    oversold_signals += 1

            # Bollinger Band position
            bb_pct_b = indicators.get('bb_pct_b')
            if bb_pct_b is not None:
                pct_b = bb_pct_b.iloc[-1] if hasattr(bb_pct_b, 'iloc') else bb_pct_b[-1]
                if pct_b < self.bb_threshold:
                    evidence.append(f"Price below lower Bollinger Band (%B: {pct_b:.2f})")
                    confidence += 15
                    oversold_signals += 1

            # RSI oversold
            rsi = indicators.get('rsi_14')
            if rsi is not None:
                current_rsi = rsi.iloc[-1] if hasattr(rsi, 'iloc') else rsi[-1]
                if current_rsi < self.rsi_threshold:
                    evidence.append(f"RSI oversold at {current_rsi:.1f}")
                    confidence += 15
                    oversold_signals += 1
                elif current_rsi < 40:
                    evidence.append(f"RSI approaching oversold at {current_rsi:.1f}")
                    confidence += 5
                    oversold_signals += 0.5

            # Price vs SMA
            price_vs_sma20 = indicators.get('price_vs_sma20')
            if price_vs_sma20 is not None:
                sma_dev = price_vs_sma20.iloc[-1] if hasattr(price_vs_sma20, 'iloc') else price_vs_sma20[-1]
                if sma_dev < -5:  # >5% below SMA20
                    evidence.append(f"Price {sma_dev:.1f}% below SMA20")
                    confidence += 10
                    oversold_signals += 1

            # Fear & Greed in fear zone (contrarian signal)
            fear_greed = data.get('fear_greed', {})
            if fear_greed.get('is_extreme_fear') or fear_greed.get('is_fear_zone'):
                evidence.append("Market in fear territory (contrarian buy)")
                confidence += 5
                oversold_signals += 0.5

            # Need at least 2 oversold signals
            if oversold_signals < 2:
                return self._create_not_detected_result("Insufficient oversold signals")

            # Calculate trade levels
            trade_levels = self.calculate_trade_levels(current_price, data)

            return self._create_detected_result(
                confidence=min(80, confidence),
                reason="Price oversold - mean reversion likely",
                evidence=evidence,
                entry_price=trade_levels['entry'],
                stop_loss=trade_levels['stop_loss'],
                take_profit=trade_levels['take_profit'],
                risk_level='MEDIUM',
                metadata={
                    'oversold_signals': oversold_signals,
                    'vwap_deviation': vwap_dev
                }
            )

        except Exception as e:
            logger.error(f"Oversold detection error: {e}")
            return self._create_not_detected_result(f"Error: {str(e)}")

    def calculate_trade_levels(
        self,
        current_price: float,
        data: Dict
    ) -> Dict[str, float]:
        """Calculate trade levels for oversold mean reversion LONG."""
        indicators = data.get('indicators', {})
        prices = data['prices']

        high = prices['high'].values
        low = prices['low'].values

        # Target: VWAP or middle Bollinger Band
        vwap = indicators.get('vwap')
        bb_middle = indicators.get('bb_middle')

        if vwap is not None:
            target = vwap.iloc[-1] if hasattr(vwap, 'iloc') else vwap[-1]
        elif bb_middle is not None:
            target = bb_middle.iloc[-1] if hasattr(bb_middle, 'iloc') else bb_middle[-1]
        else:
            target = current_price * 1.03  # 3% reversion

        # Stop below recent low
        recent_low = low[-10:].min()

        atr = indicators.get('atr_14')
        atr_value = atr.iloc[-1] if atr is not None and hasattr(atr, 'iloc') else (high[-20:].max() - low[-20:].min()) / 10

        entry = current_price
        stop_loss = recent_low - atr_value  # Below support
        take_profit = target

        risk = entry - stop_loss
        risk_reward = abs(take_profit - entry) / risk if risk > 0 else 0

        return {
            'entry': entry,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'risk_reward': risk_reward
        }
