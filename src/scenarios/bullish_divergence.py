"""
Bullish Divergence Scenario - 63.6% historical win rate

Detects when price makes lower lows but indicators make higher lows,
signaling strengthening momentum and potential reversal upward.

Key signals:
1. Price making lower low
2. RSI making higher low (bullish divergence)
3. OBV/Volume showing strength
4. MACD improving

This is the INVERSE of the bear bot's Bearish Divergence scenario.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

from ..core.base_scenario import BaseScenario, ScenarioResult

logger = logging.getLogger(__name__)


class BullishDivergenceScenario(BaseScenario):
    """
    Bullish Divergence Pattern Detector.

    This scenario has historically achieved ~63.6% win rate for long entries.

    Detection criteria:
    1. Price made a lower low relative to recent swing low
    2. RSI made a higher low (bullish divergence)
    3. OBV showing strength (not confirming price weakness)
    4. Optional: MACD histogram improving

    For LONG positions:
    - Entry: Current price (market order)
    - Stop: Below recent swing low - ATR
    - Target: Resistance level or 2:1 R:R
    """

    scenario_name = "bullish_divergence"
    historical_win_rate = "63.6%"

    def __init__(self, config=None):
        """
        Initialize bullish divergence detector.

        Args:
            config: Optional configuration with thresholds
        """
        super().__init__(config)

        # Default thresholds
        self.lookback_period = config.get('lookback_period', 20) if config else 20
        self.divergence_lookback = config.get('divergence_lookback', 14) if config else 14
        self.rsi_max_threshold = config.get('rsi_max_threshold', 45) if config else 45
        self.price_low_tolerance = config.get('price_low_tolerance', 0.005) if config else 0.005  # 0.5%

    def detect(self, data: Dict) -> ScenarioResult:
        """
        Detect if bullish divergence pattern is present.

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
            if len(prices) < self.lookback_period + self.divergence_lookback:
                return self._create_not_detected_result("Insufficient data")

            low = prices['low'].values
            close = prices['close'].values

            # Check for RSI divergence
            rsi = indicators.get('rsi_14')
            if rsi is None:
                return self._create_not_detected_result("RSI indicator not available")

            rsi_values = rsi.values if hasattr(rsi, 'values') else np.array(rsi)

            # Detect divergence
            divergence_found, divergence_info = self._detect_price_rsi_divergence(
                low, rsi_values
            )

            if not divergence_found:
                return self._create_not_detected_result(
                    divergence_info.get('reason', 'No bullish divergence')
                )

            # Build evidence and confidence
            evidence = []
            confidence = 50.0

            # RSI divergence confirmed
            evidence.append(
                f"RSI divergence: Price low ${divergence_info['price_low']:,.0f}, "
                f"RSI {divergence_info['rsi_at_price_low']:.1f} vs {divergence_info['rsi_at_prev_low']:.1f}"
            )
            confidence += 20

            # Check RSI level (should be in oversold/neutral zone)
            current_rsi = rsi_values[-1]
            if current_rsi < self.rsi_max_threshold:
                evidence.append(f"RSI in favorable zone at {current_rsi:.1f}")
                confidence += 10

            # OBV divergence check
            obv = indicators.get('obv')
            if obv is not None:
                obv_values = obv.values if hasattr(obv, 'values') else np.array(obv)
                obv_div, obv_info = self._detect_obv_divergence(low, obv_values)
                if obv_div:
                    evidence.append("OBV confirming bullish divergence")
                    confidence += 15

            # MACD improvement
            macd_hist = indicators.get('macd_hist')
            if macd_hist is not None:
                macd_values = macd_hist.values if hasattr(macd_hist, 'values') else np.array(macd_hist)
                if len(macd_values) >= 3:
                    if macd_values[-1] > macd_values[-2] > macd_values[-3]:
                        evidence.append("MACD histogram improving")
                        confidence += 10

            # Price position relative to recent range (near support = good entry)
            recent_low = low[-self.lookback_period:].min()
            if current_price <= recent_low * 1.02:
                evidence.append("Price near recent lows (prime long entry)")
                confidence += 5

            # Calculate trade levels
            trade_levels = self.calculate_trade_levels(current_price, data)

            return self._create_detected_result(
                confidence=min(90, confidence),
                reason="Bullish RSI divergence detected",
                evidence=evidence,
                entry_price=trade_levels['entry'],
                stop_loss=trade_levels['stop_loss'],
                take_profit=trade_levels['take_profit'],
                risk_level=self._assess_risk(confidence, trade_levels),
                metadata={
                    'divergence_type': 'RSI',
                    'price_low': divergence_info['price_low'],
                    'rsi_at_low': divergence_info['rsi_at_price_low'],
                    'current_rsi': current_rsi
                }
            )

        except Exception as e:
            logger.error(f"Bullish divergence detection error: {e}")
            return self._create_not_detected_result(f"Detection error: {str(e)}")

    def calculate_trade_levels(
        self,
        current_price: float,
        data: Dict
    ) -> Dict[str, float]:
        """
        Calculate entry, stop-loss, and take-profit levels for LONG.

        For bullish divergence:
        - Entry: Near current price
        - Stop: Below the divergence low - buffer
        - Target: Resistance or 2:1 R:R minimum
        """
        prices = data['prices']
        indicators = data.get('indicators', {})

        high = prices['high'].values
        low = prices['low'].values

        # Recent swing low for stop
        recent_low = low[-self.lookback_period:].min()

        # Resistance level
        resistance = high[-self.lookback_period:].max()

        # ATR for buffer
        atr = indicators.get('atr_14')
        if atr is not None:
            atr_value = atr.iloc[-1] if hasattr(atr, 'iloc') else atr[-1]
        else:
            atr_value = (high[-20:].max() - low[-20:].min()) / 10

        entry = current_price
        stop_loss = recent_low - atr_value  # Below support
        risk = entry - stop_loss

        # Target: 2:1 R:R or resistance
        take_profit_rr = entry + (2.0 * risk)
        take_profit = min(take_profit_rr, resistance) if resistance > entry else take_profit_rr

        risk_reward = abs(take_profit - entry) / risk if risk > 0 else 0

        return {
            'entry': entry,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'risk_reward': risk_reward,
            'atr': atr_value
        }

    def _detect_price_rsi_divergence(
        self,
        low: np.ndarray,
        rsi: np.ndarray
    ) -> Tuple[bool, Dict]:
        """
        Detect bullish divergence between price and RSI.

        Bullish divergence: Price makes lower low, RSI makes higher low.

        Returns:
            Tuple of (divergence_found, info_dict)
        """
        lookback = self.divergence_lookback
        info = {'reason': 'No divergence found'}

        if len(low) < lookback + 5 or len(rsi) < lookback + 5:
            info['reason'] = 'Insufficient data for divergence detection'
            return False, info

        # Find price troughs in recent data
        recent_low = low[-lookback:]
        recent_rsi = rsi[-lookback:]

        # Find the two most recent swing lows in price
        troughs = self._find_troughs(recent_low)

        if len(troughs) < 2:
            info['reason'] = 'Not enough price troughs for divergence'
            return False, info

        # Get the two most recent troughs
        trough1_idx = troughs[-2]  # Earlier trough
        trough2_idx = troughs[-1]  # Later trough

        price_at_trough1 = recent_low[trough1_idx]
        price_at_trough2 = recent_low[trough2_idx]
        rsi_at_trough1 = recent_rsi[trough1_idx]
        rsi_at_trough2 = recent_rsi[trough2_idx]

        # Bullish divergence: Price lower low, RSI higher low
        price_lower = price_at_trough2 < price_at_trough1 * (1 + self.price_low_tolerance)
        rsi_higher = rsi_at_trough2 > rsi_at_trough1 + 2  # At least 2 points higher

        if price_lower and rsi_higher:
            return True, {
                'price_low': price_at_trough2,
                'prev_price_low': price_at_trough1,
                'rsi_at_price_low': rsi_at_trough2,
                'rsi_at_prev_low': rsi_at_trough1,
                'divergence_strength': rsi_at_trough2 - rsi_at_trough1
            }

        if not price_lower:
            info['reason'] = 'Price not making lower low'
        elif not rsi_higher:
            info['reason'] = 'RSI not making higher low'

        return False, info

    def _detect_obv_divergence(
        self,
        low: np.ndarray,
        obv: np.ndarray
    ) -> Tuple[bool, Dict]:
        """Detect bullish divergence in OBV."""
        lookback = self.divergence_lookback

        if len(low) < lookback or len(obv) < lookback:
            return False, {}

        # Simple check: price trending down, OBV trending up
        price_slope = np.polyfit(range(lookback), low[-lookback:], 1)[0]
        obv_slope = np.polyfit(range(lookback), obv[-lookback:], 1)[0]

        # Normalize slopes
        price_slope_pct = price_slope / low[-lookback:].mean()
        obv_slope_pct = obv_slope / (abs(obv[-lookback:].mean()) + 1)

        # Bullish divergence: price down, OBV up
        if price_slope_pct < -0.001 and obv_slope_pct > 0.001:
            return True, {
                'price_trend': 'down',
                'obv_trend': 'up'
            }

        return False, {}

    def _find_troughs(self, data: np.ndarray, min_distance: int = 3) -> List[int]:
        """Find local troughs (lows) in data."""
        troughs = []
        for i in range(1, len(data) - 1):
            if data[i] < data[i-1] and data[i] < data[i+1]:
                if not troughs or i - troughs[-1] >= min_distance:
                    troughs.append(i)
        return troughs

    def _assess_risk(self, confidence: float, trade_levels: Dict) -> str:
        """Assess trade risk level."""
        rr = trade_levels.get('risk_reward', 0)

        if confidence >= 75 and rr >= 2.0:
            return 'LOW'
        elif confidence >= 60 and rr >= 1.5:
            return 'MEDIUM'
        else:
            return 'HIGH'
