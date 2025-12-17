"""
Technical Scorer - TA indicators for LONG signals

Uses technical indicators:
- RSI: Oversold conditions (bullish for longs)
- MACD: Bullish crossovers
- OBV: Accumulation
- Bollinger Bands: Price below lower band (oversold)

Score interpretation for LONG signals:
- Higher score (65-100) = Bullish technicals = GOOD for longs
- Lower score (0-35) = Bearish technicals = BAD for longs

Note: This is the INVERSE of the SHORT bot scoring.
"""

from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class TechnicalScorer:
    """
    Technical analysis for LONG signal scoring.

    Score 0-100 where:
    - Higher score = bullish technicals (good for longs)
    - Lower score = bearish technicals (bad for longs)
    """

    # RSI thresholds
    RSI_OVERSOLD = 30
    RSI_VERY_OVERSOLD = 20
    RSI_OVERBOUGHT = 70
    RSI_VERY_OVERBOUGHT = 80

    # Bollinger Band thresholds
    BB_UPPER_EXTREME = 1.0   # Above upper band
    BB_LOWER_EXTREME = 0.0   # Below lower band

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize technical scorer.

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}

    async def score(self, features: Dict) -> float:
        """
        Score 0-100 based on technical indicators for LONG signals.

        Args:
            features: Dictionary containing technical features like
                     rsi, macd, macd_signal, obv, bb_pct_b, etc.

        Returns:
            Score 0-100 where higher = more bullish (good for LONG)
        """
        score = 50.0  # Start neutral

        # 1. RSI Analysis - Oversold = bullish for LONG
        rsi = features.get('rsi') or features.get('rsi_14')
        if rsi is not None:
            try:
                rsi = float(rsi)
                if rsi < self.RSI_VERY_OVERSOLD:
                    # Very oversold - strong bounce signal
                    score += 20
                    logger.debug(f"RSI {rsi:.1f} very oversold → +20")
                elif rsi < self.RSI_OVERSOLD:
                    # Oversold - bullish reversal likely
                    score += 12
                    logger.debug(f"RSI {rsi:.1f} oversold → +12")
                elif rsi > self.RSI_VERY_OVERBOUGHT:
                    # Very overbought - potential pullback
                    score -= 15
                elif rsi > self.RSI_OVERBOUGHT:
                    # Overbought - slight caution
                    score -= 8
                elif rsi < 40:
                    # Slightly oversold
                    score += 5
                elif rsi > 60:
                    # Bullish momentum
                    score += 3
            except (ValueError, TypeError):
                pass

        # 2. MACD Analysis - Bullish crossover = good for LONG
        macd = features.get('macd')
        macd_signal = features.get('macd_signal')
        if macd is not None and macd_signal is not None:
            try:
                macd = float(macd)
                macd_signal = float(macd_signal)

                if macd > macd_signal:
                    # Bullish crossover or above signal
                    score += 10
                    logger.debug(f"MACD bullish crossover → +10")
                elif macd < macd_signal:
                    # Bearish crossover or below signal
                    score -= 10

                # Check MACD histogram trend
                macd_hist = features.get('macd_hist')
                if macd_hist is not None:
                    macd_hist = float(macd_hist)
                    macd_hist_prev = features.get('macd_hist_prev', 0)
                    if macd_hist > 0 and macd_hist > macd_hist_prev:
                        # Histogram increasing and positive
                        score += 5
                    elif macd_hist < 0 and macd_hist > macd_hist_prev:
                        # Histogram negative but improving
                        score += 3
                    elif macd_hist < 0 and macd_hist < macd_hist_prev:
                        # Histogram decreasing and negative
                        score -= 5
            except (ValueError, TypeError):
                pass

        # 3. OBV Analysis - Accumulation = bullish
        obv = features.get('obv')
        obv_sma = features.get('obv_sma') or features.get('obv_sma_20')
        if obv is not None and obv_sma is not None:
            try:
                obv = float(obv)
                obv_sma = float(obv_sma)

                if obv > obv_sma:
                    # Accumulation - volume entering
                    score += 10
                    logger.debug(f"OBV above SMA (accumulation) → +10")
                elif obv < obv_sma:
                    # Distribution - volume leaving
                    score -= 10
            except (ValueError, TypeError):
                pass

        # 4. Bollinger Band Analysis - Below lower band = oversold
        bb_pct = features.get('bb_pct_b') or features.get('bb_pct')
        if bb_pct is not None:
            try:
                bb_pct = float(bb_pct)

                if bb_pct < self.BB_LOWER_EXTREME:
                    # Below lower band - oversold
                    score += 15
                    logger.debug(f"BB% {bb_pct:.2f} below lower band → +15")
                elif bb_pct > self.BB_UPPER_EXTREME:
                    # Above upper band - overextended
                    score -= 10
                elif bb_pct < 0.2:
                    # Near lower band
                    score += 5
                elif bb_pct > 0.8:
                    # Near upper band
                    score -= 5
            except (ValueError, TypeError):
                pass

        # 5. Price vs Moving Averages - Below MA = potential value
        price_vs_sma20 = features.get('price_vs_sma20')
        if price_vs_sma20 is not None:
            try:
                pct_below = float(price_vs_sma20)
                if pct_below < -5.0:  # >5% below SMA20
                    score += 8
                    logger.debug(f"Price {pct_below:.1f}% below SMA20 → +8")
                elif pct_below > 5.0:  # >5% above SMA20
                    score -= 5
            except (ValueError, TypeError):
                pass

        # 6. ATR Analysis (volatility regime)
        atr_pct = features.get('atr_pct') or features.get('atr_14_pct')
        if atr_pct is not None:
            try:
                atr = float(atr_pct)
                if atr > 5.0:  # High volatility
                    # High volatility = bigger moves possible
                    score += 2
            except (ValueError, TypeError):
                pass

        logger.debug(f"Technical score: {score:.1f}")
        return min(100, max(0, score))

    def score_sync(self, features: Dict) -> float:
        """
        Synchronous version of score.

        Args:
            features: Dictionary containing technical features

        Returns:
            Score 0-100
        """
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self.score(features))
                    return future.result()
            else:
                return loop.run_until_complete(self.score(features))
        except RuntimeError:
            return asyncio.run(self.score(features))
