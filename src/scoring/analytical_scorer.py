"""
Analytical Scorer - BTC-specific fundamental analysis for LONG signals

Analyzes on-chain and market fundamentals:
- Funding rate (negative funding = overcrowded shorts = bullish)
- Market dominance (money flow direction)
- Price distance from ATH (value opportunity)
- 24h price change

Score interpretation for LONG signals:
- Higher score (65-100) = Bullish fundamentals = GOOD for longs
- Lower score (0-35) = Bearish fundamentals = BAD for longs
"""

from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class AnalyticalScorer:
    """
    Fundamental analysis for BTC LONG signal scoring.

    Analyzes:
    - Funding rate: Negative = overcrowded shorts (bullish reversal)
    - BTC dominance: Money flow direction
    - Distance from ATH: Value opportunity
    - Price momentum

    Score 0-100 where:
    - Higher score = bullish fundamentals (good for longs)
    - Lower score = bearish fundamentals (bad for longs)
    """

    # Funding rate thresholds
    FUNDING_EXTREME_NEGATIVE = -0.001  # <-0.1% = overcrowded shorts (bullish)
    FUNDING_EXTREME_POSITIVE = 0.001   # >0.1% = overleveraged longs (bearish)

    # Dominance thresholds
    DOMINANCE_LOW = 40.0   # Money flowing to alts
    DOMINANCE_HIGH = 55.0  # Flight to BTC safety

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize analytical scorer.

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}

    async def score(self, features: Dict) -> float:
        """
        Score 0-100 based on BTC fundamentals for LONG signals.

        Args:
            features: Dictionary containing fundamental data like
                     funding_rate, btc_dominance, etc.

        Returns:
            Score 0-100 where higher = more bullish (good for LONG)
        """
        score = 50.0  # Start neutral

        # 1. Funding Rate Analysis
        # For LONG signals: Negative funding = bullish (shorts are paying)
        funding = features.get('funding_rate')
        if funding is not None:
            try:
                funding = float(funding)
                if funding < self.FUNDING_EXTREME_NEGATIVE:
                    # Overleveraged shorts - bullish reversal likely
                    # GOOD for LONG (higher score)
                    funding_adjustment = 15 + min(15, (abs(funding) - 0.001) * 5000)
                    score += funding_adjustment
                    logger.debug(f"Negative funding {funding:.4%} → bullish adjustment +{funding_adjustment:.1f}")
                elif funding > self.FUNDING_EXTREME_POSITIVE:
                    # Overleveraged longs - potential correction
                    # Slightly negative for LONG
                    funding_adjustment = -10 - min(10, (funding - 0.001) * 5000)
                    score += funding_adjustment
                    logger.debug(f"Positive funding {funding:.4%} → adjustment {funding_adjustment:.1f}")
            except (ValueError, TypeError):
                logger.warning(f"Invalid funding_rate value: {funding}")

        # 1b. Price Change Analysis (if no funding rate)
        if funding is None:
            price_change = features.get('price_change_24h')
            if price_change is not None:
                try:
                    pct_change = float(price_change)
                    if pct_change < -5.0:  # >5% down in 24h
                        # Big dump = potential oversold bounce opportunity
                        score += 10
                        logger.debug(f"Price {pct_change:.1f}% 24h → +10 (oversold bounce)")
                    elif pct_change > 5.0:  # >5% up
                        # Big pump = momentum is bullish
                        score += 5
                        logger.debug(f"Price +{pct_change:.1f}% 24h → +5 (momentum)")
                    elif pct_change < -2.0:
                        score += 5
                    elif pct_change > 2.0:
                        score += 3
                except (ValueError, TypeError):
                    pass

        # 2. Distance from ATH Analysis
        ath_change = features.get('ath_change_percentage')
        if ath_change is not None:
            try:
                ath_change = float(ath_change)
                if ath_change < -50:
                    # >50% below ATH = significant value opportunity
                    score += 15
                    logger.debug(f"ATH change {ath_change:.1f}% → +15 (deep value)")
                elif ath_change < -30:
                    # 30-50% below ATH = good value
                    score += 10
                    logger.debug(f"ATH change {ath_change:.1f}% → +10 (value)")
                elif ath_change < -10:
                    # 10-30% below ATH = some value
                    score += 5
                elif ath_change > -5:
                    # Near ATH = less upside potential
                    score -= 5
            except (ValueError, TypeError):
                pass

        # 3. BTC Dominance Analysis
        dominance = features.get('btc_dominance')
        if dominance is not None:
            try:
                dominance = float(dominance)
                if dominance > self.DOMINANCE_HIGH:
                    # Flight to BTC = bullish for BTC
                    score += 10
                    logger.debug(f"High dominance {dominance:.1f}% → +10 (BTC strength)")
                elif dominance < self.DOMINANCE_LOW:
                    # Money flowing to alts - less BTC focus
                    score -= 5
                logger.debug(f"Dominance {dominance:.1f}% → score {score:.1f}")
            except (ValueError, TypeError):
                pass

        # 4. Is Far From ATH (value flag)
        if features.get('is_far_from_ath'):
            score += 5
            logger.debug("Far from ATH flag → +5")

        return min(100, max(0, score))

    def score_sync(self, features: Dict) -> float:
        """
        Synchronous version of score.

        Args:
            features: Dictionary containing fundamental data

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
