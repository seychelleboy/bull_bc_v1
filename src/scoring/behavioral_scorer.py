"""
Behavioral Scorer - News sentiment analysis for LONG signals

For BTC LONG signals:
- Uses Fear & Greed Index
- Uses news sentiment
- Weight recent news higher

Score interpretation for LONG signals:
- Higher score (65-100) = Bullish sentiment = GOOD for longs
- Lower score (0-35) = Bearish sentiment = BAD for longs

Note: This is the INVERSE of the SHORT bot scoring.
"""

from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class BehavioralScorer:
    """
    Sentiment analysis from news sources for LONG signal scoring.

    Uses Fear & Greed Index as primary sentiment indicator.
    Score 0-100 where:
    - Higher score = more bullish sentiment (good for longs)
    - Lower score = more bearish sentiment (bad for longs)

    For LONG signals, we INVERT the F&G interpretation:
    - Extreme fear = potential buying opportunity (contrarian)
    - Moderate readings with improving trend = good
    - Extreme greed = caution (but not bearish)
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize behavioral scorer.

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}

    async def score(self, news_data: Optional[Dict]) -> float:
        """
        Score 0-100 based on news sentiment for LONG signals.

        Args:
            news_data: Dictionary containing fear_greed_index and optionally
                      news sentiment scores

        Returns:
            Score 0-100 where higher = more bullish (good for LONG)
        """
        if not news_data:
            logger.debug("No news data available, returning neutral score")
            return 50.0

        score = 50.0  # Start neutral

        # Primary: Fear & Greed Index
        fear_greed = news_data.get('fear_greed_index')
        if fear_greed is None:
            # Try alternative key names
            fg = news_data.get('fear_greed', {})
            if isinstance(fg, dict):
                fear_greed = fg.get('current')

        if fear_greed is not None:
            try:
                fear_greed = float(fear_greed)

                # For LONG signals, interpret F&G differently:
                # - Extreme fear (0-25): Good buying opportunity (contrarian) -> score 60-75
                # - Fear (26-45): Decent buying zone -> score 55-65
                # - Neutral (46-55): Neutral -> score 50-55
                # - Greed (56-75): Trend is bullish -> score 55-65
                # - Extreme greed (76-100): Caution but still bullish -> score 45-55

                if fear_greed <= 25:
                    # Extreme fear = contrarian buy signal
                    score = 60 + ((25 - fear_greed) / 25) * 15  # 60-75
                    logger.debug(f"Extreme fear {fear_greed} → bullish contrarian score {score:.1f}")
                elif fear_greed <= 45:
                    # Fear zone = good accumulation
                    score = 55 + ((45 - fear_greed) / 20) * 10  # 55-65
                    logger.debug(f"Fear zone {fear_greed} → score {score:.1f}")
                elif fear_greed <= 55:
                    # Neutral
                    score = 50 + (fear_greed - 45) * 0.5  # 50-55
                elif fear_greed <= 75:
                    # Greed = bullish trend
                    score = 55 + ((fear_greed - 55) / 20) * 10  # 55-65
                    logger.debug(f"Greed trend {fear_greed} → bullish score {score:.1f}")
                else:
                    # Extreme greed = caution but not bearish
                    score = 55 - ((fear_greed - 75) / 25) * 10  # 45-55
                    logger.debug(f"Extreme greed {fear_greed} → cautious score {score:.1f}")

            except (ValueError, TypeError):
                logger.warning(f"Invalid fear_greed value: {fear_greed}")

        # Secondary: News sentiment (if available)
        news_sentiment = news_data.get('news_sentiment')
        if news_sentiment is not None:
            try:
                # EODHD sentiment is typically -1 to 1
                # Convert to 0-100 scale and blend
                sentiment_score = (float(news_sentiment) + 1) * 50
                # Blend: 70% fear_greed based, 30% news sentiment
                score = score * 0.7 + sentiment_score * 0.3
                logger.debug(f"Blended with news sentiment: {score:.1f}")
            except (ValueError, TypeError):
                pass

        # Check for long_favorable flag (from Fear & Greed analysis)
        if news_data.get('long_favorable'):
            score = min(100, score + 5)
            logger.debug(f"Long favorable conditions detected, boosted to {score:.1f}")

        return min(100, max(0, score))

    def score_sync(self, news_data: Optional[Dict]) -> float:
        """
        Synchronous version of score for non-async contexts.

        Args:
            news_data: Dictionary containing sentiment data

        Returns:
            Score 0-100
        """
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Already in async context, create task
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self.score(news_data))
                    return future.result()
            else:
                return loop.run_until_complete(self.score(news_data))
        except RuntimeError:
            return asyncio.run(self.score(news_data))
