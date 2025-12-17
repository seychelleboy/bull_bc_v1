"""
Informational Scorer - News catalyst detection for LONG signals

Detects bullish and bearish catalysts in news headlines:
- Bullish catalysts: ETF approvals, institutional adoption, upgrades
- Bearish catalysts: hacks, regulations, lawsuits, crashes

Score interpretation for LONG signals:
- Higher score (65-100) = Bullish catalysts detected = GOOD for longs
- Lower score (0-35) = Bearish catalysts detected = BAD for longs
"""

from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class InformationalScorer:
    """
    News catalyst detection for BTC LONG signal scoring.

    Scans news headlines for catalyst keywords:
    - Bullish keywords increase score (good for longs)
    - Bearish keywords reduce score (bad for longs)

    Score 0-100 where:
    - Higher score = bullish catalysts present (good for longs)
    - Lower score = bearish catalysts present (bad for longs)
    """

    # Bullish catalysts (increase score - good for LONG)
    DEFAULT_BULLISH_KEYWORDS = [
        'etf', 'approval', 'adoption', 'institution', 'treasury',
        'halving', 'upgrade', 'partnership', 'integration', 'launch',
        'accumulation', 'buy', 'bullish', 'rally', 'breakthrough',
        'record', 'milestone', 'support', 'backing', 'investment',
        'inflow', 'reserve', 'holdings', 'strategic', 'allocate',
        'growth', 'surge', 'soar', 'recover', 'rebound'
    ]

    # Bearish catalysts (decrease score - bad for LONG)
    DEFAULT_BEARISH_KEYWORDS = [
        'hack', 'exploit', 'bankruptcy', 'crash', 'sec', 'lawsuit',
        'ban', 'regulation', 'investigation', 'fraud', 'ponzi',
        'sell-off', 'dump', 'liquidation', 'delisting', 'audit',
        'subpoena', 'violation', 'charges', 'breach', 'stolen',
        'collapse', 'insolvency', 'default', 'shutdown', 'suspend'
    ]

    # High-impact catalyst weights
    HIGH_IMPACT_BULLISH = ['etf', 'approval', 'halving', 'institutional', 'treasury']
    HIGH_IMPACT_BEARISH = ['hack', 'bankruptcy', 'sec', 'fraud', 'crash']

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize informational scorer.

        Args:
            config: Optional configuration with custom keywords
        """
        self.config = config or {}

        # Use config keywords if provided, otherwise defaults
        self.bullish_keywords = self.config.get(
            'bullish_keywords', self.DEFAULT_BULLISH_KEYWORDS
        )
        self.bearish_keywords = self.config.get(
            'bearish_keywords', self.DEFAULT_BEARISH_KEYWORDS
        )

    async def score(self, news_data: Optional[Dict]) -> float:
        """
        Score 0-100 based on news catalysts for LONG signals.

        Args:
            news_data: Dictionary containing 'headlines' list

        Returns:
            Score 0-100 where higher = bullish catalysts (good for LONG)
        """
        if not news_data:
            logger.debug("No news data available, returning neutral score")
            return 50.0

        headlines = news_data.get('headlines', [])
        if not headlines:
            # Try alternative key names
            headlines = news_data.get('news', [])
            if isinstance(headlines, list) and headlines:
                # Extract titles if news is list of dicts
                if isinstance(headlines[0], dict):
                    headlines = [n.get('title', '') for n in headlines]

        if not headlines:
            return 50.0

        score = 50.0
        bullish_count = 0
        bearish_count = 0
        high_impact_found = False

        # Analyze recent headlines (limit to 15)
        for headline in headlines[:15]:
            if not isinstance(headline, str):
                continue

            text = headline.lower()

            # Check bullish keywords first (priority for LONG bot)
            bullish_found = False
            for keyword in self.bullish_keywords:
                if keyword in text:
                    # High-impact keywords have more effect
                    if keyword in self.HIGH_IMPACT_BULLISH:
                        score += 8
                        high_impact_found = True
                        logger.debug(f"High-impact bullish: '{keyword}' in '{headline[:50]}...'")
                    else:
                        score += 4
                    bullish_count += 1
                    bullish_found = True
                    break  # Only count once per headline

            # Check bearish keywords
            if not bullish_found:
                for keyword in self.bearish_keywords:
                    if keyword in text:
                        if keyword in self.HIGH_IMPACT_BEARISH:
                            score -= 8
                            high_impact_found = True
                            logger.debug(f"High-impact bearish: '{keyword}' in '{headline[:50]}...'")
                        else:
                            score -= 4
                        bearish_count += 1
                        break

        logger.debug(f"Informational: {bullish_count} bullish, {bearish_count} bearish → score {score:.1f}")

        return min(100, max(0, score))

    def score_sync(self, news_data: Optional[Dict]) -> float:
        """
        Synchronous version of score.

        Args:
            news_data: Dictionary containing news headlines

        Returns:
            Score 0-100
        """
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self.score(news_data))
                    return future.result()
            else:
                return loop.run_until_complete(self.score(news_data))
        except RuntimeError:
            return asyncio.run(self.score(news_data))
