"""
News Client - Aggregate news from multiple sources

Combines:
- EODHD News API (via EODHD client)
- CryptoPanic API (free tier)

News sentiment is used for BAIT scoring and scenario confirmation.
"""

import aiohttp
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

from ..core import ThreadSafeCache

logger = logging.getLogger(__name__)


class NewsSentiment(Enum):
    """News sentiment classification."""
    BULLISH = 'bullish'
    BEARISH = 'bearish'
    NEUTRAL = 'neutral'


@dataclass
class NewsArticle:
    """Single news article."""
    title: str
    source: str
    url: str
    published_at: datetime
    sentiment: Optional[NewsSentiment] = None
    currencies: List[str] = None
    votes: Dict[str, int] = None

    def __post_init__(self):
        if self.currencies is None:
            self.currencies = []
        if self.votes is None:
            self.votes = {}

    def to_dict(self) -> Dict:
        return {
            'title': self.title,
            'source': self.source,
            'url': self.url,
            'published_at': self.published_at.isoformat(),
            'sentiment': self.sentiment.value if self.sentiment else None,
            'currencies': self.currencies,
            'votes': self.votes
        }


class CryptoPanicClient:
    """
    CryptoPanic News API Client.

    CryptoPanic aggregates crypto news from various sources and
    provides community sentiment voting.

    API: https://cryptopanic.com/developers/api/
    Free tier: Limited to public endpoints
    """

    BASE_URL = "https://cryptopanic.com/api/v1"

    def __init__(
        self,
        auth_token: Optional[str] = None,
        cache: Optional[ThreadSafeCache] = None,
        cache_config: Optional['CacheConfig'] = None
    ):
        """
        Initialize CryptoPanic client.

        Args:
            auth_token: Optional API token (free tier works without)
            cache: Optional cache for responses
            cache_config: Optional cache configuration with TTL settings
        """
        self.auth_token = auth_token

        # Get TTL from config or use default
        default_ttl = 300  # 5 min default
        if cache_config is not None:
            default_ttl = getattr(cache_config, 'news_ttl', 300)

        self._cache_ttl = default_ttl
        self.cache = cache or ThreadSafeCache(default_ttl=default_ttl)
        self._session: Optional[aiohttp.ClientSession] = None

        logger.debug(f"CryptoPanicClient initialized with cache TTL: {default_ttl}s")

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def close(self) -> None:
        """Close the client session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def _request(
        self,
        endpoint: str,
        params: Optional[Dict] = None,
        use_cache: bool = True
    ) -> Dict:
        """
        Make an API request.

        Args:
            endpoint: API endpoint
            params: Query parameters
            use_cache: Whether to use cache

        Returns:
            JSON response
        """
        params = params or {}
        if self.auth_token:
            params['auth_token'] = self.auth_token

        url = f"{self.BASE_URL}/{endpoint}"
        cache_key = f"cp:{endpoint}:{hash(frozenset(params.items()))}"

        # Check cache
        if use_cache:
            cached = self.cache.get(cache_key)
            if cached is not None:
                logger.debug(f"CryptoPanic cache hit for {endpoint}")
                return cached

        session = await self._get_session()

        try:
            async with session.get(url, params=params) as response:
                if response.status == 429:
                    logger.warning("CryptoPanic rate limited")
                    return {'results': []}

                response.raise_for_status()
                data = await response.json()

                if use_cache:
                    self.cache.set(cache_key, data, ttl=self._cache_ttl)

                return data

        except aiohttp.ClientError as e:
            logger.error(f"CryptoPanic API error: {e}")
            return {'results': []}

    @staticmethod
    def _parse_sentiment(votes: Dict) -> Optional[NewsSentiment]:
        """Parse sentiment from vote data."""
        if not votes:
            return None

        positive = votes.get('positive', 0) + votes.get('liked', 0)
        negative = votes.get('negative', 0) + votes.get('disliked', 0)

        if positive > negative * 1.5:
            return NewsSentiment.BULLISH
        elif negative > positive * 1.5:
            return NewsSentiment.BEARISH
        else:
            return NewsSentiment.NEUTRAL

    async def get_posts(
        self,
        currencies: Optional[List[str]] = None,
        filter_type: str = 'all',
        kind: str = 'news',
        limit: int = 50
    ) -> List[NewsArticle]:
        """
        Get news posts from CryptoPanic.

        Args:
            currencies: Filter by currency symbols (e.g., ['BTC'])
            filter_type: 'rising', 'hot', 'bullish', 'bearish', 'important', 'all'
            kind: 'news' or 'media'
            limit: Maximum number of posts

        Returns:
            List of NewsArticle objects
        """
        params = {
            'kind': kind,
            'filter': filter_type,
            'public': 'true'
        }

        if currencies:
            params['currencies'] = ','.join(currencies)

        data = await self._request('posts/', params)

        articles = []
        for item in data.get('results', [])[:limit]:
            try:
                # Parse published date
                pub_date = datetime.fromisoformat(
                    item['published_at'].replace('Z', '+00:00')
                )

                # Extract currencies mentioned
                currencies_mentioned = [
                    c['code'] for c in item.get('currencies', [])
                ]

                # Parse votes for sentiment
                votes = item.get('votes', {})
                sentiment = self._parse_sentiment(votes)

                article = NewsArticle(
                    title=item.get('title', ''),
                    source=item.get('source', {}).get('title', 'Unknown'),
                    url=item.get('url', ''),
                    published_at=pub_date,
                    sentiment=sentiment,
                    currencies=currencies_mentioned,
                    votes=votes
                )
                articles.append(article)

            except (ValueError, KeyError) as e:
                logger.warning(f"Failed to parse CryptoPanic post: {e}")
                continue

        logger.info(f"Fetched {len(articles)} posts from CryptoPanic")
        return articles

    async def get_btc_news(self, limit: int = 30) -> List[NewsArticle]:
        """
        Get Bitcoin-specific news.

        Args:
            limit: Maximum number of articles

        Returns:
            List of NewsArticle objects
        """
        return await self.get_posts(currencies=['BTC'], limit=limit)

    async def get_bullish_news(self, limit: int = 20) -> List[NewsArticle]:
        """
        Get bullish-sentiment news.

        Args:
            limit: Maximum number of articles

        Returns:
            List of NewsArticle filtered for bullish sentiment
        """
        return await self.get_posts(
            currencies=['BTC'],
            filter_type='bullish',
            limit=limit
        )

    async def get_important_news(self, limit: int = 20) -> List[NewsArticle]:
        """
        Get important/impactful news.

        Args:
            limit: Maximum number of articles

        Returns:
            List of NewsArticle marked as important
        """
        return await self.get_posts(
            currencies=['BTC'],
            filter_type='important',
            limit=limit
        )


class NewsAggregator:
    """
    Aggregates news from multiple sources and provides sentiment analysis.

    Primary source: EODHD (included in subscription, has sentiment data)
    Secondary source: CryptoPanic (optional, requires API key)

    Example:
        aggregator = NewsAggregator(eodhd_client)
        news = await aggregator.get_all_btc_news()
        sentiment = aggregator.calculate_aggregate_sentiment(news)
    """

    def __init__(
        self,
        eodhd_client=None,
        cryptopanic_client: Optional[CryptoPanicClient] = None
    ):
        """
        Initialize news aggregator.

        Args:
            eodhd_client: EODHDClient instance (primary source)
            cryptopanic_client: Optional CryptoPanicClient instance
        """
        self.eodhd = eodhd_client
        self.cryptopanic = cryptopanic_client
        self._cryptopanic_enabled = self._check_cryptopanic_enabled()

    def _check_cryptopanic_enabled(self) -> bool:
        """Check if CryptoPanic has a valid API token configured."""
        if not self.cryptopanic:
            return False
        token = self.cryptopanic.auth_token
        if not token or token == 'your_cryptopanic_key_here' or len(token) < 10:
            logger.debug("CryptoPanic disabled - no valid API token")
            return False
        return True

    @staticmethod
    def _parse_eodhd_sentiment(sentiment_data: Dict) -> Optional[NewsSentiment]:
        """Parse EODHD sentiment data into NewsSentiment enum."""
        if not sentiment_data:
            return None

        # EODHD returns: {"polarity": 0.5, "neg": 0.1, "neu": 0.7, "pos": 0.2}
        polarity = sentiment_data.get('polarity', 0.5)
        pos = sentiment_data.get('pos', 0)
        neg = sentiment_data.get('neg', 0)

        # Use polarity or pos/neg ratio to determine sentiment
        if polarity > 0.6 or pos > neg * 1.5:
            return NewsSentiment.BULLISH
        elif polarity < 0.4 or neg > pos * 1.5:
            return NewsSentiment.BEARISH
        else:
            return NewsSentiment.NEUTRAL

    async def close(self) -> None:
        """Close all client sessions."""
        if self.cryptopanic:
            await self.cryptopanic.close()

    async def get_all_btc_news(self, limit: int = 50) -> List[NewsArticle]:
        """
        Get BTC news from all sources.

        Primary: EODHD (always used if available)
        Secondary: CryptoPanic (only if valid API token configured)

        Args:
            limit: Maximum articles per source

        Returns:
            Combined list of NewsArticle objects
        """
        articles = []

        # PRIMARY: Get from EODHD (included in subscription)
        if self.eodhd:
            try:
                eodhd_news = await self.eodhd.get_news(symbol='BTC-USD.CC', limit=limit)
                for item in eodhd_news:
                    try:
                        # Parse date - handle multiple formats
                        date_str = item.get('date', '')
                        if 'T' in date_str:
                            pub_date = datetime.fromisoformat(
                                date_str.replace('Z', '+00:00')
                            )
                        else:
                            # Format: "2024-01-15 12:30:00"
                            pub_date = datetime.strptime(
                                date_str, '%Y-%m-%d %H:%M:%S'
                            ).replace(tzinfo=timezone.utc)

                        # Parse EODHD sentiment
                        sentiment = self._parse_eodhd_sentiment(
                            item.get('sentiment', {})
                        )

                        article = NewsArticle(
                            title=item.get('title', ''),
                            source='EODHD',
                            url=item.get('link', ''),
                            published_at=pub_date,
                            sentiment=sentiment,
                            currencies=['BTC']
                        )
                        articles.append(article)
                    except (ValueError, KeyError) as e:
                        logger.debug(f"Failed to parse EODHD news item: {e}")
                        continue

                logger.info(f"Fetched {len(articles)} news articles from EODHD")
            except Exception as e:
                logger.warning(f"Failed to fetch EODHD news: {e}")

        # SECONDARY: Get from CryptoPanic (only if enabled)
        if self._cryptopanic_enabled:
            try:
                cp_news = await self.cryptopanic.get_btc_news(limit=limit)
                articles.extend(cp_news)
                logger.info(f"Fetched {len(cp_news)} news articles from CryptoPanic")
            except Exception as e:
                logger.warning(f"Failed to fetch CryptoPanic news: {e}")

        # Sort by date, most recent first
        articles.sort(key=lambda x: x.published_at, reverse=True)

        return articles

    def calculate_aggregate_sentiment(
        self,
        articles: List[NewsArticle],
        hours: int = 24
    ) -> Dict:
        """
        Calculate aggregate sentiment from recent articles.

        For LONG bot: Higher bullish count = more favorable

        Args:
            articles: List of NewsArticle objects
            hours: Only consider articles from last N hours

        Returns:
            Dictionary with sentiment metrics
        """
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        recent = [a for a in articles if a.published_at > cutoff]

        if not recent:
            return {
                'bullish_count': 0,
                'bearish_count': 0,
                'neutral_count': 0,
                'total_count': 0,
                'sentiment_score': 0.5,  # Neutral
                'dominant_sentiment': 'NEUTRAL',
                'long_favorable': False  # Added for LONG bot
            }

        bullish = sum(1 for a in recent if a.sentiment == NewsSentiment.BULLISH)
        bearish = sum(1 for a in recent if a.sentiment == NewsSentiment.BEARISH)
        neutral = sum(1 for a in recent if a.sentiment == NewsSentiment.NEUTRAL or a.sentiment is None)
        total = len(recent)

        # Calculate sentiment score (0 = bearish, 0.5 = neutral, 1 = bullish)
        if bullish + bearish > 0:
            sentiment_score = bullish / (bullish + bearish)
        else:
            sentiment_score = 0.5

        # Determine dominant sentiment
        if sentiment_score > 0.6:
            dominant = 'BULLISH'
        elif sentiment_score < 0.4:
            dominant = 'BEARISH'
        else:
            dominant = 'NEUTRAL'

        # For LONG bot: bullish sentiment is favorable
        long_favorable = dominant == 'BULLISH' or sentiment_score > 0.55

        return {
            'bullish_count': bullish,
            'bearish_count': bearish,
            'neutral_count': neutral,
            'total_count': total,
            'sentiment_score': round(sentiment_score, 3),
            'dominant_sentiment': dominant,
            'long_favorable': long_favorable,
            'hours_analyzed': hours
        }
