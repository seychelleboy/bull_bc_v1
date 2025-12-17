"""
Data module - Data fetching and storage for Bull BC1 bot

Hybrid Data Architecture:
- Primary: MT5 Pepperstone (FREE) for BTC and correlation assets
- Fallback: EODHD API ($79.99/mo) for historical/real-time/news
- Free: CoinGecko for market metrics, Fear & Greed for sentiment

Data sources:
- EODHDClient: Historical OHLCV, real-time quotes, news (PAID)
- MT5DataClient: BTC and correlation data from broker (FREE)
- CoinGeckoClient: BTC market metrics (FREE)
- FearGreedClient: Sentiment data (FREE)
- NewsAggregator: Combined news sources
- DataAggregator: Combines all sources
"""
from .database import Database, get_database
from .fear_greed_client import FearGreedClient, FearGreedReading, FearGreedLevel
from .coingecko_client import CoinGeckoClient, BTCMarketData
from .eodhd_client import EODHDClient, EODHDWebSocket, OHLCVBar, Interval, RateLimitError
from .news_client import NewsAggregator, CryptoPanicClient, NewsArticle, NewsSentiment
from .mt5_data_client import MT5DataClient
from .data_aggregator import DataAggregator, AggregatedData

__all__ = [
    # Database
    'Database',
    'get_database',
    # EODHD (PAID API)
    'EODHDClient',
    'EODHDWebSocket',
    'OHLCVBar',
    'Interval',
    'RateLimitError',
    # MT5 (FREE broker data)
    'MT5DataClient',
    # News
    'NewsAggregator',
    'CryptoPanicClient',
    'NewsArticle',
    'NewsSentiment',
    # Fear & Greed
    'FearGreedClient',
    'FearGreedReading',
    'FearGreedLevel',
    # CoinGecko
    'CoinGeckoClient',
    'BTCMarketData',
    # Aggregator
    'DataAggregator',
    'AggregatedData'
]
