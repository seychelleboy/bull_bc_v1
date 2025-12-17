"""
BAIT Scoring Adapter - Multi-factor scoring orchestrator for LONG signals

BAIT = Behavioral + Analytical + Informational + Technical

Uses LONG-optimized weights:
- Behavioral: 25% (sentiment for contrarian buys)
- Analytical: 30% (fundamentals - funding, value)
- Informational: 20% (news catalysts)
- Technical: 25% (TA indicators)

Score interpretation for LONG signals (INVERTED from SHORT):
- Combined score >= 75: STRONG long signal
- Combined score >= 65: MODERATE long signal
- Combined score >= 55: WEAK long signal
- Combined score < 55: AVOID (too bearish for longs)
"""

from dataclasses import dataclass, asdict
from typing import Dict, Optional, Any
import asyncio
import logging

from .behavioral_scorer import BehavioralScorer
from .analytical_scorer import AnalyticalScorer
from .informational_scorer import InformationalScorer
from .technical_scorer import TechnicalScorer

logger = logging.getLogger(__name__)


def _extract_scoring_config(config: Any) -> Dict:
    """
    Extract scoring config from Config dataclass or dict.

    Args:
        config: Config dataclass or dict

    Returns:
        Dict with scoring config
    """
    if config is None:
        return {}

    # If it's a dict, return as-is for backward compatibility
    if isinstance(config, dict):
        return config

    # If it's a Config dataclass, extract the scoring config
    try:
        scoring = getattr(config, 'scoring', None)
        if scoring is not None:
            return asdict(scoring)
    except (AttributeError, TypeError):
        pass

    return {}


@dataclass
class BAITScore:
    """BAIT scoring result with all component scores."""
    behavioral: float = 50.0      # 0-100
    analytical: float = 50.0      # 0-100
    informational: float = 50.0   # 0-100
    technical: float = 50.0       # 0-100
    combined: float = 50.0        # Weighted average
    confidence_level: str = 'NEUTRAL'  # STRONG/MODERATE/WEAK/AVOID

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'behavioral': self.behavioral,
            'analytical': self.analytical,
            'informational': self.informational,
            'technical': self.technical,
            'combined': self.combined,
            'confidence_level': self.confidence_level
        }


class BAITScorer:
    """
    Multi-factor scoring using BAIT methodology for LONG signals.

    Runs all 4 components in parallel for efficiency.
    Uses LONG-optimized weights where analytical has higher weight
    since long signals are driven by fundamentals and value.

    Score interpretation (INVERTED from SHORT bot):
    - Higher score = more bullish = GOOD for longs
    - Lower score = more bearish = BAD for longs

    Example:
        scorer = BAITScorer()
        score = await scorer.calculate_score(features, news_data)

        if score.confidence_level == 'STRONG':
            # High confidence LONG signal
            execute_long()
    """

    # LONG-optimized weights
    DEFAULT_WEIGHTS = {
        'behavioral': 0.25,      # Sentiment for contrarian buys
        'analytical': 0.30,      # Higher for longs (fundamentals matter)
        'informational': 0.20,   # News catalysts
        'technical': 0.25,       # TA indicators
    }

    # Thresholds for LONG signals (higher = more bullish = better for longs)
    STRONG_THRESHOLD = 75.0
    MODERATE_THRESHOLD = 65.0
    WEAK_THRESHOLD = 55.0

    def __init__(self, config: Any = None):
        """
        Initialize BAIT scorer with all components.

        Args:
            config: Optional Config dataclass or dict with custom weights and thresholds
        """
        # Extract scoring config from Config dataclass or use dict directly
        self.config = _extract_scoring_config(config)

        # Initialize component scorers with the extracted config
        self.behavioral_scorer = BehavioralScorer(self.config)
        self.analytical_scorer = AnalyticalScorer(self.config)
        self.informational_scorer = InformationalScorer(self.config)
        self.technical_scorer = TechnicalScorer(self.config)

        # Use config weights if provided
        self.weights = {
            'behavioral': self.config.get('behavioral_weight', self.DEFAULT_WEIGHTS['behavioral']),
            'analytical': self.config.get('analytical_weight', self.DEFAULT_WEIGHTS['analytical']),
            'informational': self.config.get('informational_weight', self.DEFAULT_WEIGHTS['informational']),
            'technical': self.config.get('technical_weight', self.DEFAULT_WEIGHTS['technical']),
        }

        # Use config thresholds if provided
        self.strong_threshold = self.config.get('strong_threshold', self.STRONG_THRESHOLD)
        self.moderate_threshold = self.config.get('moderate_threshold', self.MODERATE_THRESHOLD)
        self.weak_threshold = self.config.get('weak_threshold', self.WEAK_THRESHOLD)

        logger.info(f"BAIT Scorer (LONG) initialized with weights: {self.weights}")

    async def calculate_score(
        self,
        features: Dict,
        news_data: Optional[Dict] = None
    ) -> BAITScore:
        """
        Calculate BAIT score for LONG signal.

        Runs all 4 components in parallel for efficiency.

        Args:
            features: Technical and fundamental features dict
            news_data: News and sentiment data dict

        Returns:
            BAITScore with all component scores and confidence level
        """
        # Run all components in parallel
        try:
            behavioral, analytical, informational, technical = await asyncio.gather(
                self.behavioral_scorer.score(news_data),
                self.analytical_scorer.score(features),
                self.informational_scorer.score(news_data),
                self.technical_scorer.score(features),
                return_exceptions=True
            )

            # Handle failures with neutral score
            b = behavioral if isinstance(behavioral, (int, float)) else 50.0
            a = analytical if isinstance(analytical, (int, float)) else 50.0
            i = informational if isinstance(informational, (int, float)) else 50.0
            t = technical if isinstance(technical, (int, float)) else 50.0

            # Log any exceptions
            if isinstance(behavioral, Exception):
                logger.warning(f"Behavioral scoring failed: {behavioral}")
            if isinstance(analytical, Exception):
                logger.warning(f"Analytical scoring failed: {analytical}")
            if isinstance(informational, Exception):
                logger.warning(f"Informational scoring failed: {informational}")
            if isinstance(technical, Exception):
                logger.warning(f"Technical scoring failed: {technical}")

        except Exception as e:
            logger.error(f"BAIT scoring failed: {e}")
            b = a = i = t = 50.0

        # Calculate weighted combined score
        combined = (
            b * self.weights['behavioral'] +
            a * self.weights['analytical'] +
            i * self.weights['informational'] +
            t * self.weights['technical']
        )

        # Determine confidence level for LONG
        # Higher score = more bullish = better for LONG
        if combined >= self.strong_threshold:
            confidence_level = 'STRONG'
        elif combined >= self.moderate_threshold:
            confidence_level = 'MODERATE'
        elif combined >= self.weak_threshold:
            confidence_level = 'WEAK'
        else:
            confidence_level = 'AVOID'

        logger.info(
            f"BAIT Score (LONG): B={b:.1f} A={a:.1f} I={i:.1f} T={t:.1f} "
            f"→ Combined={combined:.1f} ({confidence_level})"
        )

        return BAITScore(
            behavioral=b,
            analytical=a,
            informational=i,
            technical=t,
            combined=combined,
            confidence_level=confidence_level
        )

    def calculate_score_sync(
        self,
        features: Dict,
        news_data: Optional[Dict] = None
    ) -> BAITScore:
        """
        Synchronous version of calculate_score.

        Args:
            features: Technical and fundamental features dict
            news_data: News and sentiment data dict

        Returns:
            BAITScore with all component scores
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Use thread pool for async-in-sync
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run,
                        self.calculate_score(features, news_data)
                    )
                    return future.result()
            else:
                return loop.run_until_complete(
                    self.calculate_score(features, news_data)
                )
        except RuntimeError:
            return asyncio.run(self.calculate_score(features, news_data))

    def get_multiplier(self, bait_score: BAITScore) -> float:
        """
        Get confidence multiplier based on BAIT score.

        For LONG signals:
        - Higher BAIT score = more bullish = higher multiplier
        - Lower BAIT score = less bullish = lower multiplier

        Args:
            bait_score: The calculated BAIT score

        Returns:
            Multiplier in range 0.5 to 1.5
        """
        # Score 75 → multiplier 1.25 (25% boost)
        # Score 50 → multiplier 1.0 (no change)
        # Score 25 → multiplier 0.75 (25% reduction)
        multiplier = 1.0 + ((bait_score.combined - 50) / 100)
        return max(0.5, min(1.5, multiplier))
