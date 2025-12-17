"""
BAIT Scoring Module - Multi-factor confidence enhancement for LONG signals

BAIT = Behavioral + Analytical + Informational + Technical

Uses LONG-optimized weights:
- Behavioral: 25% (sentiment for contrarian buys)
- Analytical: 30% (fundamentals - funding, value)
- Informational: 20% (news catalysts)
- Technical: 25% (TA indicators)

Score interpretation for LONG signals:
- Higher score = more bullish = GOOD for longs
"""

from .bait_adapter import BAITScorer, BAITScore
from .behavioral_scorer import BehavioralScorer
from .analytical_scorer import AnalyticalScorer
from .informational_scorer import InformationalScorer
from .technical_scorer import TechnicalScorer

__all__ = [
    'BAITScorer',
    'BAITScore',
    'BehavioralScorer',
    'AnalyticalScorer',
    'InformationalScorer',
    'TechnicalScorer'
]
