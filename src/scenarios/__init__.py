"""
Scenarios Module - Bullish pattern detection for LONG signals

Five scenarios optimized for LONG entries:
- Successful Breakout: 70% win rate
- Funding Extreme (negative): 65-70% win rate
- Bullish Divergence: 63.6% win rate
- Volume Breakout: 60% win rate
- Oversold: 55-60% win rate
"""

from .classifier import ScenarioClassifier, ClassifierResult
from .successful_breakout import SuccessfulBreakoutScenario
from .bullish_divergence import BullishDivergenceScenario
from .volume_breakout import VolumeBreakoutScenario
from .oversold import OversoldScenario
from .funding_extreme import FundingExtremeScenario

__all__ = [
    'ScenarioClassifier',
    'ClassifierResult',
    'SuccessfulBreakoutScenario',
    'BullishDivergenceScenario',
    'VolumeBreakoutScenario',
    'OversoldScenario',
    'FundingExtremeScenario'
]
