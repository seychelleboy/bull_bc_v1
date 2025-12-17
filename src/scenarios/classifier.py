"""
Scenario Classifier - Orchestrates all 5 bullish scenario detectors

Runs all scenarios and combines results with weighted probability
based on historical win rates.

Scenario weights (for LONG signals):
- Successful Breakout: 30% (highest historical performance)
- Funding Extreme: 25% (BTC-specific, high win rate)
- Bullish Divergence: 20%
- Volume Breakout: 15%
- Oversold: 10%
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict

from ..core.base_scenario import ScenarioResult
from .successful_breakout import SuccessfulBreakoutScenario
from .bullish_divergence import BullishDivergenceScenario
from .volume_breakout import VolumeBreakoutScenario
from .oversold import OversoldScenario
from .funding_extreme import FundingExtremeScenario

logger = logging.getLogger(__name__)


def _extract_scenario_config(config: Any, scenario_name: str) -> Dict:
    """
    Extract scenario config from Config dataclass or dict.

    Args:
        config: Config dataclass or dict
        scenario_name: Name of the scenario (e.g., 'successful_breakout')

    Returns:
        Dict with scenario config
    """
    if config is None:
        return {}

    # If it's a dict, return as-is for backward compatibility
    if isinstance(config, dict):
        return config

    # If it's a Config dataclass, extract the specific scenario config
    try:
        scenarios = getattr(config, 'scenarios', None)
        if scenarios is not None:
            scenario_config = getattr(scenarios, scenario_name, None)
            if scenario_config is not None:
                return asdict(scenario_config)
    except (AttributeError, TypeError):
        pass

    return {}


@dataclass
class ClassifierResult:
    """Result from scenario classification."""
    detected_scenarios: List[str] = field(default_factory=list)
    scenario_results: Dict[str, ScenarioResult] = field(default_factory=dict)
    weighted_probability: float = 0.0
    best_scenario: Optional[str] = None

    # Trade levels from best scenario
    entry_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'detected_scenarios': self.detected_scenarios,
            'weighted_probability': self.weighted_probability,
            'best_scenario': self.best_scenario,
            'entry_price': self.entry_price,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'scenario_details': {
                name: result.to_dict()
                for name, result in self.scenario_results.items()
            }
        }


class ScenarioClassifier:
    """
    Orchestrates all bullish scenario detectors.

    Runs each scenario detector and combines results using
    weighted probability based on historical win rates.

    Example:
        classifier = ScenarioClassifier()
        result = classifier.classify(data)

        if result.detected_scenarios:
            print(f"Best scenario: {result.best_scenario}")
            print(f"Probability: {result.weighted_probability:.1%}")
    """

    # Scenario weights based on historical win rates
    SCENARIO_WEIGHTS = {
        'successful_breakout': 0.30,   # 70% win rate
        'funding_extreme': 0.25,       # 65-70% win rate
        'bullish_divergence': 0.20,    # 63.6% win rate
        'volume_breakout': 0.15,       # 60% win rate
        'oversold': 0.10,              # 55-60% win rate
    }

    def __init__(self, config: Any = None):
        """
        Initialize classifier with all scenario detectors.

        Args:
            config: Optional Config dataclass or dictionary
        """
        self.config = config

        # Initialize all scenarios with their specific configs
        self.scenarios = {
            'successful_breakout': SuccessfulBreakoutScenario(
                _extract_scenario_config(config, 'successful_breakout')
            ),
            'bullish_divergence': BullishDivergenceScenario(
                _extract_scenario_config(config, 'bullish_divergence')
            ),
            'volume_breakout': VolumeBreakoutScenario(
                _extract_scenario_config(config, 'volume_breakout')
            ),
            'oversold': OversoldScenario(
                _extract_scenario_config(config, 'oversold')
            ),
            'funding_extreme': FundingExtremeScenario(
                _extract_scenario_config(config, 'funding_extreme')
            ),
        }

        logger.info(f"Scenario classifier initialized with {len(self.scenarios)} scenarios")

    def classify(self, data: Dict) -> ClassifierResult:
        """
        Run all scenario detectors and combine results.

        Args:
            data: Dictionary containing all market data

        Returns:
            ClassifierResult with detected scenarios and weighted probability
        """
        result = ClassifierResult()
        detected = []
        total_weight = 0.0
        weighted_prob = 0.0
        best_confidence = 0.0

        # Run each scenario
        for name, scenario in self.scenarios.items():
            try:
                detection = scenario.detect(data)
                result.scenario_results[name] = detection

                if detection.detected:
                    detected.append(name)
                    weight = self.SCENARIO_WEIGHTS.get(name, 0.1)
                    total_weight += weight

                    # Contribute to weighted probability
                    # Normalize confidence to 0-1 and weight by scenario weight
                    prob_contribution = (detection.confidence / 100) * weight
                    weighted_prob += prob_contribution

                    logger.info(
                        f"Scenario '{name}' DETECTED: "
                        f"confidence={detection.confidence:.1f}%, "
                        f"weight={weight:.0%}"
                    )

                    # Track best scenario
                    if detection.confidence > best_confidence:
                        best_confidence = detection.confidence
                        result.best_scenario = name
                        result.entry_price = detection.entry_price
                        result.stop_loss = detection.stop_loss
                        result.take_profit = detection.take_profit

                else:
                    logger.debug(f"Scenario '{name}': {detection.reason}")

            except Exception as e:
                logger.error(f"Error running scenario '{name}': {e}")
                continue

        result.detected_scenarios = detected

        # Normalize weighted probability
        if total_weight > 0:
            # Normalize to 0-1 range
            result.weighted_probability = min(1.0, weighted_prob / total_weight)
        else:
            result.weighted_probability = 0.0

        if detected:
            logger.info(
                f"Classification result: {len(detected)} scenarios detected, "
                f"weighted_prob={result.weighted_probability:.1%}, "
                f"best={result.best_scenario}"
            )
        else:
            logger.debug("No bullish scenarios detected")

        return result

    def get_scenario_status(self) -> Dict[str, Dict]:
        """
        Get status of all scenarios.

        Returns:
            Dictionary with scenario names and their configurations
        """
        return {
            name: {
                'weight': self.SCENARIO_WEIGHTS.get(name, 0.1),
                'win_rate': scenario.historical_win_rate,
                'config': scenario.config
            }
            for name, scenario in self.scenarios.items()
        }
