"""
Base Scenario - Abstract base class for scenario pattern detectors

Provides common interface and utilities for all bullish scenario detectors.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class ScenarioResult:
    """
    Result from a scenario detection.

    For Bull BC1 (LONG-only):
    - detected = True means bullish pattern found
    - entry_price is the suggested entry
    - stop_loss is BELOW entry
    - take_profit is ABOVE entry
    """
    detected: bool
    confidence: float = 0.0
    reason: str = ''
    evidence: List[str] = field(default_factory=list)

    # Trade levels (for LONG positions)
    entry_price: float = 0.0
    stop_loss: float = 0.0      # Below entry for LONG
    take_profit: float = 0.0    # Above entry for LONG

    # Risk assessment
    risk_level: str = 'MEDIUM'  # LOW, MEDIUM, HIGH
    risk_reward: float = 0.0

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'detected': self.detected,
            'confidence': self.confidence,
            'reason': self.reason,
            'evidence': self.evidence,
            'entry_price': self.entry_price,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'risk_level': self.risk_level,
            'risk_reward': self.risk_reward,
            'metadata': self.metadata
        }


class BaseScenario(ABC):
    """
    Abstract base class for bullish scenario detectors.

    Each scenario detects a specific bullish pattern:
    - Successful Breakout: Price breaks above resistance and holds
    - Bullish Divergence: Price lower low, RSI higher low
    - Volume Breakout: Price breaks above on high volume
    - Oversold: Price far below mean, RSI < 30
    - Funding Extreme: Negative funding = overcrowded shorts

    Example:
        class SuccessfulBreakoutScenario(BaseScenario):
            scenario_name = "successful_breakout"
            historical_win_rate = "70%"

            def detect(self, data):
                # Detection logic
                pass
    """

    scenario_name: str = "base"
    historical_win_rate: str = "N/A"

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize scenario detector.

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}

    @abstractmethod
    def detect(self, data: Dict) -> ScenarioResult:
        """
        Detect if the bullish pattern is present.

        Args:
            data: Dictionary containing:
                - prices: DataFrame with OHLCV
                - indicators: DataFrame with technical indicators
                - current_price: Current market price
                - btc_market: Optional BTC market metrics
                - fear_greed: Optional Fear & Greed data

        Returns:
            ScenarioResult with detection details
        """
        pass

    @abstractmethod
    def calculate_trade_levels(
        self,
        current_price: float,
        data: Dict
    ) -> Dict[str, float]:
        """
        Calculate entry, stop-loss, and take-profit levels.

        For LONG positions:
        - Entry: At or near current price
        - Stop Loss: BELOW entry (e.g., below support - ATR)
        - Take Profit: ABOVE entry (e.g., at resistance or 2:1 R:R)

        Args:
            current_price: Current market price
            data: Analysis data

        Returns:
            Dictionary with 'entry', 'stop_loss', 'take_profit', 'risk_reward'
        """
        pass

    def validate_data(self, data: Dict, required_keys: List[str]) -> bool:
        """
        Validate that required data keys are present.

        Args:
            data: Data dictionary
            required_keys: List of required keys

        Returns:
            True if all required keys present
        """
        for key in required_keys:
            if key not in data or data[key] is None:
                logger.debug(f"Missing required key: {key}")
                return False
        return True

    def _create_detected_result(
        self,
        confidence: float,
        reason: str,
        evidence: List[str],
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        risk_level: str = 'MEDIUM',
        metadata: Optional[Dict] = None
    ) -> ScenarioResult:
        """
        Create a detected result with all parameters.

        Args:
            confidence: Detection confidence (0-100)
            reason: Main reason for detection
            evidence: List of supporting evidence
            entry_price: Suggested entry price
            stop_loss: Stop loss price (below entry for LONG)
            take_profit: Take profit price (above entry for LONG)
            risk_level: Risk assessment (LOW/MEDIUM/HIGH)
            metadata: Additional metadata

        Returns:
            ScenarioResult with detected=True
        """
        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)
        risk_reward = reward / risk if risk > 0 else 0

        return ScenarioResult(
            detected=True,
            confidence=confidence,
            reason=reason,
            evidence=evidence,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_level=risk_level,
            risk_reward=risk_reward,
            metadata=metadata or {}
        )

    def _create_not_detected_result(self, reason: str = '') -> ScenarioResult:
        """
        Create a not-detected result.

        Args:
            reason: Reason why pattern was not detected

        Returns:
            ScenarioResult with detected=False
        """
        return ScenarioResult(
            detected=False,
            confidence=0.0,
            reason=reason or f"{self.scenario_name} not detected"
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(win_rate={self.historical_win_rate})"
