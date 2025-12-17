"""
Base Engine - Abstract base class for trading engines

Defines the lifecycle and interface for the Bull BC1 engine.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
import asyncio
import logging

logger = logging.getLogger(__name__)


@dataclass
class Signal:
    """
    Trading signal with all trade metadata.

    For Bull BC1 (LONG-only):
    - direction is always 'LONG'
    - stop_loss is BELOW entry price
    - take_profit is ABOVE entry price
    """
    symbol: str
    direction: str  # Always 'LONG' for bull bot
    timestamp: datetime
    confidence: float  # 0-100

    # Scenario that triggered the signal
    scenario: str

    # Trade levels
    entry_price: float
    stop_loss: float      # Below entry for LONG
    take_profit: float    # Above entry for LONG

    # Model predictions (0.0 for scenarios-only approach)
    lstm_prob: float = 0.0
    xgboost_prob: float = 0.0
    ensemble_prob: float = 0.0

    # Risk metrics
    risk_reward_ratio: float = 0.0
    position_size: float = 0.0
    risk_amount: float = 0.0

    # Analysis
    reasons: List[str] = field(default_factory=list)
    features: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate signal after initialization."""
        if self.direction != 'LONG':
            raise ValueError(f"Bull bot only supports LONG direction, got {self.direction}")

        # Validate stop/target for LONG
        if self.stop_loss >= self.entry_price:
            logger.warning(f"Stop loss {self.stop_loss} should be below entry {self.entry_price} for LONG")
        if self.take_profit <= self.entry_price:
            logger.warning(f"Take profit {self.take_profit} should be above entry {self.entry_price} for LONG")

    def to_dict(self) -> Dict[str, Any]:
        """Convert signal to dictionary."""
        return {
            'symbol': self.symbol,
            'direction': self.direction,
            'timestamp': self.timestamp.isoformat(),
            'confidence': self.confidence,
            'scenario': self.scenario,
            'entry_price': self.entry_price,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'lstm_prob': self.lstm_prob,
            'xgboost_prob': self.xgboost_prob,
            'ensemble_prob': self.ensemble_prob,
            'risk_reward_ratio': self.risk_reward_ratio,
            'position_size': self.position_size,
            'risk_amount': self.risk_amount,
            'reasons': self.reasons,
            'metadata': self.metadata
        }

    def __str__(self) -> str:
        return (
            f"Signal({self.direction} {self.symbol} @ ${self.entry_price:,.2f}, "
            f"confidence={self.confidence:.1f}%, R:R={self.risk_reward_ratio:.1f})"
        )


class BaseEngine(ABC):
    """
    Abstract base class for trading engines.

    Defines the lifecycle:
    1. initialize() - Setup components
    2. run_scan_cycle() - Main analysis loop
    3. generate_signals() - Create trading signals
    4. shutdown() - Cleanup resources

    Example:
        class BullEngine(BaseEngine):
            async def initialize(self):
                # Setup data sources, models, etc.
                pass

            async def run_scan_cycle(self):
                # Main analysis loop
                signals = await self.generate_signals()
                return signals
    """

    def __init__(self, config):
        """
        Initialize base engine.

        Args:
            config: Application Config object
        """
        self.config = config
        self._running = False
        self._initialized = False

    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialize all components.

        Should setup:
        - Data clients (CoinGecko, Fear & Greed, etc.)
        - Feature pipeline
        - Scenario classifiers
        - BAIT scorer
        - Risk management
        """
        pass

    @abstractmethod
    async def analyze(self) -> Optional[Signal]:
        """
        Run analysis and return signal if conditions met.

        Returns:
            Signal if bullish opportunity detected, None otherwise
        """
        pass

    @abstractmethod
    async def generate_signals(self) -> List[Signal]:
        """
        Generate trading signals from market data.

        Returns:
            List of signals (typically 0 or 1 for single-symbol bot)
        """
        pass

    @abstractmethod
    async def run_scan_cycle(self) -> List[Signal]:
        """
        Execute one scan cycle.

        Should:
        1. Fetch market data
        2. Run scenario detection
        3. Calculate BAIT score
        4. Generate signals if conditions met
        5. Execute trades if auto-execute enabled

        Returns:
            List of signals generated
        """
        pass

    @abstractmethod
    async def shutdown(self) -> None:
        """
        Cleanup all resources.

        Should close:
        - API sessions
        - Database connections
        - Background tasks
        """
        pass

    async def run(self) -> None:
        """
        Main run loop.

        Continuously scans for opportunities at configured interval.
        """
        if not self._initialized:
            await self.initialize()
            self._initialized = True

        self._running = True
        scan_interval = self.config.trading.scan_interval_seconds

        logger.info(f"Starting main loop with {scan_interval}s scan interval")

        while self._running:
            try:
                # Run scan cycle
                signals = await self.run_scan_cycle()

                if signals:
                    for signal in signals:
                        self._log_signal(signal)

                # Wait for next cycle
                await asyncio.sleep(scan_interval)

            except asyncio.CancelledError:
                logger.info("Run loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in scan cycle: {e}")
                import traceback
                logger.error(traceback.format_exc())
                await asyncio.sleep(60)  # Wait before retry

        await self.shutdown()

    def stop(self) -> None:
        """Stop the run loop."""
        self._running = False
        logger.info("Stop requested")

    def _log_signal(self, signal: Signal) -> None:
        """Log a generated signal."""
        logger.info("=" * 60)
        logger.info(f"BULLISH SIGNAL DETECTED")
        logger.info("=" * 60)
        logger.info(f"Symbol: {signal.symbol}")
        logger.info(f"Direction: {signal.direction}")
        logger.info(f"Confidence: {signal.confidence:.1f}%")
        logger.info(f"Scenario: {signal.scenario}")
        logger.info(f"Entry: ${signal.entry_price:,.2f}")
        logger.info(f"Stop Loss: ${signal.stop_loss:,.2f}")
        logger.info(f"Take Profit: ${signal.take_profit:,.2f}")
        logger.info(f"Risk:Reward: {signal.risk_reward_ratio:.1f}")
        logger.info(f"Reasons:")
        for reason in signal.reasons:
            logger.info(f"  - {reason}")
        logger.info("=" * 60)

    @property
    def is_running(self) -> bool:
        """Check if engine is running."""
        return self._running

    @property
    def is_initialized(self) -> bool:
        """Check if engine is initialized."""
        return self._initialized
