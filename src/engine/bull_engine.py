"""
Bull Engine - Main orchestration engine for LONG-only trading

Coordinates all components:
- Data aggregation (CoinGecko, Fear & Greed)
- Scenario detection (5 bullish scenarios)
- BAIT scoring (multi-factor confidence)
- Risk management (position sizing, stops)
- Signal generation

This is the LONG-only version of the bear_bc_1 engine.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

from ..core import BaseEngine, Signal, ThreadSafeCache
from ..data import DataAggregator, get_database
from ..scenarios import ScenarioClassifier
from ..scoring import BAITScorer, BAITScore
from ..risk import PositionSizer, StopCalculator
from ..execution import MT5Connector

logger = logging.getLogger(__name__)


class BullEngine(BaseEngine):
    """
    Bull Engine - LONG-only trading engine for BTC.

    Orchestrates:
    1. Data collection from MT5/EODHD, CoinGecko and Fear & Greed
    2. Technical indicator calculation
    3. Bullish scenario detection
    4. BAIT multi-factor scoring
    5. Position sizing and risk management
    6. Signal generation with confidence filtering

    Example:
        engine = BullEngine(config)
        await engine.initialize()
        await engine.run()  # Continuous scanning
    """

    def __init__(self, config):
        """
        Initialize bull engine.

        Args:
            config: Application Config object
        """
        super().__init__(config)
        self.config = config  # Store config reference

        # Will be initialized in initialize()
        self.cache: Optional[ThreadSafeCache] = None
        self.data_aggregator: Optional[DataAggregator] = None
        self.scenario_classifier: Optional[ScenarioClassifier] = None
        self.bait_scorer: Optional[BAITScorer] = None
        self.position_sizer: Optional[PositionSizer] = None
        self.stop_calculator: Optional[StopCalculator] = None
        self.mt5_connector: Optional[MT5Connector] = None
        self.database = None

        # State
        self._last_signal: Optional[Signal] = None
        self._signals_today = 0
        self._daily_pnl = 0.0

    async def initialize(self) -> None:
        """Initialize all components."""
        logger.info("Initializing Bull Engine...")

        try:
            # Initialize cache
            self.cache = ThreadSafeCache(default_ttl=self.config.data.cache_ttl)

            # Initialize data aggregator with config
            self.data_aggregator = DataAggregator.from_config(self.config)
            await self.data_aggregator.initialize()

            # Initialize scenario classifier with config
            self.scenario_classifier = ScenarioClassifier(self.config)

            # Initialize BAIT scorer (LONG-optimized) with config
            self.bait_scorer = BAITScorer(self.config)

            # Initialize risk management
            self.position_sizer = PositionSizer(
                account_balance=self.config.risk.initial_balance,
                risk_percent=self.config.risk.risk_per_trade_percent,
                max_leverage=self.config.risk.max_leverage
            )

            self.stop_calculator = StopCalculator(
                atr_multiplier=self.config.trading.initial_stop_atr_multiplier,
                risk_reward=self.config.risk.min_reward_risk_ratio
            )

            # Initialize MT5 connector for execution
            self.mt5_connector = MT5Connector(self.config)
            await self.mt5_connector.initialize()

            # Initialize database
            self.database = get_database(self.config.database.path)

            self._initialized = True
            logger.info("Bull Engine initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Bull Engine: {e}")
            raise

    async def analyze(self) -> Optional[Signal]:
        """
        Run analysis and return signal if bullish opportunity detected.

        Returns:
            Signal if conditions met, None otherwise
        """
        signals = await self.generate_signals()
        return signals[0] if signals else None

    async def generate_signals(self) -> List[Signal]:
        """
        Generate LONG trading signals from market data.

        Steps:
        1. Fetch aggregated market data
        2. Calculate technical indicators
        3. Run scenario detection
        4. Calculate BAIT score
        5. Apply confidence filter
        6. Calculate position size

        Returns:
            List of signals (typically 0 or 1)
        """
        signals = []

        try:
            # Step 1: Fetch market data
            aggregated_data = await self.data_aggregator.get_analysis_data(
                lookback_days=self.config.data.lookback_days
            )

            if not aggregated_data.is_valid():
                logger.warning("Invalid or insufficient market data")
                return signals

            current_price = aggregated_data.current_price
            logger.info(f"Current BTC price: ${current_price:,.2f}")

            # Step 2: Calculate technical indicators
            indicators = self._calculate_indicators(aggregated_data)

            # Step 3: Prepare data for scenarios
            scenario_data = {
                'prices': aggregated_data.prices_4h,
                'indicators': indicators,
                'current_price': current_price,
                'btc_market': aggregated_data.btc_market,
                'fear_greed': aggregated_data.fear_greed,
                'funding_rate': aggregated_data.btc_market.get('funding_rate')
            }

            # Step 4: Run scenario detection
            scenario_result = self.scenario_classifier.classify(scenario_data)

            if not scenario_result.detected_scenarios:
                logger.debug("No bullish scenarios detected")
                return signals

            logger.info(
                f"Detected scenarios: {scenario_result.detected_scenarios}, "
                f"weighted_prob={scenario_result.weighted_probability:.1%}"
            )

            # Step 5: Calculate BAIT score
            bait_features = self._prepare_bait_features(
                aggregated_data, indicators
            )
            news_data = self._prepare_news_data(aggregated_data)

            bait_score = await self.bait_scorer.calculate_score(
                bait_features, news_data
            )

            logger.info(
                f"BAIT Score: {bait_score.combined:.1f} ({bait_score.confidence_level})"
            )

            # Step 6: Calculate confidence
            base_confidence = scenario_result.weighted_probability * 100

            # Apply BAIT multiplier
            bait_multiplier = self.bait_scorer.get_multiplier(bait_score)
            final_confidence = min(100, base_confidence * bait_multiplier)

            logger.info(
                f"Confidence: base={base_confidence:.1f}%, "
                f"BAIT_mult={bait_multiplier:.2f}, "
                f"final={final_confidence:.1f}%"
            )

            # Step 7: Apply confidence threshold
            if final_confidence < self.config.trading.min_confidence_threshold:
                logger.info(
                    f"Confidence {final_confidence:.1f}% below threshold "
                    f"{self.config.trading.min_confidence_threshold}%"
                )
                return signals

            # Step 8: Get trade levels from best scenario
            entry_price = scenario_result.entry_price or current_price
            stop_loss = scenario_result.stop_loss
            take_profit = scenario_result.take_profit

            # Validate levels for LONG
            if stop_loss >= entry_price:
                logger.warning("Invalid stop loss for LONG, recalculating...")
                atr = self._get_atr(indicators)
                levels = self.stop_calculator.calculate(
                    entry_price, atr
                )
                stop_loss = levels.stop_loss
                take_profit = levels.take_profit

            # Step 9: Calculate position size
            position = self.position_sizer.calculate(
                entry_price=entry_price,
                stop_loss=stop_loss,
                confidence=final_confidence
            )

            # Calculate risk:reward
            risk = entry_price - stop_loss
            reward = take_profit - entry_price
            risk_reward = reward / risk if risk > 0 else 0

            # Check minimum R:R
            if risk_reward < self.config.risk.min_reward_risk_ratio:
                logger.info(
                    f"R:R {risk_reward:.1f} below minimum "
                    f"{self.config.risk.min_reward_risk_ratio}"
                )
                return signals

            # Step 10: Create signal
            signal = Signal(
                symbol=self.config.trading.symbol,
                direction='LONG',
                timestamp=datetime.now(timezone.utc),
                confidence=final_confidence,
                scenario=scenario_result.best_scenario or 'multiple',
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                lstm_prob=0.0,
                xgboost_prob=0.0,
                ensemble_prob=scenario_result.weighted_probability,
                risk_reward_ratio=risk_reward,
                position_size=position.size,
                risk_amount=position.risk_amount,
                reasons=[
                    f"{s}: detected" for s in scenario_result.detected_scenarios
                ],
                metadata={
                    'bait_score': bait_score.to_dict(),
                    'detected_scenarios': scenario_result.detected_scenarios
                }
            )

            signals.append(signal)

            # Save signal to database
            self._save_signal(signal)

            self._last_signal = signal
            self._signals_today += 1

        except Exception as e:
            logger.error(f"Error generating signals: {e}")
            import traceback
            logger.error(traceback.format_exc())

        return signals

    async def run_scan_cycle(self) -> List[Signal]:
        """
        Execute one scan cycle.

        Returns:
            List of signals generated
        """
        logger.info("=" * 40)
        logger.info("Starting scan cycle...")

        try:
            # Get quick price first
            price = await self.data_aggregator.get_current_price()
            if price <= 0:
                logger.warning("Failed to get price, skipping cycle")
                return []

            logger.info(f"BTC Price: ${price:,.2f}")

            # Generate signals
            signals = await self.generate_signals()

            if signals:
                logger.info(f"Generated {len(signals)} signal(s)")

                # Auto-execute if enabled
                if self.config.trading.auto_execute:
                    for signal in signals:
                        await self._execute_signal(signal)

            return signals

        except Exception as e:
            logger.error(f"Scan cycle error: {e}")
            return []

    async def shutdown(self) -> None:
        """Cleanup all resources."""
        logger.info("Shutting down Bull Engine...")

        if self.data_aggregator:
            await self.data_aggregator.close()

        if self.mt5_connector:
            await self.mt5_connector.shutdown()

        self._running = False
        logger.info("Bull Engine shutdown complete")

    def _calculate_indicators(self, data) -> Dict[str, Any]:
        """Calculate technical indicators from price data."""
        import pandas as pd
        import numpy as np

        prices = data.prices_4h
        if prices.empty:
            return {}

        indicators = pd.DataFrame(index=prices.index)

        try:
            close = prices['close']
            high = prices['high']
            low = prices['low']
            volume = prices.get('volume', pd.Series([0] * len(prices)))

            # RSI
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            indicators['rsi_14'] = 100 - (100 / (1 + rs))

            # MACD
            ema12 = close.ewm(span=12, adjust=False).mean()
            ema26 = close.ewm(span=26, adjust=False).mean()
            indicators['macd'] = ema12 - ema26
            indicators['macd_signal'] = indicators['macd'].ewm(span=9, adjust=False).mean()
            indicators['macd_hist'] = indicators['macd'] - indicators['macd_signal']

            # Bollinger Bands
            sma20 = close.rolling(window=20).mean()
            std20 = close.rolling(window=20).std()
            indicators['bb_upper'] = sma20 + (std20 * 2)
            indicators['bb_middle'] = sma20
            indicators['bb_lower'] = sma20 - (std20 * 2)
            indicators['bb_pct_b'] = (close - indicators['bb_lower']) / (indicators['bb_upper'] - indicators['bb_lower'])

            # ATR
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            indicators['atr_14'] = tr.rolling(window=14).mean()

            # OBV
            obv = (volume * np.sign(close.diff())).cumsum()
            indicators['obv'] = obv
            indicators['obv_sma'] = obv.rolling(window=20).mean()

            # VWAP (simple approximation)
            typical_price = (high + low + close) / 3
            indicators['vwap'] = (typical_price * volume).cumsum() / volume.cumsum()

            # Price vs indicators
            indicators['price_vs_sma20'] = ((close - sma20) / sma20) * 100
            indicators['price_vs_vwap'] = ((close - indicators['vwap']) / indicators['vwap']) * 100

            # Volume ratio
            indicators['volume_ratio'] = volume / volume.rolling(window=20).mean()

        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")

        return indicators

    def _prepare_bait_features(self, data, indicators) -> Dict:
        """Prepare features for BAIT scoring."""
        features = {}

        # Technical features
        if indicators is not None and len(indicators) > 0:
            last_row = indicators.iloc[-1] if hasattr(indicators, 'iloc') else indicators
            features['rsi'] = last_row.get('rsi_14')
            features['macd'] = last_row.get('macd')
            features['macd_signal'] = last_row.get('macd_signal')
            features['macd_hist'] = last_row.get('macd_hist')
            features['obv'] = last_row.get('obv')
            features['obv_sma'] = last_row.get('obv_sma')
            features['bb_pct_b'] = last_row.get('bb_pct_b')
            features['price_vs_sma20'] = last_row.get('price_vs_sma20')

        # Fundamental features
        if data.btc_market:
            features['funding_rate'] = data.btc_market.get('funding_rate')
            features['btc_dominance'] = data.btc_market.get('btc_dominance')
            features['price_change_24h'] = data.btc_market.get('price_change_24h')
            features['ath_change_percentage'] = data.btc_market.get('ath_change_percentage')
            features['is_far_from_ath'] = data.btc_market.get('is_far_from_ath')

        return features

    def _prepare_news_data(self, data) -> Dict:
        """Prepare news data for BAIT scoring."""
        news_data = {}

        # Fear & Greed
        if data.fear_greed:
            fg = data.fear_greed
            news_data['fear_greed_index'] = fg.get('current', 50)
            news_data['fear_greed'] = fg
            news_data['long_favorable'] = fg.get('long_favorable', False)

        # News headlines (if available)
        if data.news_sentiment:
            news_data['headlines'] = data.news_sentiment.get('headlines', [])

        return news_data

    def _get_atr(self, indicators) -> float:
        """Get ATR value from indicators."""
        if indicators is not None and 'atr_14' in indicators:
            atr = indicators['atr_14'].iloc[-1]
            if not pd.isna(atr):
                return atr
        return 100  # Default fallback

    def _save_signal(self, signal: Signal) -> None:
        """Save signal to database."""
        try:
            if self.database:
                self.database.save_signal(signal.to_dict())
        except Exception as e:
            logger.error(f"Failed to save signal: {e}")

    async def _execute_signal(self, signal: Signal) -> None:
        """Execute a trading signal via MT5."""
        if self.config.trading.paper_trade:
            logger.info(f"[PAPER TRADE] Would execute: {signal}")
            # Save as paper trade
            if self.database:
                self.database.save_trade({
                    'symbol': signal.symbol,
                    'direction': signal.direction,
                    'entry_price': signal.entry_price,
                    'stop_loss': signal.stop_loss,
                    'take_profit': signal.take_profit,
                    'position_size': signal.position_size,
                    'entry_time': signal.timestamp.isoformat(),
                    'is_paper': True
                })
        else:
            logger.info(f"[LIVE] Executing: {signal}")

            # Execute via MT5
            if not self.mt5_connector:
                logger.error("MT5 connector not initialized!")
                return

            try:
                # Get MT5 symbol from config
                mt5_symbol = self.config.mt5.btc_symbol  # e.g., 'BTCUSD_SB'

                # Place the order
                result = await self.mt5_connector.place_order(
                    symbol=mt5_symbol,
                    order_type='BUY',  # LONG = BUY
                    volume=signal.position_size,
                    sl=signal.stop_loss,
                    tp=signal.take_profit,
                    comment=f'bull_bc1_{signal.scenario}'
                )

                if result.success:
                    logger.info(
                        f"[MT5] Order executed successfully! "
                        f"Ticket: {result.ticket}, "
                        f"Price: ${result.price:,.2f}, "
                        f"Volume: {result.volume}"
                    )

                    # Save trade to database
                    if self.database:
                        self.database.save_trade({
                            'symbol': signal.symbol,
                            'direction': signal.direction,
                            'entry_price': result.price,
                            'stop_loss': signal.stop_loss,
                            'take_profit': signal.take_profit,
                            'position_size': result.volume,
                            'entry_time': signal.timestamp.isoformat(),
                            'mt5_ticket': result.ticket,
                            'is_paper': False
                        })
                else:
                    logger.error(
                        f"[MT5] Order FAILED: {result.error_message} "
                        f"(code: {result.error_code})"
                    )

            except Exception as e:
                logger.error(f"[MT5] Execution error: {e}")
                import traceback
                logger.error(traceback.format_exc())


# Import pandas at module level for type hints
import pandas as pd
