"""
MT5 Connector - Unified connector for both data fetching and order execution.

This connector provides:
1. Historical OHLCV data via copy_rates_from()
2. Real-time quotes via symbol_info_tick()
3. Order execution via order_send()
4. Position management via positions_get()

Thread Safety: MT5 library is not thread-safe. This connector uses
asyncio.to_thread() to run blocking MT5 calls without blocking the event loop.

Usage:
    connector = MT5Connector(config)
    await connector.initialize()

    # Data fetching
    bars = await connector.get_bars('BTCUSD_SB', 'H4', 100)
    tick = await connector.get_tick('BTCUSD_SB')

    # Execution (LONG only for Bull Bot)
    result = await connector.place_order(signal)
    await connector.close_position(ticket)
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum

import pandas as pd
import numpy as np

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    mt5 = None

logger = logging.getLogger(__name__)


class MT5Timeframe(Enum):
    """Mapping of string timeframes to MT5 constants."""
    M1 = 'M1'
    M5 = 'M5'
    M15 = 'M15'
    M30 = 'M30'
    H1 = 'H1'
    H4 = 'H4'
    D1 = 'D1'
    W1 = 'W1'
    MN1 = 'MN1'


# Timeframe string to MT5 constant mapping
TIMEFRAME_MAP = {
    '1m': 'TIMEFRAME_M1',
    '5m': 'TIMEFRAME_M5',
    '15m': 'TIMEFRAME_M15',
    '30m': 'TIMEFRAME_M30',
    '1h': 'TIMEFRAME_H1',
    '4h': 'TIMEFRAME_H4',
    'd': 'TIMEFRAME_D1',
    '1d': 'TIMEFRAME_D1',
    'w': 'TIMEFRAME_W1',
    '1w': 'TIMEFRAME_W1',
    'mn': 'TIMEFRAME_MN1',
    # MT5 native formats
    'M1': 'TIMEFRAME_M1',
    'M5': 'TIMEFRAME_M5',
    'M15': 'TIMEFRAME_M15',
    'M30': 'TIMEFRAME_M30',
    'H1': 'TIMEFRAME_H1',
    'H4': 'TIMEFRAME_H4',
    'D1': 'TIMEFRAME_D1',
    'W1': 'TIMEFRAME_W1',
    'MN1': 'TIMEFRAME_MN1',
}


@dataclass
class MT5SymbolConfig:
    """Symbol configuration for MT5."""
    btc: str = 'BTCUSD_SB'
    gold: str = 'XAUUSD_SB'
    sp500: str = 'US500_SB'
    dxy: str = 'USDX_SB'
    nasdaq: str = 'NAS100_SB'


@dataclass
class OrderResult:
    """Result of an order operation."""
    success: bool
    ticket: Optional[int] = None
    volume: float = 0.0
    price: float = 0.0
    comment: str = ''
    error_code: int = 0
    error_message: str = ''


@dataclass
class Position:
    """Represents an open position."""
    ticket: int
    symbol: str
    type: str  # 'BUY' or 'SELL'
    volume: float
    price_open: float
    price_current: float
    sl: float
    tp: float
    profit: float
    time: datetime
    comment: str = ''


class MT5Connector:
    """
    Unified MT5 connector for data fetching and order execution.

    For Bull BC1: Only executes BUY (LONG) orders.
    """

    def __init__(self, config=None):
        """
        Initialize MT5 connector.

        Args:
            config: Configuration object with MT5 settings
        """
        if not MT5_AVAILABLE:
            raise ImportError("MetaTrader5 package not installed. Install with: pip install MetaTrader5")

        self.config = config
        self._initialized = False
        self._lock = asyncio.Lock()

        # Symbol mapping (configurable)
        if config and hasattr(config, 'mt5'):
            self.symbols = MT5SymbolConfig(
                btc=getattr(config.mt5, 'btc_symbol', 'BTCUSD_SB'),
                gold=getattr(config.mt5, 'gold_symbol', 'XAUUSD_SB'),
                sp500=getattr(config.mt5, 'sp500_symbol', 'US500_SB'),
                dxy=getattr(config.mt5, 'dxy_symbol', 'USDX_SB'),
                nasdaq=getattr(config.mt5, 'nasdaq_symbol', 'NAS100_SB'),
            )
        else:
            self.symbols = MT5SymbolConfig()

        # Connection settings
        self._timeout = 30 if config is None else getattr(config.mt5, 'timeout_seconds', 30)
        self._retry_attempts = 3 if config is None else getattr(config.mt5, 'retry_attempts', 3)

        # Cache for symbol info
        self._symbol_cache: Dict[str, Any] = {}

    async def initialize(self) -> bool:
        """
        Initialize connection to MT5 terminal.

        Returns:
            True if initialization successful
        """
        async with self._lock:
            if self._initialized:
                return True

            try:
                # Run MT5 initialization in thread pool
                result = await asyncio.to_thread(self._init_mt5)
                self._initialized = result
                return result
            except Exception as e:
                logger.error(f"MT5 initialization error: {e}")
                return False

    def _init_mt5(self) -> bool:
        """Synchronous MT5 initialization."""
        if not mt5.initialize():
            error = mt5.last_error()
            logger.error(f"MT5 initialize failed: {error}")
            return False

        # Log connection info
        terminal = mt5.terminal_info()
        account = mt5.account_info()

        if terminal:
            logger.info(f"MT5 connected: {terminal.name} ({terminal.company})")

        if account:
            logger.info(f"Account: {account.login} @ {account.server}")
            logger.info(f"Balance: ${account.balance:,.2f}, Leverage: 1:{account.leverage}")

        return True

    async def shutdown(self) -> None:
        """Shutdown MT5 connection."""
        async with self._lock:
            if self._initialized:
                await asyncio.to_thread(mt5.shutdown)
                self._initialized = False
                logger.info("MT5 connection closed")

    # =========================================================================
    # DATA FETCHING METHODS
    # =========================================================================

    async def get_bars(
        self,
        symbol: str,
        timeframe: str,
        count: int = 100,
        start_pos: int = 0
    ) -> pd.DataFrame:
        """
        Get historical OHLCV bars from MT5.

        Args:
            symbol: MT5 symbol name (e.g., 'BTCUSD_SB')
            timeframe: Timeframe string (e.g., '4h', 'H4', '1d')
            count: Number of bars to fetch
            start_pos: Starting position (0 = most recent)

        Returns:
            DataFrame with columns: datetime, open, high, low, close, volume
        """
        if not self._initialized:
            await self.initialize()

        # Convert timeframe string to MT5 constant
        tf_key = timeframe.lower() if timeframe.lower() in TIMEFRAME_MAP else timeframe
        tf_attr = TIMEFRAME_MAP.get(tf_key)
        if tf_attr is None:
            raise ValueError(f"Invalid timeframe: {timeframe}")

        mt5_timeframe = getattr(mt5, tf_attr)

        # Fetch data in thread pool
        rates = await asyncio.to_thread(
            mt5.copy_rates_from_pos,
            symbol,
            mt5_timeframe,
            start_pos,
            count
        )

        if rates is None or len(rates) == 0:
            error = mt5.last_error()
            logger.warning(f"No data for {symbol} {timeframe}: {error}")
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(rates)
        df['datetime'] = pd.to_datetime(df['time'], unit='s', utc=True)
        df = df.rename(columns={
            'tick_volume': 'volume',
            'real_volume': 'real_volume'
        })

        # Keep only standard OHLCV columns
        columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']
        available_cols = [c for c in columns if c in df.columns]
        df = df[available_cols]

        return df

    async def get_tick(self, symbol: str) -> Dict:
        """
        Get current tick data for a symbol.

        Args:
            symbol: MT5 symbol name

        Returns:
            Dictionary with tick data (bid, ask, last, time)
        """
        if not self._initialized:
            await self.initialize()

        tick = await asyncio.to_thread(mt5.symbol_info_tick, symbol)

        if tick is None:
            error = mt5.last_error()
            logger.warning(f"No tick for {symbol}: {error}")
            return {}

        return {
            'symbol': symbol,
            'bid': tick.bid,
            'ask': tick.ask,
            'last': tick.last,
            'volume': tick.volume,
            'time': datetime.fromtimestamp(tick.time, tz=timezone.utc),
            'spread': tick.ask - tick.bid
        }

    async def get_current_price(self, symbol: str) -> float:
        """
        Get current price (bid) for a symbol.

        Args:
            symbol: MT5 symbol name

        Returns:
            Current bid price
        """
        tick = await self.get_tick(symbol)
        return tick.get('bid', 0.0)

    async def get_symbol_info(self, symbol: str) -> Dict:
        """
        Get symbol information.

        Args:
            symbol: MT5 symbol name

        Returns:
            Dictionary with symbol info
        """
        if symbol in self._symbol_cache:
            return self._symbol_cache[symbol]

        if not self._initialized:
            await self.initialize()

        info = await asyncio.to_thread(mt5.symbol_info, symbol)

        if info is None:
            return {}

        result = {
            'name': info.name,
            'description': info.description,
            'digits': info.digits,
            'spread': info.spread,
            'point': info.point,
            'trade_mode': info.trade_mode,
            'volume_min': info.volume_min,
            'volume_max': info.volume_max,
            'volume_step': info.volume_step,
            'trade_contract_size': info.trade_contract_size,
        }

        self._symbol_cache[symbol] = result
        return result

    async def get_available_symbols(self, group: str = "*BTC*") -> List[str]:
        """
        Get list of available symbols matching a pattern.

        Args:
            group: Symbol filter pattern (e.g., "*BTC*")

        Returns:
            List of symbol names
        """
        if not self._initialized:
            await self.initialize()

        symbols = await asyncio.to_thread(mt5.symbols_get, group)

        if symbols is None:
            return []

        return [s.name for s in symbols]

    # =========================================================================
    # EXECUTION METHODS (LONG ONLY)
    # =========================================================================

    def _validate_volume(self, volume: float, symbol_info: Dict) -> float:
        """
        Validate and adjust volume to meet symbol requirements.

        Args:
            volume: Requested volume
            symbol_info: Symbol info with volume_min, volume_max, volume_step

        Returns:
            Adjusted volume that meets symbol requirements
        """
        volume_min = symbol_info.get('volume_min', 0.01)
        volume_max = symbol_info.get('volume_max', 100.0)
        volume_step = symbol_info.get('volume_step', 0.01)

        # Round to volume_step (round to nearest valid increment)
        if volume_step > 0:
            # Round to nearest step
            steps = round(volume / volume_step)
            adjusted_volume = steps * volume_step
            # Fix floating point precision
            adjusted_volume = round(adjusted_volume, 8)
        else:
            adjusted_volume = volume

        # Clamp to min/max
        adjusted_volume = max(volume_min, min(volume_max, adjusted_volume))

        if adjusted_volume != volume:
            logger.info(
                f"Volume adjusted: {volume:.6f} -> {adjusted_volume:.6f} "
                f"(min={volume_min}, max={volume_max}, step={volume_step})"
            )

        return adjusted_volume

    async def place_order(
        self,
        symbol: str,
        order_type: str,  # 'BUY' only for Bull Bot
        volume: float,
        price: Optional[float] = None,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
        comment: str = 'bull_bc1'
    ) -> OrderResult:
        """
        Place a market order (BUY only for Bull Bot).

        Args:
            symbol: MT5 symbol
            order_type: 'BUY' (SELL not supported for Bull Bot)
            volume: Position size in lots
            price: Order price (None for market order)
            sl: Stop loss price (below entry for LONG)
            tp: Take profit price (above entry for LONG)
            comment: Order comment

        Returns:
            OrderResult with execution details
        """
        # Bull Bot only does LONG (BUY) orders
        if order_type != 'BUY':
            logger.warning(f"Bull Bot only supports BUY orders, got {order_type}")
            return OrderResult(
                success=False,
                error_message="Bull Bot only supports BUY (LONG) orders"
            )

        if not self._initialized:
            await self.initialize()

        # Get symbol info and validate volume
        symbol_info = await self.get_symbol_info(symbol)
        if not symbol_info:
            logger.error(f"Cannot get symbol info for {symbol}")
            return OrderResult(
                success=False,
                error_message=f"Cannot get symbol info for {symbol}"
            )

        # Validate and adjust volume to meet MT5 requirements
        validated_volume = self._validate_volume(volume, symbol_info)

        # Check if volume is valid after adjustment
        volume_min = symbol_info.get('volume_min', 0.01)
        if validated_volume < volume_min:
            logger.error(f"Volume {validated_volume} below minimum {volume_min}")
            return OrderResult(
                success=False,
                error_message=f"Volume {validated_volume} below minimum {volume_min}"
            )

        # Get current tick for market order
        if price is None:
            tick = await self.get_tick(symbol)
            price = tick['ask']  # BUY at ask price

        # Prepare order request
        request = {
            'action': mt5.TRADE_ACTION_DEAL,
            'symbol': symbol,
            'volume': validated_volume,
            'type': mt5.ORDER_TYPE_BUY,
            'price': price,
            'deviation': 20,  # Slippage in points
            'magic': 20241217,  # Magic number for Bull BC1
            'comment': comment,
            'type_time': mt5.ORDER_TIME_GTC,
            'type_filling': mt5.ORDER_FILLING_IOC,
        }

        if sl is not None:
            request['sl'] = sl
        if tp is not None:
            request['tp'] = tp

        logger.info(f"[MT5] Placing BUY order: {symbol} vol={validated_volume} sl={sl} tp={tp}")

        # Send order
        result = await asyncio.to_thread(mt5.order_send, request)

        if result is None:
            error = mt5.last_error()
            return OrderResult(
                success=False,
                error_code=error[0] if error else 0,
                error_message=error[1] if error else 'Unknown error'
            )

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            return OrderResult(
                success=False,
                error_code=result.retcode,
                error_message=result.comment
            )

        return OrderResult(
            success=True,
            ticket=result.order,
            volume=result.volume,
            price=result.price,
            comment=result.comment
        )

    async def close_position(self, ticket: int) -> OrderResult:
        """
        Close an open position by ticket.

        Args:
            ticket: Position ticket number

        Returns:
            OrderResult with close details
        """
        if not self._initialized:
            await self.initialize()

        # Get position info
        positions = await self.get_positions()
        position = next((p for p in positions if p.ticket == ticket), None)

        if position is None:
            return OrderResult(
                success=False,
                error_message=f'Position {ticket} not found'
            )

        # For LONG positions, close with SELL
        close_type = mt5.ORDER_TYPE_SELL if position.type == 'BUY' else mt5.ORDER_TYPE_BUY

        # Get current price for closing
        tick = await self.get_tick(position.symbol)
        if not tick:
            return OrderResult(
                success=False,
                error_message=f'Cannot get tick for {position.symbol}'
            )

        price = tick['bid'] if close_type == mt5.ORDER_TYPE_SELL else tick['ask']

        request = {
            'action': mt5.TRADE_ACTION_DEAL,
            'position': ticket,
            'symbol': position.symbol,
            'volume': position.volume,
            'type': close_type,
            'price': price,
            'deviation': 20,
            'magic': 234001,
            'comment': f'close_{ticket}',
            'type_time': mt5.ORDER_TIME_GTC,
            'type_filling': mt5.ORDER_FILLING_IOC,
        }

        result = await asyncio.to_thread(mt5.order_send, request)

        if result is None:
            error = mt5.last_error()
            return OrderResult(
                success=False,
                error_code=error[0] if error else 0,
                error_message=error[1] if error else 'Unknown error'
            )

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            return OrderResult(
                success=False,
                error_code=result.retcode,
                error_message=result.comment
            )

        return OrderResult(
            success=True,
            ticket=result.order,
            price=result.price,
            volume=result.volume
        )

    async def modify_position(
        self,
        ticket: int,
        sl: Optional[float] = None,
        tp: Optional[float] = None
    ) -> OrderResult:
        """
        Modify stop loss and/or take profit for a position.

        Args:
            ticket: Position ticket
            sl: New stop loss (None to keep current)
            tp: New take profit (None to keep current)

        Returns:
            OrderResult with modification details
        """
        if not self._initialized:
            await self.initialize()

        # Get current position
        positions = await self.get_positions()
        position = next((p for p in positions if p.ticket == ticket), None)

        if position is None:
            return OrderResult(
                success=False,
                error_message=f'Position {ticket} not found'
            )

        # Prepare modification request
        request = {
            'action': mt5.TRADE_ACTION_SLTP,
            'position': ticket,
            'symbol': position.symbol,
            'sl': sl if sl is not None else position.sl,
            'tp': tp if tp is not None else position.tp,
        }

        result = await asyncio.to_thread(mt5.order_send, request)

        if result is None:
            error = mt5.last_error()
            return OrderResult(
                success=False,
                error_code=error[0] if error else 0,
                error_message=error[1] if error else 'Unknown error'
            )

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            return OrderResult(
                success=False,
                error_code=result.retcode,
                error_message=result.comment
            )

        return OrderResult(success=True, ticket=ticket)

    async def get_positions(self, symbol: Optional[str] = None) -> List[Position]:
        """
        Get all open positions.

        Args:
            symbol: Filter by symbol (None for all)

        Returns:
            List of Position objects
        """
        if not self._initialized:
            await self.initialize()

        if symbol:
            positions = await asyncio.to_thread(mt5.positions_get, symbol=symbol)
        else:
            positions = await asyncio.to_thread(mt5.positions_get)

        if positions is None:
            return []

        result = []
        for pos in positions:
            result.append(Position(
                ticket=pos.ticket,
                symbol=pos.symbol,
                type='BUY' if pos.type == mt5.ORDER_TYPE_BUY else 'SELL',
                volume=pos.volume,
                price_open=pos.price_open,
                price_current=pos.price_current,
                sl=pos.sl,
                tp=pos.tp,
                profit=pos.profit,
                time=datetime.fromtimestamp(pos.time, tz=timezone.utc),
                comment=pos.comment
            ))

        return result

    async def get_account_info(self) -> Dict:
        """
        Get account information.

        Returns:
            Dictionary with account details
        """
        if not self._initialized:
            await self.initialize()

        info = await asyncio.to_thread(mt5.account_info)

        if info is None:
            return {}

        return {
            'login': info.login,
            'server': info.server,
            'balance': info.balance,
            'equity': info.equity,
            'margin': info.margin,
            'margin_free': info.margin_free,
            'margin_level': info.margin_level,
            'leverage': info.leverage,
            'profit': info.profit,
        }

    # =========================================================================
    # CONVENIENCE METHODS FOR BULL BC1
    # =========================================================================

    async def get_btc_bars(self, timeframe: str, count: int = 100) -> pd.DataFrame:
        """Get BTC OHLCV data using configured symbol."""
        return await self.get_bars(self.symbols.btc, timeframe, count)

    async def get_gold_bars(self, timeframe: str, count: int = 100) -> pd.DataFrame:
        """Get Gold OHLCV data using configured symbol."""
        return await self.get_bars(self.symbols.gold, timeframe, count)

    async def get_sp500_bars(self, timeframe: str, count: int = 100) -> pd.DataFrame:
        """Get S&P 500 OHLCV data using configured symbol."""
        return await self.get_bars(self.symbols.sp500, timeframe, count)

    async def get_dxy_bars(self, timeframe: str, count: int = 100) -> pd.DataFrame:
        """Get DXY OHLCV data using configured symbol."""
        return await self.get_bars(self.symbols.dxy, timeframe, count)

    async def get_btc_price(self) -> float:
        """Get current BTC price."""
        return await self.get_current_price(self.symbols.btc)

    async def check_connection(self) -> bool:
        """Check if MT5 connection is alive."""
        try:
            if not self._initialized:
                return False

            # Try to get account info as a health check
            info = await asyncio.to_thread(mt5.account_info)
            return info is not None
        except Exception:
            return False
