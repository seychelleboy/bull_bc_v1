"""
Database Module - SQLite database for signal and trade tracking

Stores:
- Generated signals
- Paper trades and their outcomes
- Performance metrics
"""

import sqlite3
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class Database:
    """
    SQLite database for Bull BC1 bot.

    Tables:
    - signals: Generated trading signals
    - trades: Paper and live trades
    - metrics: Performance metrics snapshots

    Example:
        db = Database('data/database/bull_bc_1.db')
        db.initialize()
        db.save_signal(signal_dict)
    """

    def __init__(self, db_path: str = 'data/database/bull_bc_1.db'):
        """
        Initialize database.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection: Optional[sqlite3.Connection] = None

    @contextmanager
    def _get_connection(self):
        """Get database connection with context manager."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

    def initialize(self) -> None:
        """Create database tables if they don't exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Signals table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    scenario TEXT,
                    entry_price REAL NOT NULL,
                    stop_loss REAL NOT NULL,
                    take_profit REAL NOT NULL,
                    risk_reward REAL,
                    ensemble_prob REAL,
                    bait_score REAL,
                    reasons TEXT,
                    metadata TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Trades table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    signal_id INTEGER,
                    symbol TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'OPEN',
                    entry_price REAL NOT NULL,
                    exit_price REAL,
                    stop_loss REAL NOT NULL,
                    take_profit REAL NOT NULL,
                    position_size REAL NOT NULL,
                    pnl REAL,
                    pnl_percent REAL,
                    entry_time TEXT NOT NULL,
                    exit_time TEXT,
                    exit_reason TEXT,
                    is_paper BOOLEAN NOT NULL DEFAULT 1,
                    metadata TEXT,
                    FOREIGN KEY (signal_id) REFERENCES signals(id)
                )
            ''')

            # Metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    total_signals INTEGER,
                    total_trades INTEGER,
                    winning_trades INTEGER,
                    losing_trades INTEGER,
                    win_rate REAL,
                    total_pnl REAL,
                    max_drawdown REAL,
                    sharpe_ratio REAL,
                    metadata TEXT
                )
            ''')

            logger.info(f"Database initialized at {self.db_path}")

    def save_signal(self, signal: Dict) -> int:
        """
        Save a trading signal.

        Args:
            signal: Signal dictionary

        Returns:
            Signal ID
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Extract BAIT scores from metadata if available
            metadata = signal.get('metadata', {})
            bait_data = metadata.get('bait_score', {})

            cursor.execute('''
                INSERT INTO signals (
                    timestamp, symbol, direction, confidence, scenario,
                    behavioral_score, analytical_score, informational_score,
                    technical_score, bait_combined,
                    entry_price, stop_loss, take_profit, risk_reward_ratio,
                    position_size, risk_amount, action_taken,
                    features_json, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                signal.get('timestamp', datetime.now(timezone.utc).isoformat()),
                signal.get('symbol', 'BTCUSD'),
                signal.get('direction', 'LONG'),
                signal.get('confidence', 0),
                signal.get('scenario'),
                bait_data.get('behavioral', 0),
                bait_data.get('analytical', 0),
                bait_data.get('informational', 0),
                bait_data.get('technical', 0),
                bait_data.get('combined', 0),
                signal.get('entry_price', 0),
                signal.get('stop_loss', 0),
                signal.get('take_profit', 0),
                signal.get('risk_reward_ratio', 0),
                signal.get('position_size', 0),
                signal.get('risk_amount', 0),
                'EXECUTED',
                json.dumps(signal.get('features', {})),
                json.dumps(metadata)
            ))
            return cursor.lastrowid

    def save_trade(self, trade: Dict) -> int:
        """
        Save a trade to trades or paper_trades table.

        Args:
            trade: Trade dictionary

        Returns:
            Trade ID
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            is_paper = trade.get('is_paper', False)

            if is_paper:
                # Save to paper_trades table
                cursor.execute('''
                    INSERT INTO paper_trades (
                        timestamp, signal_id, direction, entry_price,
                        stop_loss, take_profit, position_size, status, entry_time
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    datetime.now(timezone.utc).isoformat(),
                    trade.get('signal_id'),
                    trade.get('direction', 'LONG'),
                    trade.get('entry_price', 0),
                    trade.get('stop_loss', 0),
                    trade.get('take_profit', 0),
                    trade.get('position_size', 0),
                    'OPEN',
                    trade.get('entry_time', datetime.now(timezone.utc).isoformat())
                ))
            else:
                # Save to trades table (live trades)
                cursor.execute('''
                    INSERT INTO trades (
                        signal_id, mt5_ticket, symbol, direction, entry_price,
                        entry_time, position_size, initial_stop_loss,
                        initial_take_profit, status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    trade.get('signal_id'),
                    trade.get('mt5_ticket'),
                    trade.get('symbol', 'BTCUSD'),
                    trade.get('direction', 'LONG'),
                    trade.get('entry_price', 0),
                    trade.get('entry_time', datetime.now(timezone.utc).isoformat()),
                    trade.get('position_size', 0),
                    trade.get('stop_loss', 0),
                    trade.get('take_profit', 0),
                    'OPEN'
                ))
            return cursor.lastrowid

    def update_trade(self, trade_id: int, updates: Dict) -> None:
        """
        Update a trade.

        Args:
            trade_id: Trade ID
            updates: Fields to update
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Build update query
            fields = []
            values = []
            for key, value in updates.items():
                if key in ['exit_price', 'pnl', 'pnl_percent', 'exit_time', 'exit_reason', 'status']:
                    fields.append(f"{key} = ?")
                    values.append(value)

            if fields:
                values.append(trade_id)
                query = f"UPDATE trades SET {', '.join(fields)} WHERE id = ?"
                cursor.execute(query, values)

    def get_open_trades(self) -> List[Dict]:
        """
        Get all open trades.

        Returns:
            List of open trade dictionaries
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM trades WHERE status = 'OPEN'")
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def get_trade_stats(self) -> Dict:
        """
        Get trading statistics.

        Returns:
            Dictionary with trade statistics
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Total trades
            cursor.execute("SELECT COUNT(*) FROM trades WHERE status != 'OPEN'")
            total_trades = cursor.fetchone()[0]

            # Winning trades
            cursor.execute("SELECT COUNT(*) FROM trades WHERE pnl > 0")
            winning_trades = cursor.fetchone()[0]

            # Total PnL
            cursor.execute("SELECT SUM(pnl) FROM trades WHERE pnl IS NOT NULL")
            total_pnl = cursor.fetchone()[0] or 0

            # Average win/loss
            cursor.execute("SELECT AVG(pnl) FROM trades WHERE pnl > 0")
            avg_win = cursor.fetchone()[0] or 0

            cursor.execute("SELECT AVG(pnl) FROM trades WHERE pnl < 0")
            avg_loss = cursor.fetchone()[0] or 0

            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

            return {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': total_trades - winning_trades,
                'win_rate': round(win_rate, 1),
                'total_pnl': round(total_pnl, 2),
                'avg_win': round(avg_win, 2),
                'avg_loss': round(avg_loss, 2)
            }

    def get_recent_signals(self, limit: int = 10) -> List[Dict]:
        """
        Get recent signals.

        Args:
            limit: Maximum number of signals to return

        Returns:
            List of signal dictionaries
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM signals ORDER BY created_at DESC LIMIT ?",
                (limit,)
            )
            rows = cursor.fetchall()
            return [dict(row) for row in rows]


# Global database instance
_database: Optional[Database] = None


def get_database(db_path: str = 'data/database/bull_bc_1.db') -> Database:
    """Get global database instance (singleton)."""
    global _database
    if _database is None:
        _database = Database(db_path)
        _database.initialize()
    return _database
