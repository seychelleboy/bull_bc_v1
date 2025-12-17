"""
Database Initialization Script - Bull BC1 (LONG Only)

Creates SQLite database with WAL mode for concurrent read/write access.
Implements schema for signals, trades, performance tracking, and ML feedback loop.
"""

import sqlite3
import os
import sys
from pathlib import Path
from datetime import datetime, timezone

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def get_database_path() -> Path:
    """Get the database path, creating directory if needed."""
    db_dir = PROJECT_ROOT / "data" / "database"
    db_dir.mkdir(parents=True, exist_ok=True)
    return db_dir / "bull_bc_1.db"


def create_schema(conn: sqlite3.Connection) -> None:
    """Create all database tables."""
    cursor = conn.cursor()

    # =========================================================================
    # CORE TRADING TABLES
    # =========================================================================

    # Signals table - All generated trading signals (LONG only for Bull Bot)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME NOT NULL,
            symbol TEXT NOT NULL DEFAULT 'BTC-USD',
            direction TEXT NOT NULL DEFAULT 'LONG' CHECK (direction = 'LONG'),
            scenario TEXT NOT NULL,
            confidence REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 100),

            -- BAIT scoring components
            behavioral_score REAL,
            analytical_score REAL,
            informational_score REAL,
            technical_score REAL,
            bait_combined REAL,

            -- Trade levels (LONG: stop below entry, target above entry)
            entry_price REAL NOT NULL,
            stop_loss REAL NOT NULL,
            take_profit REAL NOT NULL,
            risk_reward_ratio REAL,

            -- Risk metrics
            position_size REAL,
            risk_amount REAL,

            -- Execution
            action_taken TEXT NOT NULL CHECK (action_taken IN ('EXECUTED', 'FILTERED', 'MANUAL_REVIEW')),
            filter_reason TEXT,

            -- Metadata
            features_json TEXT,  -- JSON blob of features for analysis
            metadata_json TEXT,  -- Additional context
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Trades table - Executed trades with outcomes
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            signal_id INTEGER REFERENCES signals(id),

            -- MT5 connection
            mt5_ticket INTEGER UNIQUE,
            mt5_order_id INTEGER,

            -- Execution details (LONG only)
            symbol TEXT NOT NULL DEFAULT 'BTC-USD',
            direction TEXT NOT NULL DEFAULT 'LONG' CHECK (direction = 'LONG'),

            -- Entry
            entry_price REAL NOT NULL,
            entry_time DATETIME NOT NULL,
            position_size REAL NOT NULL,

            -- Exit
            exit_price REAL,
            exit_time DATETIME,
            exit_reason TEXT CHECK (exit_reason IN ('TP_HIT', 'SL_HIT', 'TRAILING_STOP', 'MANUAL', 'TIME_EXIT', NULL)),

            -- P&L
            profit_loss REAL,
            profit_loss_pct REAL,
            commission REAL DEFAULT 0,

            -- Stop management (LONG: stop below entry)
            initial_stop_loss REAL,
            final_stop_loss REAL,  -- After trailing
            initial_take_profit REAL,

            -- Status
            status TEXT NOT NULL DEFAULT 'OPEN' CHECK (status IN ('OPEN', 'CLOSED', 'CANCELLED')),

            -- Metadata
            notes TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Paper trades table - Simulated trades for forward testing
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS paper_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME NOT NULL,
            signal_id INTEGER REFERENCES signals(id),

            -- Trade details (LONG only)
            direction TEXT NOT NULL DEFAULT 'LONG' CHECK (direction = 'LONG'),
            entry_price REAL NOT NULL,
            exit_price REAL,
            stop_loss REAL NOT NULL,
            take_profit REAL NOT NULL,
            position_size REAL NOT NULL,

            -- Status and lifecycle
            status TEXT NOT NULL DEFAULT 'OPEN' CHECK (status IN ('OPEN', 'CLOSED')),
            exit_reason TEXT CHECK (exit_reason IN ('TP_HIT', 'SL_HIT', 'MANUAL', NULL)),

            -- P&L
            profit_loss REAL,
            profit_loss_pct REAL,

            -- Timestamps
            entry_time DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            exit_time DATETIME,

            -- Metadata
            notes TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Performance table - Daily aggregated metrics
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS performance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date DATE NOT NULL UNIQUE,

            -- Trade counts
            signals_generated INTEGER DEFAULT 0,
            signals_filtered INTEGER DEFAULT 0,
            trades_executed INTEGER DEFAULT 0,
            trades_won INTEGER DEFAULT 0,
            trades_lost INTEGER DEFAULT 0,

            -- Win rates
            win_rate REAL,
            avg_win_pct REAL,
            avg_loss_pct REAL,
            profit_factor REAL,

            -- P&L
            gross_profit REAL DEFAULT 0,
            gross_loss REAL DEFAULT 0,
            net_pnl REAL DEFAULT 0,
            cumulative_pnl REAL DEFAULT 0,

            -- Risk metrics
            max_drawdown REAL,
            sharpe_ratio REAL,
            sortino_ratio REAL,

            -- Account
            starting_balance REAL,
            ending_balance REAL,

            -- BAIT scoring performance
            avg_bait_score REAL,
            avg_behavioral REAL,
            avg_analytical REAL,
            avg_informational REAL,
            avg_technical REAL,

            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # =========================================================================
    # BAIT SCORING HISTORY
    # =========================================================================

    # BAIT scores history - Track scoring over time
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS bait_scores (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME NOT NULL,
            signal_id INTEGER REFERENCES signals(id),

            -- Component scores (0-100, higher = more bullish)
            behavioral REAL NOT NULL,
            analytical REAL NOT NULL,
            informational REAL NOT NULL,
            technical REAL NOT NULL,
            combined REAL NOT NULL,

            -- Classification
            strength TEXT CHECK (strength IN ('STRONG', 'MODERATE', 'WEAK', 'AVOID')),

            -- Raw inputs
            fear_greed_value INTEGER,
            funding_rate REAL,
            news_sentiment REAL,
            rsi REAL,

            -- Outcome (filled after trade closes)
            actual_outcome INTEGER CHECK (actual_outcome IN (1, 0, NULL)),  -- 1=correct, 0=wrong

            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Component performance - Rolling accuracy per BAIT component
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS component_performance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date DATE NOT NULL,
            component TEXT NOT NULL,  -- 'behavioral', 'analytical', 'informational', 'technical', 'scenario_X'

            -- Daily stats
            predictions INTEGER DEFAULT 0,
            correct INTEGER DEFAULT 0,
            accuracy REAL,

            -- Confidence calibration
            avg_score REAL,
            score_when_correct REAL,
            score_when_wrong REAL,

            -- Rolling stats (last N predictions)
            rolling_accuracy_20 REAL,
            rolling_accuracy_50 REAL,
            rolling_accuracy_100 REAL,

            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,

            UNIQUE(date, component)
        )
    """)

    # BAIT weights history - Track weight adjustments over time
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS bait_weights_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME NOT NULL,

            -- Weights
            behavioral_weight REAL NOT NULL,
            analytical_weight REAL NOT NULL,
            informational_weight REAL NOT NULL,
            technical_weight REAL NOT NULL,

            -- Reason for change
            adjustment_reason TEXT,  -- 'DAILY_AUTO', 'MANUAL', 'EMERGENCY'

            -- Performance at time of adjustment
            behavioral_rolling_accuracy REAL,
            analytical_rolling_accuracy REAL,
            informational_rolling_accuracy REAL,
            technical_rolling_accuracy REAL,

            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # =========================================================================
    # PRICE DATA CACHE (for technical analysis)
    # =========================================================================

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS price_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL DEFAULT 'BTC-USD',
            timeframe TEXT NOT NULL,  -- '1h', '4h', '1d'
            timestamp DATETIME NOT NULL,

            -- OHLCV
            open REAL NOT NULL,
            high REAL NOT NULL,
            low REAL NOT NULL,
            close REAL NOT NULL,
            volume REAL NOT NULL,

            -- Pre-calculated indicators (optional, for speed)
            rsi_14 REAL,
            macd_line REAL,
            macd_signal REAL,
            atr_14 REAL,
            obv REAL,

            UNIQUE(symbol, timeframe, timestamp)
        )
    """)

    # =========================================================================
    # INDICES
    # =========================================================================

    # Signals indices
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_signals_timestamp ON signals(timestamp)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_signals_confidence ON signals(confidence)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_signals_action ON signals(action_taken)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_signals_scenario ON signals(scenario)")

    # Trades indices
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_signal_id ON trades(signal_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_entry_time ON trades(entry_time)")

    # Paper trades indices
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_paper_trades_signal_id ON paper_trades(signal_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_paper_trades_status ON paper_trades(status)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_paper_trades_entry_time ON paper_trades(entry_time)")

    # Performance indices
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_performance_date ON performance(date)")

    # BAIT scores indices
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_bait_scores_timestamp ON bait_scores(timestamp)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_bait_scores_outcome ON bait_scores(actual_outcome)")

    # Price data indices
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_price_data_lookup ON price_data(symbol, timeframe, timestamp)")

    conn.commit()
    print("Schema created successfully")


def create_triggers(conn: sqlite3.Connection) -> None:
    """Create database triggers for automatic updates."""
    cursor = conn.cursor()

    # Auto-update updated_at on trades
    cursor.execute("""
        CREATE TRIGGER IF NOT EXISTS trades_updated_at
        AFTER UPDATE ON trades
        BEGIN
            UPDATE trades SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
        END
    """)

    conn.commit()
    print("Triggers created successfully")


def insert_initial_weights(conn: sqlite3.Connection) -> None:
    """Insert initial BAIT weights from config."""
    cursor = conn.cursor()

    # Check if weights already exist
    cursor.execute("SELECT COUNT(*) FROM bait_weights_history")
    if cursor.fetchone()[0] > 0:
        print("BAIT weights already initialized")
        return

    # Insert initial weights (from config defaults - LONG optimized)
    cursor.execute("""
        INSERT INTO bait_weights_history
        (timestamp, behavioral_weight, analytical_weight, informational_weight, technical_weight, adjustment_reason)
        VALUES (?, 0.30, 0.25, 0.25, 0.20, 'INITIAL')
    """, (datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),))

    conn.commit()
    print("Initial BAIT weights inserted")


def enable_wal_mode(conn: sqlite3.Connection) -> None:
    """Enable WAL mode for concurrent access."""
    cursor = conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    result = cursor.fetchone()
    print(f"Journal mode set to: {result[0]}")

    # Additional optimizations
    cursor.execute("PRAGMA synchronous=NORMAL")  # Faster, still safe with WAL
    cursor.execute("PRAGMA cache_size=-64000")   # 64MB cache
    cursor.execute("PRAGMA temp_store=MEMORY")   # Temp tables in memory

    conn.commit()


def verify_database(conn: sqlite3.Connection) -> bool:
    """Verify database was created correctly."""
    cursor = conn.cursor()

    expected_tables = [
        'signals', 'trades', 'paper_trades', 'performance',
        'bait_scores', 'component_performance',
        'bait_weights_history', 'price_data'
    ]

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]

    missing = set(expected_tables) - set(tables)
    if missing:
        print(f"ERROR: Missing tables: {missing}")
        return False

    print(f"Verified {len(expected_tables)} tables exist")
    return True


def main():
    """Initialize the database."""
    print("=" * 60)
    print("Bull BC1 Database Initialization (LONG Only)")
    print("=" * 60)

    db_path = get_database_path()
    print(f"\nDatabase path: {db_path}")

    # Check if database exists
    db_exists = db_path.exists()
    if db_exists:
        print("Database already exists. Checking schema...")
    else:
        print("Creating new database...")

    # Connect and setup
    conn = sqlite3.connect(str(db_path))

    try:
        # Enable WAL mode first
        enable_wal_mode(conn)

        # Create schema
        create_schema(conn)

        # Create triggers
        create_triggers(conn)

        # Insert initial data
        insert_initial_weights(conn)

        # Verify
        if verify_database(conn):
            print("\nDatabase initialization SUCCESSFUL")
        else:
            print("\nDatabase initialization FAILED")
            sys.exit(1)

    finally:
        conn.close()

    print("=" * 60)


if __name__ == "__main__":
    main()
