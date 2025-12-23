"""
Bull BC1 - Bitcoin Bull Detection Bot (Long-Only)

A BAIT-enhanced trading bot for detecting bullish opportunities in BTC.

Usage:
    python main.py              # Continuous scanning (swing mode)
    python main.py --mode scalp # Scalping mode (5m timeframe)
    python main.py --analyze    # Single analysis cycle
    python main.py --status     # Show system status
    python main.py --paper      # Paper trading mode
    python main.py --no-auto    # Disable auto-execution
"""

import asyncio
import argparse
import signal
import sys
import logging
from pathlib import Path
from datetime import datetime, timezone

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Load .env file
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / 'config' / '.env')

from config import setup_logging, get_config, Config
from src.engine import BullEngine
from src.data import get_database


# Global engine reference for signal handling
_engine = None


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    print("\nShutdown signal received...")
    if _engine:
        _engine.stop()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Bull BC1 - Bitcoin Bull Detection Bot (Long-Only)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py                # Run continuous scanning
    python main.py --analyze      # Run single analysis
    python main.py --paper        # Enable paper trading
    python main.py --no-auto      # Disable auto-execution
    python main.py --status       # Show system status
        """
    )

    parser.add_argument(
        '--analyze', '-a',
        action='store_true',
        help='Run single analysis cycle and exit'
    )

    parser.add_argument(
        '--status', '-s',
        action='store_true',
        help='Show system status and exit'
    )

    parser.add_argument(
        '--paper', '-p',
        action='store_true',
        help='Enable paper trading mode'
    )

    parser.add_argument(
        '--no-auto',
        action='store_true',
        help='Disable auto-execution of trades'
    )

    parser.add_argument(
        '--debug', '-d',
        action='store_true',
        help='Enable debug logging'
    )

    parser.add_argument(
        '--config', '-c',
        type=str,
        default=None,
        help='Path to YAML config file'
    )

    parser.add_argument(
        '--mode', '-m',
        type=str,
        choices=['swing', 'scalp'],
        default='swing',
        help='Trading mode: swing (4h, default) or scalp (5m)'
    )

    return parser.parse_args()


async def run_analysis(engine: BullEngine) -> None:
    """Run a single analysis cycle."""
    print("\n" + "=" * 60)
    print("BULL BC1 - Single Analysis Mode")
    print("=" * 60)

    try:
        await engine.initialize()

        signal = await engine.analyze()

        if signal:
            print("\n" + "=" * 60)
            print("BULLISH SIGNAL DETECTED!")
            print("=" * 60)
            print(f"Symbol:      {signal.symbol}")
            print(f"Direction:   {signal.direction}")
            print(f"Confidence:  {signal.confidence:.1f}%")
            print(f"Scenario:    {signal.scenario}")
            print(f"Entry:       ${signal.entry_price:,.2f}")
            print(f"Stop Loss:   ${signal.stop_loss:,.2f}")
            print(f"Take Profit: ${signal.take_profit:,.2f}")
            print(f"R:R Ratio:   {signal.risk_reward_ratio:.1f}")
            print(f"\nReasons:")
            for reason in signal.reasons:
                print(f"  - {reason}")
            print("=" * 60)
        else:
            print("\nNo bullish signal detected at this time.")

    finally:
        await engine.shutdown()


async def run_continuous(engine: BullEngine, mode: str = 'swing') -> None:
    """Run continuous scanning loop."""
    global _engine
    _engine = engine

    mode_label = "SCALP MODE (5m)" if mode == 'scalp' else "SWING MODE (4h)"

    print("\n" + "=" * 60)
    print(f"BULL BC1 - {mode_label}")
    print("=" * 60)
    print(f"Symbol:       {engine.config.trading.symbol}")
    print(f"Timeframe:    {engine.config.trading.primary_timeframe}")
    print(f"Auto-Execute: {engine.config.trading.auto_execute}")
    print(f"Paper Trade:  {engine.config.trading.paper_trade}")
    print(f"Confidence:   {engine.config.trading.min_confidence_threshold}%")
    print(f"Scan Interval: {engine.config.trading.scan_interval_seconds}s")
    print(f"BAIT Weights: B={engine.config.scoring.behavioral_weight} "
          f"A={engine.config.scoring.analytical_weight} "
          f"I={engine.config.scoring.informational_weight} "
          f"T={engine.config.scoring.technical_weight}")
    print("=" * 60)
    print("Press Ctrl+C to stop...")

    try:
        await engine.initialize()
        await engine.run()
    except asyncio.CancelledError:
        pass
    finally:
        await engine.shutdown()


def show_status(config: Config) -> None:
    """Show system status."""
    print("\n" + "=" * 60)
    print("BULL BC1 - System Status")
    print("=" * 60)

    # Configuration
    print(f"\nConfiguration:")
    print(f"  Symbol:            {config.trading.symbol}")
    print(f"  Direction:         LONG (only)")
    print(f"  Auto-Execute:      {config.trading.auto_execute}")
    print(f"  Paper Trade:       {config.trading.paper_trade}")
    print(f"  Confidence:        {config.trading.min_confidence_threshold}%")
    print(f"  Scan Interval:     {config.trading.scan_interval_seconds}s")

    print(f"\nRisk Management:")
    print(f"  Initial Balance:   ${config.risk.initial_balance:,.2f}")
    print(f"  Risk Per Trade:    {config.risk.risk_per_trade_percent}%")
    print(f"  Max Daily Loss:    {config.risk.max_daily_loss_percent}%")
    print(f"  Max Leverage:      {config.risk.max_leverage}x")

    print(f"\nBAIT Scoring (LONG optimized):")
    print(f"  Behavioral Weight:     {config.scoring.behavioral_weight}")
    print(f"  Analytical Weight:     {config.scoring.analytical_weight}")
    print(f"  Informational Weight:  {config.scoring.informational_weight}")
    print(f"  Technical Weight:      {config.scoring.technical_weight}")
    print(f"  Strong Threshold:      >= {config.scoring.strong_threshold}")

    # Database stats
    try:
        db = get_database(config.database.path)
        stats = db.get_trade_stats()

        print(f"\nTrading Statistics:")
        print(f"  Total Trades:      {stats.get('total_trades', 0)}")
        print(f"  Winning Trades:    {stats.get('winning_trades', 0)}")
        print(f"  Win Rate:          {stats.get('win_rate', 0):.1f}%")
        print(f"  Total P&L:         ${stats.get('total_pnl', 0):,.2f}")
    except Exception as e:
        print(f"\nDatabase: Not initialized or error - {e}")

    print("=" * 60)


def main():
    """Main entry point."""
    args = parse_args()

    # Setup logging
    log_level = 'DEBUG' if args.debug else 'INFO'
    setup_logging(level=log_level)

    logger = logging.getLogger(__name__)

    # Determine config file based on mode
    config_file = args.config
    if args.mode == 'scalp' and not config_file:
        config_file = str(Path(__file__).parent / 'config' / 'scalping.yaml')
        print(f"[SCALP MODE] Loading scalping configuration...")

    # Load configuration
    config = Config.load(config_file)

    # Validate configuration
    errors = config.validate()
    if errors:
        print("Configuration errors:")
        for error in errors:
            print(f"  - {error}")
        if 'EODHD_API_KEY' in str(errors):
            print("\nHint: Set EODHD_API_KEY in config/.env")
        return 1

    # Apply command line overrides
    if args.paper:
        config.trading.paper_trade = True
    if args.no_auto:
        config.trading.auto_execute = False

    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Handle different modes
    if args.status:
        show_status(config)
        return 0

    # Create engine
    engine = BullEngine(config)

    try:
        if args.analyze:
            asyncio.run(run_analysis(engine))
        else:
            asyncio.run(run_continuous(engine, args.mode))

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
