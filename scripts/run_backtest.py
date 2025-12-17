"""
Run Backtest - CLI for historical strategy validation (LONG Only)

Standalone script that runs backtesting without touching live trading.

Usage:
    python scripts/run_backtest.py --days 365
    python scripts/run_backtest.py --days 90 --timeframe 1h
    python scripts/run_backtest.py --days 365 --output custom_output_dir
"""

import asyncio
import argparse
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env file from config folder (must be before importing config)
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / 'config' / '.env')

from config.settings import Config


async def main():
    """Run backtest with CLI arguments."""
    parser = argparse.ArgumentParser(
        description='Run backtest on historical data (LONG positions only)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Backtest last 365 days
  python scripts/run_backtest.py --days 365

  # Backtest last 90 days on 1h timeframe
  python scripts/run_backtest.py --days 90 --timeframe 1h

  # Custom output directory
  python scripts/run_backtest.py --days 365 --output data/backtest/custom

  # Quick test (30 days)
  python scripts/run_backtest.py --days 30
        """
    )

    parser.add_argument(
        '--days',
        type=int,
        default=365,
        help='Number of days to backtest (default: 365)'
    )

    parser.add_argument(
        '--timeframe',
        type=str,
        default='4h',
        choices=['1h', '4h', '1d'],
        help='Timeframe for analysis (default: 4h)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='data/backtest',
        help='Output directory for results (default: data/backtest)'
    )

    parser.add_argument(
        '--no-csv',
        action='store_true',
        help='Skip CSV export (JSON only)'
    )

    parser.add_argument(
        '--no-json',
        action='store_true',
        help='Skip JSON export (CSV only)'
    )

    args = parser.parse_args()

    # Validate
    if args.days <= 0:
        print("Error: --days must be positive")
        sys.exit(1)

    if args.no_csv and args.no_json:
        print("Error: Cannot skip both CSV and JSON exports")
        sys.exit(1)

    print("=" * 60)
    print("BULL BC1 - BACKTEST RUNNER (LONG Only)")
    print("=" * 60)
    print(f"Mode: ISOLATED (no live trading impact)")
    print(f"Direction: LONG positions only")
    print(f"Period: Last {args.days} days")
    print(f"Timeframe: {args.timeframe}")
    print(f"Output: {args.output}")
    print("=" * 60)

    try:
        # Load config
        print("\nLoading configuration...")
        config = Config.load()

        # Check if backtest module exists
        try:
            from src.backtest import BacktestRunner, BacktestReporter
        except ImportError:
            print("\n[WARN] Backtest module not yet implemented.")
            print("To run backtests, implement:")
            print("  - src/backtest/runner.py")
            print("  - src/backtest/reporter.py")
            print("\nFor now, you can run the main bot in analyze mode:")
            print("  python main.py --analyze")
            sys.exit(0)

        # Create backtest runner
        print("Initializing backtest runner...")
        runner = BacktestRunner(config)

        # Initialize (loads models, connects to data sources)
        success = await runner.initialize()
        if not success:
            print("[FAILED] Initialization failed")
            sys.exit(1)

        print("[OK] Initialization successful")

        # Run backtest
        print(f"\nRunning backtest ({args.days} days)...")
        print("This may take a few minutes depending on period length...\n")

        result = await runner.run(
            days_back=args.days,
            timeframe=args.timeframe
        )

        # Generate reports
        print("\nGenerating reports...")
        reporter = BacktestReporter()

        # Create output directory
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save trades CSV
        if not args.no_csv:
            trades_file = output_dir / f"backtest_trades_{timestamp}.csv"
            reporter.save_trades_csv(result, trades_file)
            print(f"  [OK] Trades CSV: {trades_file}")

            # Save equity curve CSV
            equity_file = output_dir / f"backtest_equity_{timestamp}.csv"
            reporter.save_equity_curve_csv(result, equity_file)
            print(f"  [OK] Equity CSV: {equity_file}")

        # Save summary JSON
        if not args.no_json:
            summary_file = output_dir / f"backtest_summary_{timestamp}.json"
            reporter.save_summary_json(result, summary_file)
            print(f"  [OK] Summary JSON: {summary_file}")

        # Print summary
        reporter.print_summary(result)

        # Clean up
        await runner.close()

        print("\n[OK] Backtest complete!")
        print("\nNOTE: These results are from simulated LONG-only trading.")
        print("Live trading results may differ due to slippage, latency, and market conditions.")

    except KeyboardInterrupt:
        print("\n\n[WARN] Backtest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    asyncio.run(main())
