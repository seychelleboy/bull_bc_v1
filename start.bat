@echo off
REM ============================================================================
REM Bull BC1 - Startup Script (LONG Only)
REM ============================================================================
REM This script handles all prerequisites and starts the bot.
REM Run with: start.bat [--setup] [--analyze] [--status] [--paper]
REM ============================================================================

setlocal enabledelayedexpansion

REM Script directory
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

echo.
echo ============================================================================
echo   BULL BC1 - Bitcoin Bull Detection Bot (LONG Only)
echo ============================================================================
echo.

REM Check for --setup flag
set "SETUP_MODE=0"
set "BOT_ARGS="
for %%a in (%*) do (
    if "%%a"=="--setup" (
        set "SETUP_MODE=1"
    ) else (
        set "BOT_ARGS=!BOT_ARGS! %%a"
    )
)

REM ============================================================================
REM Step 1: Check Python
REM ============================================================================
echo [1/7] Checking Python installation...

where python >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ERROR: Python not found. Please install Python 3.10+
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo       Python version: %PYTHON_VERSION%

REM ============================================================================
REM Step 2: Create/Check Virtual Environment
REM ============================================================================
echo [2/7] Checking virtual environment...

if not exist "venv\Scripts\python.exe" (
    echo       Creating virtual environment...
    python -m venv venv
    if %ERRORLEVEL% neq 0 (
        echo ERROR: Failed to create virtual environment
        exit /b 1
    )
    echo       Virtual environment created
    set "SETUP_MODE=1"
) else (
    echo       Virtual environment exists
)

REM ============================================================================
REM Step 3: Install Dependencies (if setup mode or first run)
REM ============================================================================
echo [3/7] Checking dependencies...

if "%SETUP_MODE%"=="1" (
    echo       Installing/upgrading pip...
    venv\Scripts\python.exe -m pip install --upgrade pip --quiet

    echo       Installing requirements...
    venv\Scripts\pip.exe install -r requirements.txt --quiet
    if %ERRORLEVEL% neq 0 (
        echo ERROR: Failed to install requirements
        exit /b 1
    )

    REM Install MetaTrader5 separately (not in requirements.txt)
    venv\Scripts\pip.exe install MetaTrader5 --quiet

    echo       Dependencies installed
) else (
    echo       Skipping - use --setup to reinstall
)

REM ============================================================================
REM Step 4: Check .env file
REM ============================================================================
echo [4/7] Checking environment configuration...

if not exist "config\.env" (
    if exist "config\.env.example" (
        echo       Creating .env from template...
        copy "config\.env.example" "config\.env" >nul
        echo       WARNING: Please edit config\.env with your API keys
    ) else (
        echo       WARNING: No .env file found. Create config\.env with:
        echo              EODHD_API_KEY=your_key_here
    )
) else (
    echo       .env file exists
)

REM ============================================================================
REM Step 5: Initialize Database
REM ============================================================================
echo [5/7] Checking database...

if not exist "data\database" (
    mkdir "data\database" 2>nul
)

if not exist "data\database\bull_bc_1.db" (
    echo       Initializing database...
    venv\Scripts\python.exe scripts\init_database.py
    if %ERRORLEVEL% neq 0 (
        echo ERROR: Database initialization failed
        exit /b 1
    )
    echo       Database initialized
) else (
    echo       Database exists
)

REM ============================================================================
REM Step 6: Check MT5 Connection
REM ============================================================================
echo [6/7] Checking MT5 connection...

venv\Scripts\python.exe -c "import MetaTrader5 as mt5; mt5.initialize(); info=mt5.terminal_info(); print('       MT5:', info.name if info else 'NOT CONNECTED'); mt5.shutdown()" 2>nul
if %ERRORLEVEL% neq 0 (
    echo       WARNING: MT5 not connected - will use EODHD fallback
)

REM ============================================================================
REM Step 7: Create necessary directories
REM ============================================================================
echo [7/7] Checking directories...

if not exist "data\models" mkdir "data\models" 2>nul
if not exist "data\logs" mkdir "data\logs" 2>nul
if not exist "data\backtest" mkdir "data\backtest" 2>nul
echo       Directories OK

REM ============================================================================
REM Ready to start
REM ============================================================================
echo.
echo ============================================================================
echo   SETUP COMPLETE - Ready to run
echo ============================================================================
echo.

if "%BOT_ARGS%"=="" (
    echo Usage:
    echo   start.bat              - Run this setup check
    echo   start.bat --setup      - Force reinstall dependencies
    echo   start.bat --analyze    - Run single analysis
    echo   start.bat --status     - Show system status
    echo   start.bat --paper      - Run in paper trading mode
    echo   start.bat run          - Start continuous scanning [swing mode, 4h]
    echo   start.bat run scalp    - Start scalping mode [5m timeframe]
    echo.
    exit /b 0
)

REM Check for scalp mode: "run scalp" or just "scalp"
echo %* | findstr /i "scalp" >nul
if %ERRORLEVEL%==0 goto :scalp_mode

REM Check for plain run command
set "CHECK_RUN=!BOT_ARGS: =!"
if /i "!CHECK_RUN!"=="run" goto :swing_mode
if /i "!CHECK_RUN!"=="--run" goto :swing_mode

REM Other arguments (--analyze, --status, --paper, etc.)
echo.
echo Starting Bull BC1...
echo.
venv\Scripts\python.exe main.py !BOT_ARGS!
goto :done

:scalp_mode
echo.
echo Starting Bull BC1 - SCALP MODE [5m timeframe]...
echo.
venv\Scripts\python.exe main.py --mode scalp
goto :done

:swing_mode
echo.
echo Starting Bull BC1 - SWING MODE [4h timeframe]...
echo.
venv\Scripts\python.exe main.py
goto :done

:done

endlocal
