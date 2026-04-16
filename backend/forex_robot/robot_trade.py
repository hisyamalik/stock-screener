import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
import os
import requests
import urllib3
from robot_runtime import configure_logging, load_env_file, parse_bool_env, parse_csv_env, parse_str_env
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


@dataclass
class TelegramConfig:
    bot_token: Optional[str] = None
    chat_id: Optional[str] = None


@dataclass
class BotRuntimeConfig:
    risk_per_trade: float
    magic_number: int
    max_drawdown_percent: float
    drawdown_period_hours: int
    run_duration_minutes: int
    report_period_hours: int
    symbols: List[str] = None
    log_level: str = "INFO"
    log_file: str = "logs/robot_trade.jsonl"
    entry_timeframe: int = mt5.TIMEFRAME_M1
    trend_timeframe: int = mt5.TIMEFRAME_M5
    enable_jpy_tuning: bool = True
    trading_mode: str = "normal"
    sma_short: int = 7
    sma_long: int = 25
    sma_trend: int = 200
    rsi_period: int = 14
    rsi_readybought: float = 65.0
    rsi_overbought: float = 80.0
    rsi_neutral: float = 50.0
    rsi_readysold: float = 35.0
    rsi_oversold: float = 20.0
    atr_period: int = 14
    htf_fast_ma: int = 50
    htf_slow_ma: int = 200
    min_rr_ratio: float = 1.5
    sl_atr_multiplier: float = 1.2
    tp_rr_multiplier: float = 2.0
    max_spread_atr_ratio: float = 0.20
    trend_strength_min: float = 0.20
    rsi_buy_min: float = 45.0
    rsi_buy_max: float = 68.0
    rsi_sell_min: float = 32.0
    rsi_sell_max: float = 55.0

    def __post_init__(self):
        if self.symbols is None:
            self.symbols = ["XAUUSD", "EURUSD"]

def _require_env(name: str) -> str:
    value = os.getenv(name)
    if value is None or str(value).strip() == "":
        raise ValueError(f"Missing required environment variable: {name}")
    return str(value).strip()

def _parse_mt5_timeframe(value: str, var_name: str) -> int:
    label = value.strip().upper()
    mapping = {
        "M1": mt5.TIMEFRAME_M1,
        "M2": getattr(mt5, "TIMEFRAME_M2", mt5.TIMEFRAME_M1),
        "M3": getattr(mt5, "TIMEFRAME_M3", mt5.TIMEFRAME_M1),
        "M4": getattr(mt5, "TIMEFRAME_M4", mt5.TIMEFRAME_M1),
        "M5": mt5.TIMEFRAME_M5,
        "M6": getattr(mt5, "TIMEFRAME_M6", mt5.TIMEFRAME_M5),
        "M10": getattr(mt5, "TIMEFRAME_M10", mt5.TIMEFRAME_M5),
        "M12": getattr(mt5, "TIMEFRAME_M12", mt5.TIMEFRAME_M5),
        "M15": mt5.TIMEFRAME_M15,
        "M20": getattr(mt5, "TIMEFRAME_M20", mt5.TIMEFRAME_M15),
        "M30": mt5.TIMEFRAME_M30,
        "H1": mt5.TIMEFRAME_H1,
        "H2": getattr(mt5, "TIMEFRAME_H2", mt5.TIMEFRAME_H1),
        "H3": getattr(mt5, "TIMEFRAME_H3", mt5.TIMEFRAME_H1),
        "H4": mt5.TIMEFRAME_H4,
        "H6": getattr(mt5, "TIMEFRAME_H6", mt5.TIMEFRAME_H4),
        "H8": getattr(mt5, "TIMEFRAME_H8", mt5.TIMEFRAME_H4),
        "H12": getattr(mt5, "TIMEFRAME_H12", mt5.TIMEFRAME_H4),
        "D1": mt5.TIMEFRAME_D1,
        "W1": mt5.TIMEFRAME_W1,
        "MN1": mt5.TIMEFRAME_MN1,
    }
    if label not in mapping:
        raise ValueError(
            f"Invalid {var_name}='{value}'. Valid values: "
            "M1,M5,M15,M30,H1,H4,D1,W1,MN1 (and broker-supported variants)"
        )
    return mapping[label]

def _timeframe_label(timeframe: int) -> str:
    labels = {
        mt5.TIMEFRAME_M1: "M1",
        mt5.TIMEFRAME_M5: "M5",
        mt5.TIMEFRAME_M15: "M15",
        mt5.TIMEFRAME_M30: "M30",
        mt5.TIMEFRAME_H1: "H1",
        mt5.TIMEFRAME_H4: "H4",
        mt5.TIMEFRAME_D1: "D1",
        mt5.TIMEFRAME_W1: "W1",
        mt5.TIMEFRAME_MN1: "MN1",
    }
    return labels.get(timeframe, str(timeframe))

def get_trading_mode_profile(mode: str) -> Dict[str, float]:
    mode_key = mode.strip().lower()
    profiles: Dict[str, Dict[str, float]] = {
        "normal": {
            "sma_short": 7,
            "sma_long": 25,
            "sma_trend": 200,
            "rsi_period": 14,
            "rsi_readybought": 65.0,
            "rsi_overbought": 80.0,
            "rsi_neutral": 50.0,
            "rsi_readysold": 35.0,
            "rsi_oversold": 20.0,
            "atr_period": 14,
            "htf_fast_ma": 50,
            "htf_slow_ma": 200,
            "min_rr_ratio": 1.5,
            "sl_atr_multiplier": 1.2,
            "tp_rr_multiplier": 2.0,
            "max_spread_atr_ratio": 0.20,
            "trend_strength_min": 0.20,
            "rsi_buy_min": 45.0,
            "rsi_buy_max": 68.0,
            "rsi_sell_min": 32.0,
            "rsi_sell_max": 55.0,
        },
        "conservative": {
            "sma_short": 9,
            "sma_long": 30,
            "sma_trend": 200,
            "rsi_period": 14,
            "rsi_readybought": 62.0,
            "rsi_overbought": 78.0,
            "rsi_neutral": 50.0,
            "rsi_readysold": 38.0,
            "rsi_oversold": 22.0,
            "atr_period": 14,
            "htf_fast_ma": 55,
            "htf_slow_ma": 200,
            "min_rr_ratio": 2.0,
            "sl_atr_multiplier": 1.5,
            "tp_rr_multiplier": 2.8,
            "max_spread_atr_ratio": 0.12,
            "trend_strength_min": 0.26,
            "rsi_buy_min": 48.0,
            "rsi_buy_max": 62.0,
            "rsi_sell_min": 38.0,
            "rsi_sell_max": 52.0,
        },
        "extreme": {
            "sma_short": 5,
            "sma_long": 18,
            "sma_trend": 100,
            "rsi_period": 10,
            "rsi_readybought": 72.0,
            "rsi_overbought": 88.0,
            "rsi_neutral": 50.0,
            "rsi_readysold": 28.0,
            "rsi_oversold": 12.0,
            "atr_period": 10,
            "htf_fast_ma": 34,
            "htf_slow_ma": 144,
            "min_rr_ratio": 1.3,
            "sl_atr_multiplier": 0.9,
            "tp_rr_multiplier": 1.6,
            "max_spread_atr_ratio": 0.28,
            "trend_strength_min": 0.14,
            "rsi_buy_min": 42.0,
            "rsi_buy_max": 74.0,
            "rsi_sell_min": 26.0,
            "rsi_sell_max": 58.0,
        },
    }
    return profiles[mode_key]

def load_runtime_config() -> BotRuntimeConfig:
    load_env_file()
    risk_per_trade = float(_require_env("MT5_RISK_PER_TRADE"))
    magic_number = int(_require_env("MT5_MAGIC_NUMBER"))
    max_drawdown_percent = float(_require_env("MT5_MAX_DRAWDOWN_PERCENT"))
    drawdown_period_hours = int(_require_env("MT5_DRAWDOWN_PERIOD_HOURS"))
    run_duration_minutes = int(_require_env("MT5_RUN_DURATION_MINUTES"))
    report_period_hours = int(_require_env("MT5_REPORT_PERIOD_HOURS"))
    entry_timeframe = _parse_mt5_timeframe(parse_str_env("MT5_ENTRY_TIMEFRAME", "M1"), "MT5_ENTRY_TIMEFRAME")
    trend_timeframe = _parse_mt5_timeframe(parse_str_env("MT5_TREND_TIMEFRAME", "M5"), "MT5_TREND_TIMEFRAME")
    trading_mode = parse_str_env("MT5_TRADING_MODE", "normal").strip().lower()
    if trading_mode not in {"conservative", "normal", "extreme"}:
        raise ValueError("MT5_TRADING_MODE must be one of: conservative, normal, extreme")
    profile = get_trading_mode_profile(trading_mode)
    symbols = [symbol.upper() for symbol in parse_csv_env(_require_env("MT5_SYMBOLS"), [])]
    if not symbols:
        raise ValueError("MT5_SYMBOLS must contain at least one symbol")

    return BotRuntimeConfig(
        risk_per_trade=risk_per_trade,
        magic_number=magic_number,
        max_drawdown_percent=max_drawdown_percent,
        drawdown_period_hours=drawdown_period_hours,
        run_duration_minutes=run_duration_minutes,
        report_period_hours=report_period_hours,
        symbols=symbols,
        entry_timeframe=entry_timeframe,
        trend_timeframe=trend_timeframe,
        log_level=parse_str_env("ROBOT_LOG_LEVEL", "INFO"),
        log_file=parse_str_env("ROBOT_LOG_FILE", "logs/robot_trade.jsonl"),
        enable_jpy_tuning=parse_bool_env(parse_str_env("MT5_ENABLE_JPY_TUNING", None), True),
        trading_mode=trading_mode,
        sma_short=int(profile["sma_short"]),
        sma_long=int(profile["sma_long"]),
        sma_trend=int(profile["sma_trend"]),
        rsi_period=int(profile["rsi_period"]),
        rsi_readybought=float(profile["rsi_readybought"]),
        rsi_overbought=float(profile["rsi_overbought"]),
        rsi_neutral=float(profile["rsi_neutral"]),
        rsi_readysold=float(profile["rsi_readysold"]),
        rsi_oversold=float(profile["rsi_oversold"]),
        atr_period=int(profile["atr_period"]),
        htf_fast_ma=int(profile["htf_fast_ma"]),
        htf_slow_ma=int(profile["htf_slow_ma"]),
        min_rr_ratio=float(profile["min_rr_ratio"]),
        sl_atr_multiplier=float(profile["sl_atr_multiplier"]),
        tp_rr_multiplier=float(profile["tp_rr_multiplier"]),
        max_spread_atr_ratio=float(profile["max_spread_atr_ratio"]),
        trend_strength_min=float(profile["trend_strength_min"]),
        rsi_buy_min=float(profile["rsi_buy_min"]),
        rsi_buy_max=float(profile["rsi_buy_max"]),
        rsi_sell_min=float(profile["rsi_sell_min"]),
        rsi_sell_max=float(profile["rsi_sell_max"]),
    )


def load_telegram_config() -> TelegramConfig:
    return TelegramConfig(
        bot_token=parse_str_env("TELEGRAM_BOT_TOKEN", ""),
        chat_id=parse_str_env("TELEGRAM_CHAT_ID", ""),
    )


logger = logging.getLogger(__name__)

class MT5ForexRobot:
    def __init__(self, risk_per_trade: float, magic_number: int,
                 max_drawdown_percent: float, drawdown_period_hours: int,
                 entry_timeframe: int, trend_timeframe: int,
                 enable_jpy_tuning: bool, report_period_hours: int, trading_mode: str,
                 sma_short: int, sma_long: int, sma_trend: int,
                 rsi_period: int, rsi_readybought: float, rsi_overbought: float, rsi_neutral: float,
                 rsi_readysold: float, rsi_oversold: float, atr_period: int,
                 htf_fast_ma: int, htf_slow_ma: int, min_rr_ratio: float, sl_atr_multiplier: float,
                 tp_rr_multiplier: float, max_spread_atr_ratio: float, trend_strength_min: float,
                 rsi_buy_min: float, rsi_buy_max: float, rsi_sell_min: float, rsi_sell_max: float,
                 telegram_config: Optional[TelegramConfig] = None):
        """
        Initialize the MT5 Forex Trading Robot
        
        Args:
            risk_per_trade: Risk percentage per trade (e.g., 0.02 = 2%)
            magic_number: Unique identifier for this robot's trades
            max_drawdown_percent: Maximum allowed equity drawdown percentage
            drawdown_period_hours: Period in hours to monitor drawdown
            entry_timeframe: Entry/trigger timeframe for positions (e.g. M1, M5, M15)
            trend_timeframe: Higher timeframe for trend filtering (M30, H1)
            enable_jpy_tuning: Enable/disable JPY pair profile adjustments
            report_period_hours: Default lookback window for trade reports
            trading_mode: Strategy profile (conservative, normal, extreme)
            telegram_config: Telegram delivery settings loaded from environment
        """
        self.risk_per_trade = risk_per_trade
        self.magic_number = magic_number
        self.is_running = False
        self.entry_timeframe = entry_timeframe
        self.timeframe = entry_timeframe  # Backward-compat alias
        self.trend_timeframe = trend_timeframe
        self.report_period_hours = report_period_hours
        self.telegram_config = telegram_config or TelegramConfig()
        self.trading_mode = trading_mode.strip().lower()
        
        # Trading parameters (resolved from selected trading mode profile)
        self.sma_short = sma_short
        self.sma_long = sma_long
        self.sma_trend = sma_trend
        self.rsi_period = rsi_period
        self.rsi_readybought = rsi_readybought
        self.rsi_overbought = rsi_overbought
        self.rsi_neutral = rsi_neutral
        self.rsi_readysold = rsi_readysold
        self.rsi_oversold = rsi_oversold
        self.atr_period = atr_period
        self.htf_fast_ma = htf_fast_ma
        self.htf_slow_ma = htf_slow_ma
        self.min_rr_ratio = min_rr_ratio
        self.sl_atr_multiplier = sl_atr_multiplier
        self.tp_rr_multiplier = tp_rr_multiplier
        self.max_spread_atr_ratio = max_spread_atr_ratio
        self.trend_strength_min = trend_strength_min
        self.rsi_buy_min = rsi_buy_min
        self.rsi_buy_max = rsi_buy_max
        self.rsi_sell_min = rsi_sell_min
        self.rsi_sell_max = rsi_sell_max
        self.enable_jpy_tuning = enable_jpy_tuning
        
        # Drawdown protection parameters
        self.max_drawdown_percent = max_drawdown_percent
        self.drawdown_period_hours = drawdown_period_hours
        self.equity_history = []  # Store equity snapshots with timestamps
        self.initial_equity = 0.0
        self.peak_equity = 0.0
        self.trading_stopped_due_to_drawdown = False
        self.tracked_positions: Dict[int, Dict] = {}
        self.last_history_check = datetime.now() - timedelta(minutes=10)
        
        # Initialize MT5 connection
        self.initialize_mt5()

    def initialize_mt5(self) -> bool:
        """Initialize MT5 connection"""
        try:
            # Initialize MT5 connection
            if not mt5.initialize():
                logger.error(f"initialize() failed, error code = {mt5.last_error()}")
                return False
            
            # Get account info
            account_info = mt5.account_info()
            if account_info is None:
                logger.error("Failed to get account info")
                return False
            
            logger.info(f"Connected to MT5 account: {account_info.login}")
            logger.info(f"Account balance: ${account_info.balance:.2f}")
            logger.info(f"Account equity: ${account_info.equity:.2f}")
            
            # Initialize equity tracking
            self.initial_equity = account_info.equity
            self.peak_equity = account_info.equity
            self.server_time = mt5.symbol_info("EURUSD").time
            self.record_equity_snapshot()
            
            logger.info(f"Drawdown protection: Max {self.max_drawdown_percent}% over {self.drawdown_period_hours} hours")
            
            return True
            
        except Exception as e:
            logger.error(f"MT5 initialization failed: {e}")
            return False
    
    def get_account_balance(self) -> float:
        """Get current account balance"""
        account_info = mt5.account_info()
        return account_info.balance if account_info else 0.0
    
    def get_account_equity(self) -> float:
        """Get current account equity"""
        account_info = mt5.account_info()
        return account_info.equity if account_info else 0.0
    
    def record_equity_snapshot(self):
        """Record current equity with timestamp"""
        current_equity = self.get_account_equity()
        current_time = datetime.now()
        
        self.equity_history.append({
            'timestamp': current_time,
            'equity': current_equity
        })
        
        # Update peak equity
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
        
        # Clean old equity records (keep only records within the monitoring period)
        cutoff_time = current_time - timedelta(hours=self.drawdown_period_hours)
        self.equity_history = [
            record for record in self.equity_history 
            if record['timestamp'] > cutoff_time
        ]
    
    def check_drawdown_protection(self) -> bool:
        """
        Check if current drawdown exceeds the maximum allowed percentage
        
        Returns:
            True if trading should be stopped due to drawdown, False otherwise
        """
        if not self.equity_history:
            return False
        
        current_equity = self.get_account_equity()
        
        # Method 1: Check drawdown from peak equity within the period
        if self.equity_history:
            period_peak = max(record['equity'] for record in self.equity_history)
            drawdown_from_peak = ((period_peak - current_equity) / period_peak) * 100
        else:
            drawdown_from_peak = 0.0
        
        # Method 2: Check drawdown from initial equity
        drawdown_from_initial = ((self.initial_equity - current_equity) / self.initial_equity) * 100
        
        # Method 3: Check drawdown from period start
        if len(self.equity_history) > 1:
            period_start_equity = self.equity_history[0]['equity']
            drawdown_from_period_start = ((period_start_equity - current_equity) / period_start_equity) * 100
        else:
            drawdown_from_period_start = 0.0
        
        # Use the most conservative (highest) drawdown
        max_drawdown = max(drawdown_from_peak, drawdown_from_initial, drawdown_from_period_start)
        
        logger.info(f"Equity Monitor - Current: ${current_equity:.2f}, Peak: ${period_peak:.2f}, "
                   f"Drawdown: {max_drawdown:.2f}% (Max allowed: {self.max_drawdown_percent}%)")
        
        if max_drawdown >= self.max_drawdown_percent:
            logger.critical(f"⚠️  DRAWDOWN PROTECTION TRIGGERED! ⚠️")
            logger.critical(f"Current drawdown: {max_drawdown:.2f}% exceeds maximum allowed: {self.max_drawdown_percent}%")
            logger.critical(f"Trading will be stopped and all positions will be closed!")
            
            self.trading_stopped_due_to_drawdown = True
            return True
        
        return False
    
    def close_all_positions(self) -> bool:
        """Emergency close all positions"""
        try:
            positions = self.get_current_positions()
            if not positions:
                logger.info("No positions to close")
                return True
            
            logger.info(f"Emergency closing {len(positions)} positions...")
            
            for position in positions:
                try:
                    # Determine order type for closing
                    if position.type == mt5.POSITION_TYPE_BUY:
                        order_type = mt5.ORDER_TYPE_SELL
                        price = mt5.symbol_info_tick(position.symbol).bid
                    else:
                        order_type = mt5.ORDER_TYPE_BUY  
                        price = mt5.symbol_info_tick(position.symbol).ask
                    
                    # Prepare close request
                    request = {
                        "action": mt5.TRADE_ACTION_DEAL,
                        "symbol": position.symbol,
                        "volume": position.volume,
                        "type": order_type,
                        "position": position.ticket,
                        "price": price,
                        "magic": self.magic_number,
                        "comment": "Emergency close - Drawdown protection",
                        "type_time": mt5.ORDER_TIME_GTC,
                        "type_filling": mt5.ORDER_FILLING_IOC,
                    }
                    
                    result = mt5.order_send(request)
                    
                    if result.retcode == mt5.TRADE_RETCODE_DONE:
                        logger.info(f"✅ Closed position {position.symbol}: P&L = ${position.profit:.2f}")
                    else:
                        logger.error(f"❌ Failed to close position {position.symbol}: {result.comment}")
                        
                except Exception as e:
                    logger.error(f"Error closing position {position.symbol}: {e}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error in emergency close all positions: {e}")
            return False
    
    def get_market_data(self, symbol: str, timeframe: Optional[int] = None, count: int = 1000):
        """
        Get historical market data from MT5.

        timeframe:
            - None => uses self.entry_timeframe
            - int  => explicit MT5 timeframe override
        """
        try:
            tf = timeframe if timeframe is not None else self.entry_timeframe
            rates = mt5.copy_rates_from_pos(symbol, tf, 0, count)
            if rates is None:
                logger.error(f"Failed to get rates for {symbol}")
                return None
            
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            return df
            
        except Exception as e:
            logger.error(f"Error getting market data for {symbol}: {e}")
            return None
    
    def calculate_sma(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate Simple Moving Average"""
        return prices.rolling(window=period).mean()
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index (Wilder's Smoothing)"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # Wilder's Smoothing using EWM
        avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr
    
    def find_local_extrema(self, series: pd.Series, order: int = 5, mode="max"):
        """
        Detect local maxima/minima without scipy.
        mode = "max" for highs, "min" for lows
        """
        if mode == "max":
            return series[(series.shift(order) < series) & (series.shift(-order) < series)].index
        else:  # mode == "min"
            return series[(series.shift(order) > series) & (series.shift(-order) > series)].index
        

    def detect_rsi_divergence(self, prices: pd.Series, rsi: pd.Series, order: int = 5) -> str:
        # Find local highs and lows
        price_highs = self.find_local_extrema(prices, order, mode="max")
        price_lows  = self.find_local_extrema(prices, order, mode="min")
        rsi_highs   = self.find_local_extrema(rsi, order, mode="max")
        rsi_lows    = self.find_local_extrema(rsi, order, mode="min")

        # --- Bullish divergence ---
        for i in range(1, len(price_lows)):
            p1, p2 = price_lows[i-1], price_lows[i]
            if p1 in rsi_lows and p2 in rsi_lows:
                if prices[p2] < prices[p1] and rsi[p2] > rsi[p1]:
                    return "bullish"

        # --- Bearish divergence ---
        for i in range(1, len(price_highs)):
            p1, p2 = price_highs[i-1], price_highs[i]
            if p1 in rsi_highs and p2 in rsi_highs:
                if prices[p2] > prices[p1] and rsi[p2] < rsi[p1]:
                    return "bearish"

        return "none"

    def detect_crosses(self, prices: pd.Series) -> pd.Series:
        """
        Detect golden cross and death cross events.
        Returns a Series with:
          1  -> Golden Cross (bullish)
         -1  -> Death Cross (bearish)
          0  -> No cross
        """
        short_sma = self.calculate_sma(prices, self.sma_short)
        long_sma = self.calculate_sma(prices, self.sma_long)

        golden_cross = (short_sma > long_sma) & (short_sma.shift(1) <= long_sma.shift(1))
        death_cross = (short_sma < long_sma) & (short_sma.shift(1) >= long_sma.shift(1))

        signals = pd.Series(0, index=prices.index)
        signals[golden_cross] = 1
        signals[death_cross] = -1
        return signals

    def get_signal_profile(self, symbol: str) -> Dict[str, float]:
        """Return per-symbol thresholds for trend-following behavior."""
        profile = {
            "htf_fast_ma": float(self.htf_fast_ma),
            "htf_slow_ma": float(self.htf_slow_ma),
            "min_rr_ratio": float(self.min_rr_ratio),
            "sl_atr_multiplier": float(self.sl_atr_multiplier),
            "tp_rr_multiplier": float(self.tp_rr_multiplier),
            "max_spread_atr_ratio": float(self.max_spread_atr_ratio),
            "trend_strength_min": float(self.trend_strength_min),
            "rsi_buy_min": float(self.rsi_buy_min),
            "rsi_buy_max": float(self.rsi_buy_max),
            "rsi_sell_min": float(self.rsi_sell_min),
            "rsi_sell_max": float(self.rsi_sell_max),
        }

        symbol_upper = symbol.upper()
        if symbol_upper == "XAUUSD":
            # Gold is noisier and wider-spread; use stricter trend and wider risk buffers.
            profile.update({
                "min_rr_ratio": 1.8,
                "sl_atr_multiplier": 1.6,
                "tp_rr_multiplier": 2.4,
                "max_spread_atr_ratio": 0.12,
                "trend_strength_min": 0.30,
                "rsi_buy_min": 48.0,
                "rsi_buy_max": 64.0,
                "rsi_sell_min": 36.0,
                "rsi_sell_max": 52.0,
            })
        elif symbol_upper == "EURUSD":
            # EURUSD typically rewards slightly tighter trend thresholds and spread filtering.
            profile.update({
                "min_rr_ratio": 1.6,
                "sl_atr_multiplier": 1.3,
                "tp_rr_multiplier": 2.2,
                "max_spread_atr_ratio": 0.08,
                "trend_strength_min": 0.22,
                "rsi_buy_min": 46.0,
                "rsi_buy_max": 66.0,
                "rsi_sell_min": 34.0,
                "rsi_sell_max": 54.0,
            })
        elif self.enable_jpy_tuning and symbol_upper.endswith("JPY"):
            profile.update({
                "sl_atr_multiplier": 1.3,
                "tp_rr_multiplier": 2.1,
                "trend_strength_min": 0.24,
            })

        return profile

    def generate_trend_following_signal(self, symbol: str) -> Dict[str, Any]:
        """
        Generate disciplined trend-following signal and risk plan.

        Returns:
            {
                "signal": "BUY" | "SELL" | "HOLD",
                "entry_price": float | None,
                "stop_loss": float | None,
                "take_profit": float | None,
                "rr_ratio": float | None,
                "reason": str
            }
        """
        hold_result = {
            "signal": "HOLD",
            "entry_price": None,
            "stop_loss": None,
            "take_profit": None,
            "rr_ratio": None,
            "reason": "No valid setup",
        }

        try:
            profile = self.get_signal_profile(symbol)
            symbol_info = mt5.symbol_info(symbol)
            tick = mt5.symbol_info_tick(symbol)
            if not symbol_info or not tick:
                hold_result["reason"] = "Missing symbol info/tick"
                return hold_result

            htf_fast_ma = int(profile["htf_fast_ma"])
            htf_slow_ma = int(profile["htf_slow_ma"])
            min_rr_ratio = float(profile["min_rr_ratio"])
            sl_atr_multiplier = float(profile["sl_atr_multiplier"])
            tp_rr_multiplier = float(profile["tp_rr_multiplier"])
            max_spread_atr_ratio = float(profile["max_spread_atr_ratio"])
            trend_strength_min = float(profile["trend_strength_min"])
            rsi_buy_min = float(profile["rsi_buy_min"])
            rsi_buy_max = float(profile["rsi_buy_max"])
            rsi_sell_min = float(profile["rsi_sell_min"])
            rsi_sell_max = float(profile["rsi_sell_max"])

            htf_count = max(htf_slow_ma + 20, self.atr_period + 20)
            ltf_count = max(self.sma_long + 30, self.atr_period + 20)

            df_htf = self.get_market_data(symbol, timeframe=self.trend_timeframe, count=htf_count)
            df_ltf = self.get_market_data(symbol, timeframe=self.entry_timeframe, count=ltf_count)
            if df_htf is None or df_ltf is None:
                hold_result["reason"] = "Insufficient market data"
                return hold_result

            if len(df_htf) < htf_slow_ma or len(df_ltf) < self.sma_long:
                hold_result["reason"] = "Not enough candles"
                return hold_result

            # Higher timeframe trend filter
            df_htf["ma_fast"] = self.calculate_sma(df_htf["close"], htf_fast_ma)
            df_htf["ma_slow"] = self.calculate_sma(df_htf["close"], htf_slow_ma)
            df_htf["atr"] = self.calculate_atr(df_htf["high"], df_htf["low"], df_htf["close"], self.atr_period)

            # Lower timeframe entry filter
            df_ltf["sma_short"] = self.calculate_sma(df_ltf["close"], self.sma_short)
            df_ltf["sma_long"] = self.calculate_sma(df_ltf["close"], self.sma_long)
            df_ltf["rsi"] = self.calculate_rsi(df_ltf["close"], self.rsi_period)
            df_ltf["atr"] = self.calculate_atr(df_ltf["high"], df_ltf["low"], df_ltf["close"], self.atr_period)

            htf_latest = df_htf.iloc[-1]
            htf_prev = df_htf.iloc[-2]
            ltf_latest = df_ltf.iloc[-1]
            ltf_prev = df_ltf.iloc[-2]

            required_values = [
                htf_latest["ma_fast"], htf_latest["ma_slow"], htf_latest["atr"],
                ltf_latest["sma_short"], ltf_latest["sma_long"], ltf_latest["rsi"], ltf_latest["atr"],
            ]
            if any(pd.isna(value) for value in required_values):
                hold_result["reason"] = "Indicators not ready"
                return hold_result

            htf_atr = float(htf_latest["atr"])
            ltf_atr = float(ltf_latest["atr"])
            if htf_atr <= 0 or ltf_atr <= 0:
                hold_result["reason"] = "ATR invalid"
                return hold_result

            spread = float(tick.ask - tick.bid)
            if spread > (max_spread_atr_ratio * ltf_atr):
                hold_result["reason"] = f"Spread too high ({spread:.5f})"
                return hold_result

            trend_strength = abs(float(htf_latest["ma_fast"] - htf_latest["ma_slow"])) / htf_atr
            bullish_trend = (
                htf_latest["close"] > htf_latest["ma_slow"]
                and htf_latest["ma_fast"] > htf_latest["ma_slow"]
                and htf_latest["ma_fast"] > htf_prev["ma_fast"]
                and trend_strength >= trend_strength_min
            )
            bearish_trend = (
                htf_latest["close"] < htf_latest["ma_slow"]
                and htf_latest["ma_fast"] < htf_latest["ma_slow"]
                and htf_latest["ma_fast"] < htf_prev["ma_fast"]
                and trend_strength >= trend_strength_min
            )

            bullish_entry = (
                ltf_latest["sma_short"] > ltf_latest["sma_long"]
                and ltf_prev["low"] <= ltf_prev["sma_short"]
                and ltf_latest["close"] > ltf_latest["open"]
                and rsi_buy_min <= ltf_latest["rsi"] <= rsi_buy_max
            )
            bearish_entry = (
                ltf_latest["sma_short"] < ltf_latest["sma_long"]
                and ltf_prev["high"] >= ltf_prev["sma_short"]
                and ltf_latest["close"] < ltf_latest["open"]
                and rsi_sell_min <= ltf_latest["rsi"] <= rsi_sell_max
            )

            signal = "HOLD"
            if bullish_trend and bullish_entry:
                signal = "BUY"
            elif bearish_trend and bearish_entry:
                signal = "SELL"

            if signal == "HOLD":
                hold_result["reason"] = "Trend/entry not aligned"
                logger.info(
                    f"Market Scan: {symbol} | Trend strength={trend_strength:.2f} | "
                    f"RSI={ltf_latest['rsi']:.2f} | Signal=HOLD"
                )
                return hold_result

            rr_ratio = max(min_rr_ratio, tp_rr_multiplier)
            stops_level_points = getattr(symbol_info, "trade_stops_level", 0) or 0
            min_stop_distance = float(stops_level_points * symbol_info.point)
            sl_distance = max(sl_atr_multiplier * ltf_atr, min_stop_distance, spread * 1.5)
            tp_distance = sl_distance * rr_ratio

            digits = int(symbol_info.digits)
            if signal == "BUY":
                entry_price = float(tick.ask)
                stop_loss = round(entry_price - sl_distance, digits)
                take_profit = round(entry_price + tp_distance, digits)
            else:
                entry_price = float(tick.bid)
                stop_loss = round(entry_price + sl_distance, digits)
                take_profit = round(entry_price - tp_distance, digits)

            if signal == "BUY" and not (stop_loss < entry_price < take_profit):
                hold_result["reason"] = "Invalid BUY levels"
                return hold_result
            if signal == "SELL" and not (take_profit < entry_price < stop_loss):
                hold_result["reason"] = "Invalid SELL levels"
                return hold_result

            logger.info(
                f"Signal: {symbol} {signal} | entry={entry_price:.5f}, "
                f"SL={stop_loss:.5f}, TP={take_profit:.5f}, RR={rr_ratio:.2f}, "
                f"spread={spread:.5f}, trend_strength={trend_strength:.2f}, "
                f"profile_rr_min={min_rr_ratio:.2f}"
            )

            return {
                "signal": signal,
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "rr_ratio": rr_ratio,
                "min_rr_ratio": min_rr_ratio,
                "reason": f"{signal} trend-following setup",
            }

        except Exception as e:
            logger.error(f"Error generating trend-following signal for {symbol}: {e}")
            hold_result["reason"] = str(e)
            return hold_result
    
    def analyze_market(self, symbol: str) -> str:
        """
        Analyze market conditions and generate trading signal with HTF trend filtering.
        Optimized for M5-M30 scalping/day-trading.
        """
        try:
            signal_plan = self.generate_trend_following_signal(symbol)
            return signal_plan.get("signal", "HOLD")

        except Exception as e:
            logger.error(f"Error in analyze_market: {e}")
            return 'HOLD'

    def calculate_position_size(self, symbol: str, entry_price: float, stop_loss: float) -> float:
        """Calculate position size based on risk management"""
        try:
            account_info = mt5.account_info()
            if not account_info:
                return 0.0
            
            balance = account_info.balance
            risk_amount = balance * self.risk_per_trade
            
            # Get symbol info for lot size calculation
            symbol_info = mt5.symbol_info(symbol)
            if not symbol_info:
                return 0.0
            
            # Calculate pip value and position size
            pip_value = symbol_info.trade_tick_value
            pips_risked = abs(entry_price - stop_loss) / symbol_info.point
            
            if pips_risked == 0:
                return 0.0
            
            # Calculate lot size
            lot_size = risk_amount / (pips_risked * pip_value)
            
            # Ensure lot size is within broker limits
            lot_size = max(symbol_info.volume_min, 
                          min(lot_size, symbol_info.volume_max))
            
            # Round to step size
            step = symbol_info.volume_step
            lot_size = round(lot_size / step) * step
            
            return lot_size
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0.0
    
    def get_current_positions(self) -> List:
        """Get current open positions"""
        positions = mt5.positions_get()
        return [pos for pos in positions if pos.magic == self.magic_number] if positions else []

    def sync_positions_with_mt5(self) -> List[Dict]:
        """
        Reconcile locally tracked positions with MT5 live positions.
        Detects positions closed manually from the MT5 app or elsewhere.

        Returns:
            A list of position-close events detected during this sync.
        """
        try:
            current_positions = self.get_current_positions()
            current_map = {position.ticket: position for position in current_positions}
            closed_tickets = set(self.tracked_positions) - set(current_map)
            close_events = []

            for ticket in closed_tickets:
                tracked = self.tracked_positions.pop(ticket)
                close_info = {}
                for _ in range(3):
                    close_info = self.get_position_close_info(ticket, tracked["symbol"])
                    if close_info:
                        break
                    # MT5 history can lag briefly after manual close; retry quickly.
                    time.sleep(0.4)
                event = {
                    "ticket": ticket,
                    "symbol": tracked["symbol"],
                    "volume": tracked["volume"],
                    "close_reason": close_info.get("reason", "unknown"),
                    "profit": close_info.get("profit"),
                    "closed_at": close_info.get("time"),
                }
                close_events.append(event)

                profit_text = (
                    f"${event['profit']:.2f}"
                    if event["profit"] is not None
                    else "unknown"
                )
                closed_at_text = (
                    event["closed_at"].strftime("%Y-%m-%d %H:%M:%S")
                    if event["closed_at"] is not None
                    else "unknown time"
                )
                logger.warning(
                    f"Position closed outside robot flow: {event['symbol']} "
                    f"(ticket={ticket}, reason={event['close_reason']}, "
                    f"profit={profit_text}, closed_at={closed_at_text})"
                )

            for position in current_positions:
                self.tracked_positions[position.ticket] = {
                    "symbol": position.symbol,
                    "volume": position.volume,
                    "type": position.type,
                    "opened_at": datetime.fromtimestamp(position.time),
                }

            self.last_history_check = datetime.now()
            return close_events

        except Exception as e:
            logger.error(f"Error syncing MT5 positions: {e}")
            return []

    def _map_deal_reason(self, reason_code: Optional[int]) -> str:
        """Convert MT5 deal reason code into readable text."""
        deal_reason_map = {}
        for attr_name, label in [
            ("DEAL_REASON_CLIENT", "manual/client"),
            ("DEAL_REASON_EXPERT", "expert/robot"),
            ("DEAL_REASON_SL", "stop loss"),
            ("DEAL_REASON_TP", "take profit"),
            ("DEAL_REASON_SO", "stop out"),
            ("DEAL_REASON_ROLLOVER", "rollover"),
            ("DEAL_REASON_VMARGIN", "variation margin"),
            ("DEAL_REASON_SPLIT", "split"),
        ]:
            attr_value = getattr(mt5, attr_name, None)
            if attr_value is not None:
                deal_reason_map[attr_value] = label
        return deal_reason_map.get(reason_code, f"reason_{reason_code if reason_code is not None else 'unknown'}")

    def get_position_close_info(self, position_ticket: int, symbol: str) -> Dict:
        """Get the latest close deal info for a specific position ticket."""
        try:
            end_time = datetime.now() + timedelta(minutes=1)
            close_entries = (mt5.DEAL_ENTRY_OUT, getattr(mt5, "DEAL_ENTRY_OUT_BY", -1))

            # First: narrow search around the last sync to keep this fast.
            search_windows = [
                (self.last_history_check - timedelta(minutes=10), end_time),
                # Fallback: broader window for delayed history propagation/manual closures.
                (datetime.now() - timedelta(hours=24), end_time),
            ]

            for start_time, end_time_window in search_windows:
                deals = mt5.history_deals_get(start_time, end_time_window)
                if not deals:
                    continue

                closing_deals = []
                for deal in deals:
                    position_id = getattr(deal, "position_id", None)
                    if position_id != position_ticket:
                        continue
                    if getattr(deal, "entry", None) not in close_entries:
                        continue
                    if symbol and getattr(deal, "symbol", None) != symbol:
                        continue
                    closing_deals.append(deal)

                if not closing_deals:
                    continue

                last_close = max(closing_deals, key=lambda deal: deal.time)
                net_profit = (
                    (getattr(last_close, "profit", 0.0) or 0.0) +
                    (getattr(last_close, "commission", 0.0) or 0.0) +
                    (getattr(last_close, "swap", 0.0) or 0.0) +
                    (getattr(last_close, "fee", 0.0) or 0.0)
                )

                return {
                    "profit": net_profit,
                    "time": datetime.fromtimestamp(last_close.time),
                    "reason": self._map_deal_reason(getattr(last_close, "reason", None)),
                }

            return {}

        except Exception as e:
            logger.error(f"Error reading close info for ticket {position_ticket}: {e}")
            return {}
    
    def place_order(self, symbol: str, order_type: int, volume: float, price: float, 
                   stop_loss: float = None, take_profit: float = None) -> bool:
        """Place a trading order"""
        try:
            symbol_info = mt5.symbol_info(symbol)
            if not symbol_info:
                logger.error(f"Symbol {symbol} not found")
                return False
            
            # Check if symbol is available for trading
            if not symbol_info.visible:
                logger.error(f"Symbol {symbol} is not visible")
                return False
                
            # Determine supported filling mode
            filling_mode = None
            if symbol_info.filling_mode & mt5.ORDER_FILLING_IOC:
                filling_mode = mt5.ORDER_FILLING_IOC
            elif symbol_info.filling_mode & mt5.ORDER_FILLING_RETURN:
                filling_mode = mt5.ORDER_FILLING_RETURN
            elif symbol_info.filling_mode & mt5.ORDER_FILLING_FOK:
                filling_mode = mt5.ORDER_FILLING_FOK
            else:
                filling_mode = None

            if filling_mode is None:
                raise RuntimeError("No supported filling mode for this symbol")
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": order_type,
                "price": price,
                "sl": stop_loss,
                "tp": take_profit,
                "magic": self.magic_number,
                "comment": "Python Forex Robot",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_FOK,
            }
            
            # Send order
            result = mt5.order_send(request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(f"Order failed: {result.retcode} - {result.comment}")
                return False
            
            logger.info(f"Order successful: {result.order} - {result.comment}")
            return True
            
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return False
    
    def execute_trade(self, symbol: str, signal: str, signal_plan: Optional[Dict[str, Any]] = None) -> bool:
        """Execute a trade based on signal"""
        try:
            # Check drawdown protection first
            if self.trading_stopped_due_to_drawdown:
                logger.warning(f"Trading stopped due to drawdown protection - ignoring {signal} signal for {symbol}")
                return False

            # Check if we already have a position for this symbol
            existing_positions = [pos for pos in self.get_current_positions() if pos.symbol == symbol]
            if existing_positions:
                logger.info(f"Already have position for {symbol}")
                return False

            plan = signal_plan or self.generate_trend_following_signal(symbol)
            plan_signal = plan.get("signal")
            if plan_signal != signal or signal not in ["BUY", "SELL"]:
                logger.info(f"Signal plan mismatch or HOLD for {symbol}. Requested={signal}, Plan={plan_signal}")
                return False

            entry_price = plan.get("entry_price")
            stop_loss = plan.get("stop_loss")
            take_profit = plan.get("take_profit")
            rr_ratio = plan.get("rr_ratio") or 0.0
            min_rr_ratio = float(plan.get("min_rr_ratio") or self.min_rr_ratio)
            if entry_price is None or stop_loss is None or take_profit is None:
                logger.warning(f"Missing SL/TP plan for {symbol}: {plan.get('reason')}")
                return False

            risk_distance = abs(entry_price - stop_loss)
            reward_distance = abs(take_profit - entry_price)
            if risk_distance <= 0:
                logger.warning(f"Invalid risk distance for {symbol}")
                return False

            actual_rr = reward_distance / risk_distance
            if actual_rr < min_rr_ratio:
                logger.warning(
                    f"RR filter blocked {symbol} {signal}: actual RR {actual_rr:.2f} < minimum {min_rr_ratio:.2f}"
                )
                return False

            volume = self.calculate_position_size(symbol, entry_price, stop_loss)
            if volume <= 0:
                logger.warning(f"Calculated volume invalid for {symbol}")
                return False

            order_type = mt5.ORDER_TYPE_BUY if signal == "BUY" else mt5.ORDER_TYPE_SELL
            success = self.place_order(
                symbol=symbol,
                order_type=order_type,
                volume=volume,
                price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit
            )

            if success:
                logger.info(
                    f"{signal} order placed for {symbol}: Volume={volume}, "
                    f"Entry={entry_price:.5f}, SL={stop_loss:.5f}, TP={take_profit:.5f}, "
                    f"RR={actual_rr:.2f} (planned {rr_ratio:.2f})"
                )
            return success

        except Exception as e:
            logger.error(f"Error executing trade for {symbol}: {e}")
            return False

    def monitor_positions(self):
        """Monitor and manage existing positions"""
        try:
            self.sync_positions_with_mt5()
            positions = self.get_current_positions()
            
            for position in positions:
                # Log position info
                profit = position.profit
                logger.info(f"Position {position.symbol}: Type={position.type}, Volume={position.volume}, Profit=${profit:.2f}")
                
                # You can add additional position management logic here
                # For example, trailing stops, partial closes, etc.
                
        except Exception as e:
            logger.error(f"Error monitoring positions: {e}")
    
    def check_and_handle_drawdown(self) -> bool:
        """
        Check drawdown and handle emergency stop if needed
        
        Returns:
            True if trading should continue, False if stopped due to drawdown
        """
        # Record current equity
        self.record_equity_snapshot()
        
        # Check if drawdown protection should trigger
        if self.check_drawdown_protection():
            # Close all positions immediately
            self.close_all_positions()
            
            # Stop the trading bot
            self.is_running = False
            
            logger.critical("🛑 TRADING BOT STOPPED DUE TO DRAWDOWN PROTECTION 🛑")
            return False
        
        return True
    
    def run_trading_bot(self, symbols: List[str], run_duration_minutes: int = 60):
        """
        Run the trading bot
        
        Args:
            symbols: List of currency pairs to trade
            run_duration_minutes: How long to run the bot
        """
        logger.info(f"Starting MT5 Forex Robot for {run_duration_minutes} minutes")
        logger.info(f"Trading symbols: {symbols}")
        
        self.is_running = True
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=run_duration_minutes)
        
        try:
            while self.is_running and datetime.now() < end_time:
                logger.info("--- Trading Cycle ---")

                # Keep local position state aligned with MT5, including manual closes
                self.sync_positions_with_mt5()
                
                # PRIORITY: Check drawdown protection first
                if not self.check_and_handle_drawdown():
                    logger.critical("Trading stopped due to drawdown protection!")
                    break
                
                # Monitor existing positions
                self.monitor_positions()
                
                # Analyze each symbol and execute trades (only if drawdown protection hasn't triggered)
                if not self.trading_stopped_due_to_drawdown:
                    for symbol in symbols:
                        try:
                            signal_plan = self.generate_trend_following_signal(symbol)
                            signal = signal_plan.get("signal", "HOLD")
                            logger.info(f"{symbol}: Signal = {signal}")
                            
                            if signal in ['BUY', 'SELL']:
                                self.execute_trade(symbol, signal, signal_plan)
                                
                        except Exception as e:
                            logger.error(f"Error processing {symbol}: {e}")
                else:
                    logger.warning("New trades disabled due to drawdown protection")
                
                # Wait before next cycle (adjust based on your strategy)
                time.sleep(60)  # Wait 1 minute between cycles
                
        except KeyboardInterrupt:
            logger.info("Trading bot stopped by user")
        except Exception as e:
            logger.error(f"Trading bot error: {e}")
        finally:
            # Final drawdown status report
            if self.trading_stopped_due_to_drawdown:
                current_equity = self.get_account_equity()
                total_loss = self.initial_equity - current_equity
                loss_percentage = (total_loss / self.initial_equity) * 100
                
                logger.critical("=== DRAWDOWN PROTECTION SUMMARY ===")
                logger.critical(f"Initial Equity: ${self.initial_equity:.2f}")
                logger.critical(f"Final Equity: ${current_equity:.2f}")
                logger.critical(f"Total Loss: ${total_loss:.2f} ({loss_percentage:.2f}%)")
                logger.critical(f"Protection triggered at {self.max_drawdown_percent}% drawdown")
            
            self.is_running = False
            logger.info("Trading bot stopped")
    
    def get_trading_statistics(self) -> Dict:
        """Get trading performance statistics"""
        try:
            # Get all deals (closed positions) for this magic number
            deals = mt5.history_deals_get(
                datetime.now() - timedelta(days=30),  # Last 30 days
                datetime.now()
            )
            
            if not deals:
                return {"message": "No trading history found"}
            
            # Filter deals by magic number
            robot_deals = [deal for deal in deals if deal.magic == self.magic_number]
            
            if not robot_deals:
                return {"message": "No deals found for this robot"}
            
            total_profit = sum(deal.profit for deal in robot_deals)
            total_trades = len(robot_deals)
            winning_trades = len([deal for deal in robot_deals if deal.profit > 0])
            win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
            
            return {
                "total_trades": total_trades,
                "winning_trades": winning_trades,
                "losing_trades": total_trades - winning_trades,
                "total_profit": total_profit,
                "win_rate": win_rate,
                "current_balance": self.get_account_balance()
            }
            
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {"error": str(e)}

    def get_trading_cycle(self, report_period_hours: Optional[int] = None) -> Dict:
        """Get trading statistics each cycle for reporting"""
        try:
            end_time = datetime.now()
            period_hours = report_period_hours or self.report_period_hours
            start_time = end_time - timedelta(hours=period_hours)
            deals = mt5.history_deals_get(start_time, end_time)

            if not deals:
                return {"message": "No trading history found"}

            position_groups = {}
            for deal in deals:
                position_id = getattr(deal, "position_id", None)
                if not position_id:
                    continue
                position_groups.setdefault(position_id, []).append(deal)

            robot_positions = []
            for position_id, position_deals in position_groups.items():
                if any(getattr(deal, "magic", None) == self.magic_number for deal in position_deals):
                    robot_positions.append((position_id, position_deals))

            if not robot_positions:
                return {"message": "No deals found for this robot"}

            close_entries = (mt5.DEAL_ENTRY_OUT, getattr(mt5, "DEAL_ENTRY_OUT_BY", -1))
            closed_positions = []

            for position_id, position_deals in robot_positions:
                if not any(deal.entry in close_entries for deal in position_deals):
                    continue

                net_profit = sum(
                    (getattr(deal, "profit", 0.0) or 0.0) +
                    (getattr(deal, "commission", 0.0) or 0.0) +
                    (getattr(deal, "swap", 0.0) or 0.0) +
                    (getattr(deal, "fee", 0.0) or 0.0)
                    for deal in position_deals
                )
                close_times = [
                    datetime.fromtimestamp(deal.time)
                    for deal in position_deals
                    if deal.entry in close_entries
                ]

                closed_positions.append({
                    "position_id": position_id,
                    "symbol": next((deal.symbol for deal in position_deals if getattr(deal, "symbol", None)), "UNKNOWN"),
                    "net_profit": net_profit,
                    "closed_at": max(close_times) if close_times else None,
                })

            if not closed_positions:
                return {"message": f"No closed trades found for this robot in the last {period_hours} hours"}

            total_profit = sum(position["net_profit"] for position in closed_positions)
            total_trades = len(closed_positions)
            winning_trades = sum(1 for position in closed_positions if position["net_profit"] > 0)
            losing_trades = sum(1 for position in closed_positions if position["net_profit"] < 0)
            breakeven_trades = total_trades - winning_trades - losing_trades
            win_rate = (winning_trades / total_trades) * 100 if total_trades else 0.0
            latest_close = max(
                (position["closed_at"] for position in closed_positions if position["closed_at"] is not None),
                default=None
            )

            return {
                "Period": f"{start_time.strftime('%Y-%m-%d %H:%M:%S')} -> {end_time.strftime('%Y-%m-%d %H:%M:%S')}",
                "Report Period Hours": period_hours,
                "Total Trades": total_trades,
                "Winning Trades": winning_trades,
                "Losing Trades": losing_trades,
                "Breakeven Trades": breakeven_trades,
                "Win Rate": f"{win_rate:.2f}%",
                "Net Profit": round(total_profit, 2),
                "Open Positions": len(self.get_current_positions()),
                "Initial Balance": round(self.initial_equity, 2),
                "Current Balance": round(self.get_account_balance(), 2),
                "Last Closed Trade": latest_close.strftime('%Y-%m-%d %H:%M:%S') if latest_close else "N/A"
            }

        except Exception as e:
            logger.error(f"Error getting cycle statistics: {e}")
            return {"error": str(e)}

    def send_telegram_report(self, stats: dict):
        bot_token = self.telegram_config.bot_token
        chat_id = self.telegram_config.chat_id

        if not bot_token or not chat_id:
            logger.warning("Telegram report skipped because TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID is not configured")
            return

        # Format the stats dictionary into a readable string
        stats_text = "\n".join([f"{key}: {value}" for key, value in stats.items()])

        # Final message
        message = f"Trading Robot Report\n\n{stats_text}\n\nTime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        if message is None:
            payload = {"chat_id": chat_id, "text": "No trade history"}
        else:
            payload = {"chat_id": chat_id, "text": message}

        max_retries = 3
        timeout_seconds = 12
        dns_markers = ("NameResolutionError", "getaddrinfo failed", "Temporary failure in name resolution")

        for attempt in range(1, max_retries + 1):
            try:
                response = requests.post(url, data=payload, timeout=timeout_seconds, verify=False)
                if response.status_code == 200:
                    logger.info("Telegram report sent successfully")
                    return

                logger.error(
                    f"Telegram send failed (attempt {attempt}/{max_retries}): "
                    f"status={response.status_code}, body={response.text}"
                )
            except requests.exceptions.RequestException as e:
                err_msg = str(e)
                if any(marker in err_msg for marker in dns_markers):
                    logger.error(
                        "Telegram send failed due to DNS/network name resolution. "
                        "Please check internet/DNS settings, then retry."
                    )
                    return
                logger.error(f"Telegram send exception (attempt {attempt}/{max_retries}): {e}")

            if attempt < max_retries:
                time.sleep(2 * attempt)

        logger.error("Telegram report failed after max retries")

    def shutdown(self):
        """Shutdown the robot and MT5 connection"""
        self.is_running = False
        mt5.shutdown()
        logger.info("MT5 connection closed")

def create_robot(config: BotRuntimeConfig) -> MT5ForexRobot:
    return MT5ForexRobot(
        risk_per_trade=config.risk_per_trade,
        magic_number=config.magic_number,
        max_drawdown_percent=config.max_drawdown_percent,
        drawdown_period_hours=config.drawdown_period_hours,
        entry_timeframe=config.entry_timeframe,
        trend_timeframe=config.trend_timeframe,
        enable_jpy_tuning=config.enable_jpy_tuning,
        report_period_hours=config.report_period_hours,
        trading_mode=config.trading_mode,
        sma_short=config.sma_short,
        sma_long=config.sma_long,
        sma_trend=config.sma_trend,
        rsi_period=config.rsi_period,
        rsi_readybought=config.rsi_readybought,
        rsi_overbought=config.rsi_overbought,
        rsi_neutral=config.rsi_neutral,
        rsi_readysold=config.rsi_readysold,
        rsi_oversold=config.rsi_oversold,
        atr_period=config.atr_period,
        htf_fast_ma=config.htf_fast_ma,
        htf_slow_ma=config.htf_slow_ma,
        min_rr_ratio=config.min_rr_ratio,
        sl_atr_multiplier=config.sl_atr_multiplier,
        tp_rr_multiplier=config.tp_rr_multiplier,
        max_spread_atr_ratio=config.max_spread_atr_ratio,
        trend_strength_min=config.trend_strength_min,
        rsi_buy_min=config.rsi_buy_min,
        rsi_buy_max=config.rsi_buy_max,
        rsi_sell_min=config.rsi_sell_min,
        rsi_sell_max=config.rsi_sell_max,
        telegram_config=load_telegram_config(),
    )


def print_report(stats: Dict[str, Any]) -> None:
    print("\n=== TRADING ROBOT STATISTICS ===")
    for key, value in stats.items():
        print(f"{key}: {value}")


def main() -> None:
    config = load_runtime_config()
    configure_logging(config.log_level, config.log_file)
    logger.info(
        "Starting trading bot with config: symbols=%s run_duration_minutes=%s report_period_hours=%s trading_mode=%s entry_tf=%s trend_tf=%s",
        config.symbols,
        config.run_duration_minutes,
        config.report_period_hours,
        config.trading_mode,
        _timeframe_label(config.entry_timeframe),
        _timeframe_label(config.trend_timeframe),
    )

    robot = create_robot(config)

    try:
        robot.run_trading_bot(config.symbols, run_duration_minutes=config.run_duration_minutes)
        stats = robot.get_trading_cycle(report_period_hours=config.report_period_hours)
        print_report(stats)
        robot.send_telegram_report(stats)
    except Exception as e:
        logger.error(f"Main execution error: {e}")
    finally:
        robot.shutdown()


if __name__ == "__main__":
    main()

