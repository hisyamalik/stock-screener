import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MT5ForexRobot:
    def __init__(
        self,
        risk_per_trade: float = 0.02,
        magic_number: int = 11123,
        max_drawdown_percent: float = 25.0,
        drawdown_period_hours: int = 1,
        # === Scalping options ===
        enable_scalping: bool = False,
        scalping_timeframe: int = mt5.TIMEFRAME_M5,
        max_spread_points: int = 20,              # e.g., 30 = 3.0 pips on 5-digit broker
        cooldown_seconds: int = 120,              # minimal time between trades per symbol
        active_hours: Optional[tuple] = (6, 22),  # trading window in server time (HH, HH)
        cycle_seconds: int = 10,                  # loop delay when scalping is enabled
        rr_tp_mult: float = 1.5,                  # TP multiple of ATR (scalping)
        rr_sl_mult: float = 1.0,                  # SL multiple of ATR (scalping)
        trailing_atr_mult: Optional[float] = 1.0, # set None to disable trailing stop
        # === Order Block options ===
        ob_confirmation: bool = True,             # use OB as additional confirmation (optional)
        ob_required: bool = False,                # if True, require OB alignment to take trades
        ob_lookback: int = 120,
        ob_atr_mult: float = 1.5,
        ob_zone_tolerance_pts: int = 0            # optional extra tolerance (points) for zone check
    ):
        """
        Initialize the MT5 Forex Trading Robot

        Args:
            risk_per_trade: Risk percentage per trade (e.g., 0.02 = 2%)
            magic_number: Unique identifier for this robot's trades
            max_drawdown_percent: Maximum allowed equity drawdown percentage
            drawdown_period_hours: Period in hours to monitor drawdown
            enable_scalping: Toggle scalping logic for <15M timeframes (M1/M5)
            scalping_timeframe: mt5 timeframe for scalping (mt5.TIMEFRAME_M1 or mt5.TIMEFRAME_M5)
            max_spread_points: Skip trades if spread is above this number (points)
            cooldown_seconds: Minimum delay between consecutive trades per symbol
            active_hours: Tuple of (start_hour, end_hour) in broker/server time to allow trading
            cycle_seconds: Main loop delay when scalping enabled (otherwise defaults to 60s)
            rr_tp_mult / rr_sl_mult: Take-profit and stop-loss multipliers of ATR for scalping
            trailing_atr_mult: If set, apply ATR-based trailing stop during monitoring
            ob_confirmation: If True, use OB as additional confirmation (but not required unless ob_required=True)
            ob_required: If True, trades only taken when OB aligns with signal
            ob_lookback: Number of candles to scan for OBs
            ob_atr_mult: Impulse threshold multiple (vs ATR) to tag an OB
            ob_zone_tolerance_pts: Extra points added to OB zone bounds for proximity tolerance
        """
        self.risk_per_trade = risk_per_trade
        self.magic_number = magic_number
        self.is_running = False

        # Trading parameters (swing/base system)
        self.sma_short = 20
        self.sma_long = 50
        self.rsi_period = 14
        self.rsi_overbought = 70
        self.rsi_oversold = 30
        self.atr_period = 14

        # Drawdown protection parameters
        self.max_drawdown_percent = max_drawdown_percent
        self.drawdown_period_hours = drawdown_period_hours
        self.equity_history = []  # Store equity snapshots with timestamps
        self.initial_equity = 0.0
        self.peak_equity = 0.0
        self.trading_stopped_due_to_drawdown = False

        # Scalping
        self.enable_scalping = enable_scalping
        self.scalping_timeframe = scalping_timeframe
        self.max_spread_points = max_spread_points
        self.cooldown_seconds = cooldown_seconds
        self.active_hours = active_hours
        self.cycle_seconds = cycle_seconds
        self.rr_tp_mult = rr_tp_mult
        self.rr_sl_mult = rr_sl_mult
        self.trailing_atr_mult = trailing_atr_mult
        self._last_trade_time: Dict[str, datetime] = {}

        # OB settings
        self.ob_confirmation = ob_confirmation
        self.ob_required = ob_required
        self.ob_lookback = ob_lookback
        self.ob_atr_mult = ob_atr_mult
        self.ob_zone_tolerance_pts = ob_zone_tolerance_pts

        # Initialize MT5 connection
        self.initialize_mt5()

    def initialize_mt5(self) -> bool:
        """Initialize MT5 connection"""
        try:
            if not mt5.initialize():
                logger.error(f"initialize() failed, error code = {mt5.last_error()}" )
                return False

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
            self.record_equity_snapshot()

            logger.info(f"Drawdown protection: Max {self.max_drawdown_percent}% over {self.drawdown_period_hours} hours")

            return True

        except Exception as e:
            logger.error(f"MT5 initialization failed: {e}")
            return False

    def get_account_balance(self) -> float:
        account_info = mt5.account_info()
        return account_info.balance if account_info else 0.0

    def get_account_equity(self) -> float:
        account_info = mt5.account_info()
        return account_info.equity if account_info else 0.0

    def record_equity_snapshot(self):
        current_equity = self.get_account_equity()
        current_time = datetime.now()

        self.equity_history.append({'timestamp': current_time, 'equity': current_equity})

        if current_equity > self.peak_equity:
            self.peak_equity = current_equity

        cutoff_time = current_time - timedelta(hours=self.drawdown_period_hours)
        self.equity_history = [r for r in self.equity_history if r['timestamp'] > cutoff_time]

    def check_drawdown_protection(self) -> bool:
        if not self.equity_history:
            return False

        current_equity = self.get_account_equity()
        period_peak = max(record['equity'] for record in self.equity_history)
        drawdown_from_peak = ((period_peak - current_equity) / period_peak) * 100 if period_peak > 0 else 0.0
        drawdown_from_initial = ((self.initial_equity - current_equity) / self.initial_equity) * 100 if self.initial_equity > 0 else 0.0

        if len(self.equity_history) > 1:
            period_start_equity = self.equity_history[0]['equity']
            drawdown_from_period_start = ((period_start_equity - current_equity) / period_start_equity) * 100 if period_start_equity > 0 else 0.0
        else:
            drawdown_from_period_start = 0.0

        max_drawdown = max(drawdown_from_peak, drawdown_from_initial, drawdown_from_period_start)

        logger.info(
            f"Equity Monitor - Current: ${current_equity:.2f}, Peak: ${period_peak:.2f}, "
            f"Drawdown: {max_drawdown:.2f}% (Max allowed: {self.max_drawdown_percent}%)"
        )

        if max_drawdown >= self.max_drawdown_percent:
            logger.critical("⚠️  DRAWDOWN PROTECTION TRIGGERED! ⚠️")
            self.trading_stopped_due_to_drawdown = True
            return True

        return False

    def close_all_positions(self) -> bool:
        try:
            positions = self.get_current_positions()
            if not positions:
                logger.info("No positions to close")
                return True

            logger.info(f"Emergency closing {len(positions)} positions...")

            for position in positions:
                try:
                    if position.type == mt5.POSITION_TYPE_BUY:
                        order_type = mt5.ORDER_TYPE_SELL
                        price = mt5.symbol_info_tick(position.symbol).bid
                    else:
                        order_type = mt5.ORDER_TYPE_BUY
                        price = mt5.symbol_info_tick(position.symbol).ask

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

    def get_market_data(self, symbol: str, timeframe=mt5.TIMEFRAME_M15, count: int = 300):
        try:
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
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
        return prices.rolling(window=period).mean()

    def calculate_ema(self, prices: pd.Series, period: int) -> pd.Series:
        return prices.ewm(span=period, adjust=False).mean()

    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr

    def calculate_bbands(self, close: pd.Series, period: int = 20, std_mult: float = 2.0):
        ma = close.rolling(window=period).mean()
        std = close.rolling(window=period).std()
        upper = ma + std_mult * std
        lower = ma - std_mult * std
        return upper, ma, lower

    # ===== Order Block detection =====
    def detect_orderblocks(self, df, lookback=None, atr_mult=None):
        """Return a list of orderblocks with fields: type, zone_low, zone_high."""
        if lookback is None:
            lookback = self.ob_lookback
        if atr_mult is None:
            atr_mult = self.ob_atr_mult

        orderblocks = []
        df = df.copy()
        df['atr'] = self.calculate_atr(df['high'], df['low'], df['close'], period=14)

        start = max(1, len(df) - lookback - 1)
        for i in range(start, len(df) - 1):
            candle = df.iloc[i]
            next_candle = df.iloc[i + 1]
            atr = candle['atr']
            if atr is None or np.isnan(atr):
                continue
            # Bullish impulse
            if (next_candle['close'] - next_candle['open']) > atr_mult * atr and candle['close'] < candle['open']:
                orderblocks.append({
                    "type": "bullish",
                    "zone_low": float(candle['low']),
                    "zone_high": float(candle['high'])
                })
            # Bearish impulse
            if (next_candle['open'] - next_candle['close']) > atr_mult * atr and candle['close'] > candle['open']:
                orderblocks.append({
                    "type": "bearish",
                    "zone_low": float(candle['low']),
                    "zone_high": float(candle['high'])
                })
        return orderblocks

    def _ob_alignment(self, df, signal: str, symbol: str) -> bool:
        """Check whether price is within a matching OB zone (with tolerance)."""
        if signal not in ('BUY','SELL'):
            return False
        obs = self.detect_orderblocks(df)
        if not obs:
            return False
        last_price = float(df['close'].iloc[-1])
        # add tolerance in points
        tol = 0.0
        if self.ob_zone_tolerance_pts and mt5.symbol_info(symbol):
            tol = self.ob_zone_tolerance_pts * mt5.symbol_info(symbol).point

        for ob in reversed(obs[-6:]):  # check a handful of latest OBs
            low = ob['zone_low'] - tol
            high = ob['zone_high'] + tol
            if ob['type'] == 'bullish' and signal == 'BUY':
                if low <= last_price <= high:
                    return True
            if ob['type'] == 'bearish' and signal == 'SELL':
                if low <= last_price <= high:
                    return True
        return False

    # ===== Base swing analysis (M15) =====
    def analyze_market(self, symbol: str) -> str:
        try:
            df = self.get_market_data(symbol, timeframe=mt5.TIMEFRAME_M15, count=300)
            if df is None or len(df) < self.sma_long:
                return 'HOLD'

            df['sma_short'] = self.calculate_sma(df['close'], self.sma_short)
            df['sma_long'] = self.calculate_sma(df['close'], self.sma_long)
            df['rsi'] = self.calculate_rsi(df['close'], self.rsi_period)

            latest = df.iloc[-1]
            current_price = latest['close']
            sma_short = latest['sma_short']
            sma_long = latest['sma_long']
            rsi = latest['rsi']

            if pd.isna(sma_short) or pd.isna(sma_long) or pd.isna(rsi):
                return 'HOLD'

            signal = 'HOLD'
            if sma_short > sma_long and rsi < self.rsi_overbought and current_price > sma_short:
                signal = 'BUY'
            elif sma_short < sma_long and rsi > self.rsi_oversold and current_price < sma_short:
                signal = 'SELL'

            # OB confirmation (optional)
            if self.ob_confirmation and signal in ('BUY','SELL'):
                aligned = self._ob_alignment(df, signal, symbol)
                if self.ob_required and not aligned:
                    return 'HOLD'  # require OB, skip if not aligned
                if aligned:
                    logger.info(f"{symbol} swing {signal} confirmed by OB")
                else:
                    logger.info(f"{symbol} swing {signal} without OB confirmation")

            return signal

        except Exception as e:
            logger.error(f"Error analyzing market for {symbol}: {e}")
            return 'HOLD'

    # ===== Scalping analysis (< 15M) =====
    def analyze_market_scalping(self, symbol: str) -> str:
        """
        Trend-follow scalping with pullback confirmation:
        - Use EMA(9/21) for trend
        - RSI(3) for momentum exhaustion
        - Bollinger Bands(20,2): re-entry after band touch
        """
        try:
            tf = self.scalping_timeframe
            df = self.get_market_data(symbol, timeframe=tf, count=400)
            if df is None or len(df) < 60:
                return 'HOLD'

            # Filters
            sinfo = mt5.symbol_info(symbol)
            if not sinfo or (sinfo.spread is None):
                return 'HOLD'
            if self.max_spread_points and sinfo.spread > self.max_spread_points:
                logger.info(f"{symbol} spread too high ({sinfo.spread} pts > {self.max_spread_points}) - skip")
                return 'HOLD'

            # Trading hours filter (server time)
            if self.active_hours is not None:
                sh, eh = self.active_hours
                hour_now = datetime.now().hour
                if not (sh <= hour_now < eh):
                    return 'HOLD'

            # Indicators
            close = df['close']
            high = df['high']
            low = df['low']
            df['ema9'] = self.calculate_ema(close, 9)
            df['ema21'] = self.calculate_ema(close, 21)
            df['rsi3'] = self.calculate_rsi(close, 3)
            upper, mid, lowerb = self.calculate_bbands(close, 20, 2.0)
            df['bb_upper'] = upper
            df['bb_mid'] = mid
            df['bb_lower'] = lowerb

            # Latest two candles for cross confirmation
            prev = df.iloc[-2]
            last = df.iloc[-1]

            raw_signal = 'HOLD'
            # Long setup
            long_trend = last['ema9'] > last['ema21']
            long_pullback = prev['close'] <= prev['bb_lower'] and last['close'] > last['bb_lower']
            long_rsi = last['rsi3'] < 25
            if long_trend and long_pullback and long_rsi:
                raw_signal = 'BUY'

            # Short setup
            short_trend = last['ema9'] < last['ema21']
            short_pullback = prev['close'] >= prev['bb_upper'] and last['close'] < last['bb_upper']
            short_rsi = last['rsi3'] > 75
            if short_trend and short_pullback and short_rsi:
                raw_signal = 'SELL'

            signal = raw_signal
            # OB confirmation (optional)
            if self.ob_confirmation and signal in ('BUY','SELL'):
                aligned = self._ob_alignment(df, signal, symbol)
                if self.ob_required and not aligned:
                    return 'HOLD'
                if aligned:
                    logger.info(f"{symbol} scalping {signal} confirmed by OB")
                else:
                    logger.info(f"{symbol} scalping {signal} without OB confirmation")

            return signal
        except Exception as e:
            logger.error(f"Error analyzing market (scalping) for {symbol}: {e}")
            return 'HOLD'

    def calculate_position_size(self, symbol: str, entry_price: float, stop_loss: float) -> float:
        try:
            account_info = mt5.account_info()
            if not account_info:
                return 0.0

            balance = account_info.balance
            risk_amount = balance * self.risk_per_trade

            symbol_info = mt5.symbol_info(symbol)
            if not symbol_info:
                return 0.0

            pip_value = symbol_info.trade_tick_value
            pips_risked = abs(entry_price - stop_loss) / symbol_info.point

            if pips_risked == 0:
                return 0.0

            lot_size = risk_amount / (pips_risked * pip_value)
            lot_size = max(symbol_info.volume_min, min(lot_size, symbol_info.volume_max))
            step = symbol_info.volume_step
            lot_size = round(lot_size / step) * step
            return lot_size

        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0.0

    def get_current_positions(self) -> List:
        positions = mt5.positions_get()
        return [pos for pos in positions if pos.magic == self.magic_number] if positions else []

    def _respect_cooldown(self, symbol: str) -> bool:
        """Return True if allowed to trade now (cooldown passed or no trade yet)."""
        last = self._last_trade_time.get(symbol)
        if last is None:
            return True
        return (datetime.now() - last).total_seconds() >= self.cooldown_seconds

    def _mark_trade_time(self, symbol: str):
        self._last_trade_time[symbol] = datetime.now()

    def place_order(self, symbol: str, order_type: int, volume: float, price: float, 
                   stop_loss: float = None, take_profit: float = None, comment: str = "Python Forex Robot") -> bool:
        try:
            symbol_info = mt5.symbol_info(symbol)
            if not symbol_info:
                logger.error(f"Symbol {symbol} not found")
                return False

            if not symbol_info.visible:
                logger.error(f"Symbol {symbol} is not visible")
                return False

            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": order_type,
                "price": price,
                "sl": stop_loss,
                "tp": take_profit,
                "magic": self.magic_number,
                "comment": comment,
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            result = mt5.order_send(request)

            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(f"Order failed: {result.retcode} - {result.comment}")
                return False

            logger.info(f"Order successful: {result.order} - {result.comment}")
            return True

        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return False

    def execute_trade(self, symbol: str, signal: str) -> bool:
        try:
            if self.trading_stopped_due_to_drawdown:
                logger.warning(f"Trading stopped due to drawdown protection - ignoring {signal} for {symbol}")
                return False

            if not self._respect_cooldown(symbol):
                logger.info(f"{symbol}: cooldown active - skip")
                return False

            existing_positions = [pos for pos in self.get_current_positions() if pos.symbol == symbol]
            if existing_positions:
                logger.info(f"Already have position for {symbol}")
                return False

            timeframe = self.scalping_timeframe if self.enable_scalping else mt5.TIMEFRAME_M15
            df = self.get_market_data(symbol, timeframe=timeframe, count=300)
            if df is None or len(df) < self.atr_period:
                return False

            df['atr'] = self.calculate_atr(df['high'], df['low'], df['close'], self.atr_period)
            current_atr = df['atr'].iloc[-1]

            tick = mt5.symbol_info_tick(symbol)
            if not tick or current_atr is None or np.isnan(current_atr):
                return False

            # OB alignment info for logging
            ob_aligned = False
            if self.ob_confirmation and signal in ('BUY','SELL'):
                ob_aligned = self._ob_alignment(df, signal, symbol)

            if signal == 'BUY':
                entry_price = tick.ask
                if self.enable_scalping:
                    stop_loss = entry_price - (self.rr_sl_mult * current_atr)
                    take_profit = entry_price + (self.rr_tp_mult * current_atr)
                else:
                    stop_loss = entry_price - (2 * current_atr)
                    take_profit = entry_price + (3 * current_atr)

                volume = self.calculate_position_size(symbol, entry_price, stop_loss)

                if volume > 0:
                    comment = f"Scalp BUY (OB)" if ob_aligned else "Scalp BUY" if self.enable_scalping else ("Swing BUY (OB)" if ob_aligned else "Swing BUY")
                    success = self.place_order(
                        symbol=symbol,
                        order_type=mt5.ORDER_TYPE_BUY,
                        volume=volume,
                        price=entry_price,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        comment=comment
                    )
                    if success:
                        logger.info(
                            f"BUY {symbol}: Vol={volume}, Price={entry_price:.5f}, SL={stop_loss:.5f}, TP={take_profit:.5f}, OB={ob_aligned}"
                        )
                        self._mark_trade_time(symbol)
                    return success

            elif signal == 'SELL':
                entry_price = tick.bid
                if self.enable_scalping:
                    stop_loss = entry_price + (self.rr_sl_mult * current_atr)
                    take_profit = entry_price - (self.rr_tp_mult * current_atr)
                else:
                    stop_loss = entry_price + (2 * current_atr)
                    take_profit = entry_price - (3 * current_atr)

                volume = self.calculate_position_size(symbol, entry_price, stop_loss)

                if volume > 0:
                    comment = f"Scalp SELL (OB)" if ob_aligned else "Scalp SELL" if self.enable_scalping else ("Swing SELL (OB)" if ob_aligned else "Swing SELL")
                    success = self.place_order(
                        symbol=symbol,
                        order_type=mt5.ORDER_TYPE_SELL,
                        volume=volume,
                        price=entry_price,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        comment=comment
                    )
                    if success:
                        logger.info(
                            f"SELL {symbol}: Vol={volume}, Price={entry_price:.5f}, SL={stop_loss:.5f}, TP={take_profit:.5f}, OB={ob_aligned}"
                        )
                        self._mark_trade_time(symbol)
                    return success

            return False

        except Exception as e:
            logger.error(f"Error executing trade for {symbol}: {e}")
            return False

    def monitor_positions(self):
        try:
            positions = self.get_current_positions()
            for position in positions:
                profit = position.profit
                logger.info(f"Position {position.symbol}: Type={position.type}, Vol={position.volume}, Profit=${profit:.2f}")

                # ATR-based trailing stop for scalping
                if self.enable_scalping and self.trailing_atr_mult is not None:
                    timeframe = self.scalping_timeframe
                    df = self.get_market_data(position.symbol, timeframe=timeframe, count=200)
                    if df is None or len(df) < self.atr_period:
                        continue
                    df['atr'] = self.calculate_atr(df['high'], df['low'], df['close'], self.atr_period)
                    atr = df['atr'].iloc[-1]
                    if atr is None or np.isnan(atr):
                        continue

                    tick = mt5.symbol_info_tick(position.symbol)
                    if not tick:
                        continue

                    # Compute new SL based on current price and ATR
                    if position.type == mt5.POSITION_TYPE_BUY:
                        new_sl = max(position.sl or 0.0, tick.bid - self.trailing_atr_mult * atr)
                        if position.sl is None or new_sl > position.sl:
                            self._modify_sl(position, new_sl)
                    elif position.type == mt5.POSITION_TYPE_SELL:
                        new_sl = min(position.sl or 1e12, tick.ask + self.trailing_atr_mult * atr)
                        if position.sl is None or new_sl < position.sl:
                            self._modify_sl(position, new_sl)
        except Exception as e:
            logger.error(f"Error monitoring positions: {e}")

    def _modify_sl(self, position, new_sl: float):
        try:
            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "position": position.ticket,
                "sl": new_sl,
                "tp": position.tp,
                "symbol": position.symbol,
                "magic": self.magic_number,
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            result = mt5.order_send(request)
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(f"Updated SL for {position.symbol} -> {new_sl:.5f}")
            else:
                logger.error(f"Failed to update SL for {position.symbol}: {result.comment}")
        except Exception as e:
            logger.error(f"Error updating SL: {e}")

    def check_and_handle_drawdown(self) -> bool:
        self.record_equity_snapshot()
        if self.check_drawdown_protection():
            self.close_all_positions()
            self.is_running = False
            logger.critical("🛑 TRADING BOT STOPPED DUE TO DRAWDOWN PROTECTION 🛑")
            return False
        return True

    def run_trading_bot(self, symbols: List[str], run_duration_minutes: int = 60):
        logger.info(f"Starting MT5 Forex Robot for {run_duration_minutes} minutes")
        logger.info(f"Trading symbols: {symbols}")

        self.is_running = True
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=run_duration_minutes)

        try:
            while self.is_running and datetime.now() < end_time:
                logger.info("--- Trading Cycle ---")

                if not self.check_and_handle_drawdown():
                    logger.critical("Trading stopped due to drawdown protection!")
                    break

                self.monitor_positions()

                if not self.trading_stopped_due_to_drawdown:
                    for symbol in symbols:
                        try:
                            if self.enable_scalping and self.scalping_timeframe in (mt5.TIMEFRAME_M1, mt5.TIMEFRAME_M2, mt5.TIMEFRAME_M3, mt5.TIMEFRAME_M4, mt5.TIMEFRAME_M5):
                                signal = self.analyze_market_scalping(symbol)
                            else:
                                signal = self.analyze_market(symbol)

                            logger.info(f"{symbol}: Signal = {signal}")
                            if signal in ['BUY', 'SELL']:
                                self.execute_trade(symbol, signal)
                        except Exception as e:
                            logger.error(f"Error processing {symbol}: {e}")
                else:
                    logger.warning("New trades disabled due to drawdown protection")

                time.sleep(self.cycle_seconds if self.enable_scalping else 60)

        except KeyboardInterrupt:
            logger.info("Trading bot stopped by user")
        except Exception as e:
            logger.error(f"Trading bot error: {e}")
        finally:
            if self.trading_stopped_due_to_drawdown:
                current_equity = self.get_account_equity()
                total_loss = self.initial_equity - current_equity
                loss_percentage = (total_loss / self.initial_equity) * 100 if self.initial_equity > 0 else 0.0

                logger.critical("=== DRAWDOWN PROTECTION SUMMARY ===")
                logger.critical(f"Initial Equity: ${self.initial_equity:.2f}")
                logger.critical(f"Final Equity: ${current_equity:.2f}")
                logger.critical(f"Total Loss: ${total_loss:.2f} ({loss_percentage:.2f}%)")
                logger.critical(f"Protection triggered at {self.max_drawdown_percent}% drawdown")

            self.is_running = False
            logger.info("Trading bot stopped")

    def get_trading_statistics(self) -> Dict:
        try:
            deals = mt5.history_deals_get(
                datetime.now() - timedelta(days=30),
                datetime.now()
            )
            if not deals:
                return {"message": "No trading history found"}

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

    def shutdown(self):
        self.is_running = False
        mt5.shutdown()
        logger.info("MT5 connection closed")

# Example usage
if __name__ == "__main__":
    # Create the robot with scalping enabled on M5 and OB confirmation optional
    robot = MT5ForexRobot(
        risk_per_trade=0.02,
        magic_number=11123,
        max_drawdown_percent=25.0,
        drawdown_period_hours=1,
        enable_scalping=True,
        # scalping_timeframe=mt5.TIMEFRAME_M15,
        scalping_timeframe=mt5.TIMEFRAME_M1,
        max_spread_points=20,
        cooldown_seconds=120,
        active_hours=(6, 22),
        cycle_seconds=10,
        rr_tp_mult=2.0,
        rr_sl_mult=1.0,
        trailing_atr_mult=1.0,
        ob_confirmation=True,
        ob_required=False,
        ob_lookback=120,
        ob_atr_mult=1.5,
        ob_zone_tolerance_pts=0
    )

    symbols = ['EURUSD','XAUUSD']

    try:
        robot.run_trading_bot(symbols, run_duration_minutes=120)

        stats = robot.get_trading_statistics()
        print("\n=== TRADING ROBOT STATISTICS ===")
        for key, value in stats.items():
            print(f"{key}: {value}")

    except Exception as e:
        logger.error(f"Main execution error: {e}")
    finally:
        robot.shutdown()
