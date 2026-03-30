import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional
import logging
import requests
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MT5ForexRobot:
    def __init__(self, risk_per_trade: float = 0.01, magic_number: int = 12345,
                 max_drawdown_percent: float = 25.0, drawdown_period_hours: int = 24,
                 timeframe: int = mt5.TIMEFRAME_M5, trend_timeframe: int = mt5.TIMEFRAME_M30):
        """
        Initialize the MT5 Forex Trading Robot
        
        Args:
            risk_per_trade: Risk percentage per trade (e.g., 0.02 = 2%)
            magic_number: Unique identifier for this robot's trades
            max_drawdown_percent: Maximum allowed equity drawdown percentage
            drawdown_period_hours: Period in hours to monitor drawdown
            timeframe: Current chart timeframe for entries (M5, M15, M30)
            trend_timeframe: Higher timeframe for trend filtering (M30, H1)
        """
        self.risk_per_trade = risk_per_trade
        self.magic_number = magic_number
        self.is_running = False
        self.timeframe = timeframe
        self.trend_timeframe = trend_timeframe
        
        # Trading parameters
        self.sma_short = 7   # Slightly slower to reduce noise
        self.sma_long = 25   # Better for M5-M30 cycles
        self.sma_trend = 200 # Standard for Trend Filter
        self.rsi_period = 14 # Standard period for more stability
        self.rsi_readybought = 65 # More conservative for buy entries
        self.rsi_overbought = 80
        self.rsi_neutral = 50
        self.rsi_readysold = 35 # More conservative for sell entries
        self.rsi_oversold = 20
        self.atr_period = 14
        
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
    
    def get_market_data(self, symbol: str, timeframe=mt5.TIMEFRAME_M5, count: int = 1000):
        """Get historical market data from MT5"""
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
    
    def analyze_market(self, symbol: str) -> str:
        """
        Analyze market conditions and generate trading signal with HTF trend filtering.
        Optimized for M5-M30 scalping/day-trading.
        """
        try:
            # 1. Get Higher Timeframe (HTF) Trend data
            df_htf = self.get_market_data(symbol, timeframe=self.trend_timeframe, count=self.sma_trend + 1)
            if df_htf is None or len(df_htf) < self.sma_trend:
                return 'HOLD'
            
            df_htf['sma_trend'] = self.calculate_sma(df_htf['close'], self.sma_trend)
            htf_price = df_htf.iloc[-1]['close']
            htf_trend_sma = df_htf.iloc[-1]['sma_trend']
            
            # Determine overall market direction from HTF
            is_bullish_trend = htf_price > htf_trend_sma
            is_bearish_trend = htf_price < htf_trend_sma
            
            # 2. Get Lower Timeframe (LTF) Entry data
            df = self.get_market_data(symbol, timeframe=self.timeframe, count=self.sma_long + 10)
            if df is None or len(df) < self.sma_long:
                return 'HOLD'
            
            # Calculate LTF Indicators
            df['sma_short'] = self.calculate_sma(df['close'], self.sma_short)
            df['sma_long'] = self.calculate_sma(df['close'], self.sma_long)
            df['rsi'] = self.calculate_rsi(df['close'], self.rsi_period)
            
            latest = df.iloc[-1]
            prev = df.iloc[-2]
            current_price = latest['close']
            rsi = latest['rsi']
            
            # Signal Detection
            golden_cross = (prev['sma_short'] <= prev['sma_long']) and (latest['sma_short'] > latest['sma_long'])
            death_cross = (prev['sma_short'] >= prev['sma_long']) and (latest['sma_short'] < latest['sma_long'])
            
            # Log current state
            trend_str = "BULLISH" if is_bullish_trend else "BEARISH"
            logger.info(f"Market Scan: {symbol} | HTF Trend ({self.trend_timeframe}): {trend_str} | LTF RSI ({self.timeframe}): {rsi:.2f}")

            # 3. Decision Logic: LTF entry MUST align with HTF trend
            
            # BUY: HTF is bullish + LTF Golden Cross + RSI not overbought
            if is_bullish_trend and golden_cross:
                if rsi < self.rsi_readybought:
                    logger.info(f"✅ BUY SIGNAL: Confirmed on HTF. RSI: {rsi:.2f}")
                    return 'BUY'
            
            # SELL: HTF is bearish + LTF Death Cross + RSI not oversold
            elif is_bearish_trend and death_cross:
                if rsi > self.rsi_readysold:
                    logger.info(f"✅ SELL SIGNAL: Confirmed on HTF. RSI: {rsi:.2f}")
                    return 'SELL'
            
            return 'HOLD'

        except Exception as e:
            logger.error(f"Error in analyze_market: {e}")
            return 'HOLD'
            #         return 'BUY'
            #     else :
            #         return 'SELL'
            # # follow trend find rsi oversold and overbought
            # elif sma_short > sma_long and current_price <= sma_short and current_price >= sma_long :
            #     if rsi <= 90 and rsi > self.rsi_overbought :
            #         return 'SELL'
            #     elif rsi > 90 :
            #         return 'SELL'
            #     else :
            #         return 'BUY'
            # elif sma_short < sma_long and current_price >= sma_short and current_price <= sma_long :
            #     if rsi >= 10 and rsi < self.rsi_oversold :
            #         return 'BUY'
            #     elif rsi < 10 :
            #         return 'BUY'
            #     else :
            #         return 'SELL'
            # else:
            #     return 'HOLD => miss all logic'
                
        except Exception as e:
            logger.error(f"Error analyzing market for {symbol}: {e}")
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
                close_info = self.get_position_close_info(ticket, tracked["symbol"])
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

    def get_position_close_info(self, position_ticket: int, symbol: str) -> Dict:
        """Get the latest close deal info for a specific position ticket."""
        try:
            start_time = self.last_history_check - timedelta(minutes=5)
            end_time = datetime.now() + timedelta(minutes=1)
            deals = mt5.history_deals_get(start_time, end_time)

            if not deals:
                return {}

            closing_deals = []
            for deal in deals:
                if getattr(deal, "position_id", None) != position_ticket:
                    continue
                if deal.entry != mt5.DEAL_ENTRY_OUT:
                    continue
                if symbol and deal.symbol != symbol:
                    continue
                closing_deals.append(deal)

            if not closing_deals:
                return {}

            last_close = max(closing_deals, key=lambda deal: deal.time)
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

            return {
                "profit": last_close.profit,
                "time": datetime.fromtimestamp(last_close.time),
                "reason": deal_reason_map.get(
                    getattr(last_close, "reason", None),
                    f"reason_{getattr(last_close, 'reason', 'unknown')}"
                ),
            }

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
                "type_filling": mt5.ORDER_FILLING_FOK,   # <-- use detected filling mode
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
    
    def execute_trade(self, symbol: str, signal: str) -> bool:
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
            
            # Get current market data for ATR calculation
            df = self.get_market_data(symbol)
            if df is None or len(df) < self.atr_period:
                return False
            
            # Calculate ATR for stop loss
            df['atr'] = self.calculate_atr(df['high'], df['low'], df['close'], self.atr_period)
            current_atr = df['atr'].iloc[-1]
            
            # Get current price
            tick = mt5.symbol_info_tick(symbol)
            if not tick:
                return False
            
            if signal == 'BUY':
                entry_price = tick.ask
                stop_loss = entry_price - (1 * current_atr)
                take_profit = entry_price + (1.5 * current_atr)
                
                volume = self.calculate_position_size(symbol, entry_price, stop_loss)
                
                if volume > 0:
                    success = self.place_order(
                        symbol=symbol,
                        order_type=mt5.ORDER_TYPE_BUY,
                        volume=volume,
                        price=entry_price,
                        stop_loss=stop_loss,
                        take_profit=take_profit
                    )
                    
                    if success:
                        logger.info(f"BUY order placed for {symbol}: Volume={volume}, Price={entry_price:.5f}, SL={stop_loss:.5f}, TP={take_profit:.5f}")
                    return success
            
            elif signal == 'SELL':
                entry_price = tick.bid
                stop_loss = entry_price + (1 * current_atr)
                take_profit = entry_price - (1.5 * current_atr)
                
                volume = self.calculate_position_size(symbol, entry_price, stop_loss)
                
                if volume > 0:
                    success = self.place_order(
                        symbol=symbol,
                        order_type=mt5.ORDER_TYPE_SELL,
                        volume=volume,
                        price=entry_price,
                        stop_loss=stop_loss,
                        take_profit=take_profit
                    )
                    
                    if success:
                        logger.info(f"SELL order placed for {symbol}: Volume={volume}, Price={entry_price:.5f}, SL={stop_loss:.5f}, TP={take_profit:.5f}")
                    return success
            
            return False
            
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
                            signal = self.analyze_market(symbol)
                            logger.info(f"{symbol}: Signal = {signal}")
                            
                            if signal in ['BUY', 'SELL']:
                                self.execute_trade(symbol, signal)
                                
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

    def get_trading_cycle(self) -> Dict:
        """Get trading statistics each cycle for reporting"""
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=4)
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
                return {"message": "No closed trades found for this robot in the last 4 hours"}

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
        bot_token = "8283118826:AAFOazKqekUIpXZR2sg09Sk1O6X15RM_wCY"  # bot token
        chat_id = "911796580"  # chat id

        # Format the stats dictionary into a readable string
        stats_text = "\n".join([f"{key}: {value}" for key, value in stats.items()])

        # Final message
        message = f"📊 Trading Robot Report\n\n{stats_text}\n\n⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        if message is None :
            payload = {"chat_id": chat_id, "text": "No trade history"}
        else :
            payload = {"chat_id": chat_id, "text": message}

        try:
            # r = requests.post(url, data=payload, verify=certifi.where())
            r = requests.post(url, data=payload, verify=False)
            if r.status_code == 200:
                print("✅ Report sent to Telegram!")
            else:
                print("❌ Failed report to Telegram:", r.text)
        except Exception as e:
            print("❌ Error on sent report to Telegram", e)
    
    def shutdown(self):
        """Shutdown the robot and MT5 connection"""
        self.is_running = False
        mt5.shutdown()
        logger.info("MT5 connection closed")

# Example usage
if __name__ == "__main__":
    # Create the robot with drawdown protection
    robot = MT5ForexRobot(
        risk_per_trade=0.001,           # 2% risk per trade
        magic_number=19910,            # Unique robot ID
        max_drawdown_percent=25.0,     # Stop trading if 25% drawdown
        drawdown_period_hours=4       # Monitor drawdown over 24 hours
    )
    
    # Define symbols to trade (make sure these are available in your MT5)
    # symbols = ['USDCHF']
    symbols = ['XAUUSD']
    
    try:
        # Run the trading bot for 30 minutes (adjust as needed)
        robot.run_trading_bot(symbols, run_duration_minutes=360)
        
        # Get and display statistics
        stats = robot.get_trading_cycle()
        print("\n=== TRADING ROBOT STATISTICS ===")
        for key, value in stats.items():
            print(f"{key}: {value}")
        
        robot.send_telegram_report(stats)
            
    except Exception as e:
        logger.error(f"Main execution error: {e}")
    finally:
        # Always shutdown properly
        robot.shutdown() 
