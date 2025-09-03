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
                 max_drawdown_percent: float = 25.0, drawdown_period_hours: int = 24):
        """
        Initialize the MT5 Forex Trading Robot
        
        Args:
            risk_per_trade: Risk percentage per trade (e.g., 0.02 = 2%)
            magic_number: Unique identifier for this robot's trades
            max_drawdown_percent: Maximum allowed equity drawdown percentage (e.g., 25.0 = 25%)
            drawdown_period_hours: Period in hours to monitor drawdown (e.g., 24 = 24 hours)
        """
        self.risk_per_trade = risk_per_trade
        self.magic_number = magic_number
        self.is_running = False
        
        
        # Trading parameters
        self.sma_short = 10
        self.sma_long = 20
        self.rsi_period = 9
        self.rsi_readybought = 70
        self.rsi_overbought = 80
        self.rsi_neutral = 50
        self.rsi_readysold = 30
        self.rsi_oversold = 20
        self.atr_period = 14
        
        # Drawdown protection parameters
        self.max_drawdown_percent = max_drawdown_percent
        self.drawdown_period_hours = drawdown_period_hours
        self.equity_history = []  # Store equity snapshots with timestamps
        self.initial_equity = 0.0
        self.peak_equity = 0.0
        self.trading_stopped_due_to_drawdown = False
        
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
    
    def get_market_data(self, symbol: str, timeframe=mt5.TIMEFRAME_M1, count: int = 1000):
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
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
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
    
    def analyze_market(self, symbol: str) -> str:
        """
        Analyze market conditions and generate trading signal
        
        Returns:
            'BUY', 'SELL', or 'HOLD'
        """
        try:
            # Get market data
            df = self.get_market_data(symbol)
            if df is None or len(df) < self.sma_long:
                return 'HOLD'
            
            # Calculate indicators
            df['sma_short'] = self.calculate_sma(df['close'], self.sma_short)
            df['sma_long'] = self.calculate_sma(df['close'], self.sma_long)
            df['rsi'] = self.calculate_rsi(df['close'], self.rsi_period)
            
            # Get latest values
            latest = df.iloc[-1]
            current_price = latest['close']
            current_price_open = latest['open']
            sma_short = latest['sma_short']
            sma_long = latest['sma_long']
            rsi = latest['rsi']
            divergence_flag = self.detect_rsi_divergence(df["close"], df["rsi"], order=25)
            logger.info(f"RSI Divergence Status {divergence_flag} !")

            # Check if we have valid indicator values
            if pd.isna(sma_short) or pd.isna(sma_long) or pd.isna(rsi):
                return 'HOLD'
                
            # follow trend under/above moving average
            if sma_short > sma_long and current_price_open <= sma_short and current_price_open <= sma_long: #uptrend
                if self.rsi_neutral < rsi <= self.rsi_readysold :
                    return 'BUY'
                else:
                    return 'HOLD => follow trend BUY'
            elif sma_short < sma_long and current_price_open >= sma_short and current_price_open >= sma_long : #downtrend
                if self.rsi_neutral > rsi >= self.rsi_readybought :
                    return 'SELL'
                else:
                    return 'HOLD => follow trend SELL'
            # find rsi divergence
            if sma_short > sma_long and current_price_open > sma_long and current_price_open <= sma_short:
                if rsi >= self.rsi_overbought and divergence_flag == 'bearish':
                    return 'SELL'
                elif rsi < self.rsi_neutral and rsi >= self.rsi_readybought and divergence_flag == 'bearish':
                    return 'SELL'
                else :
                    return 'BUY'
            elif sma_short < sma_long and current_price_open < sma_long and current_price_open >= sma_short:
                if rsi <= self.rsi_oversold and divergence_flag == 'bullish':
                    return 'BUY'
                elif rsi > self.rsi_neutral and rsi >= self.rsi_readysold and divergence_flag == 'bullish':
                    return 'BUY'
                else :
                    return 'SELL'
            # follow trend find rsi oversold and overbought
            elif sma_short > sma_long and current_price <= sma_short and current_price >= sma_long :
                if rsi <= 90 and rsi > self.rsi_overbought :
                    return 'SELL'
                elif rsi > 90 :
                    return 'SELL'
                else :
                    return 'BUY'
            elif sma_short < sma_long and current_price >= sma_short and current_price <= sma_long :
                if rsi >= 10 and rsi < self.rsi_oversold :
                    return 'BUY'
                elif rsi < 10 :
                    return 'BUY'
                else :
                    return 'SELL'
            else:
                return 'HOLD => miss all logic'
                
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
                stop_loss = entry_price - (1.3 * current_atr)
                take_profit = entry_price + (1.8 * current_atr)
                
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
                stop_loss = entry_price + (1.3 * current_atr)
                take_profit = entry_price - (1.8 * current_atr)
                
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
            # Get all deals (closed positions) for this magic number
            end_time = datetime.fromtimestamp(self.server_time)
            start_time = end_time - timedelta(hours=4)

            deals = mt5.history_deals_get(start_time, end_time)
            print(f"Found {len(deals)} deals in last timestamp")
            for d in deals:
                print(d.ticket, d.symbol, d.magic, d.type, d.entry, d.volume, d.profit, d.commission, d.swap)

            if not deals:
                return {"message": "No trading history found"}

            # All deals for this robot (profit will include fees, commission, swaps, etc.)
            robot_deals = [d for d in deals if d.magic == self.magic_number]
            if not robot_deals:
                return {"message": "No deals found for this robot"}

            # Total profit = all deals (commissions/swaps included)
            total_profit = sum(d.profit for d in robot_deals)

            # Win/Loss = only from closing trades
            closing_deals = [d for d in robot_deals if d.entry == mt5.DEAL_ENTRY_OUT]

            total_trades = len(closing_deals)
            winning_trades = sum(1 for d in closing_deals if d.profit > 0)
            losing_trades = total_trades - winning_trades
            win_rate = (winning_trades / total_trades) * 100 if total_trades else 0

            return {
                "🚀 Total Trades": total_trades,
                "👍 Winning Trades": winning_trades,
                "👎 Losing Trades": losing_trades,
                "😎 Total Profit (incl. fees)": round(total_profit, 2),
                "❤️ Win Rate": f"{win_rate:.2f}%",
                "😀 Initial Balance": self.initial_equity,
                "😂 Current Balance": self.get_account_balance()
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
    symbols = ['XAUUSD']
    # symbols = ['USDJPY']
    
    try:
        # Run the trading bot for 30 minutes (adjust as needed)
        robot.run_trading_bot(symbols, run_duration_minutes=240)
        
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
