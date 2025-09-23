# Complete MT5 CRT Power of 3 Trading System
# All-in-one file with adjustable timeframes
# Requires: pip install MetaTrader5 pandas numpy matplotlib

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import time
import logging
from typing import Dict, List, Tuple, Optional
import threading
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('crt_po3_trading.log'),
        logging.StreamHandler()
    ]
)

class MT5_CRT_PowerOfThree:
    """Complete MT5 CRT Power of 3 Trading System with Adjustable Timeframes"""
    
    def __init__(self, 
                 login: int,
                 password: str, 
                 server: str,
                 symbol: str = 'EURUSD',
                 timeframe: str = 'H1',  # Easy timeframe configuration
                 magic_number: int = 234567):
        """
        Initialize MT5 CRT Power of 3 Trading System
        
        Args:
            login: MT5 account login
            password: MT5 account password  
            server: MT5 broker server
            symbol: Trading symbol
            timeframe: Chart timeframe (M1, M5, M15, M30, H1, H4, D1, etc.)
            magic_number: Unique identifier for EA trades
        """
        self.login = login
        self.password = password
        self.server = server
        self.symbol = symbol
        
        # Timeframe mapping for easy configuration
        self.timeframe_map = {
            'M1': mt5.TIMEFRAME_M1,      # 1 minute
            'M2': mt5.TIMEFRAME_M2,      # 2 minutes  
            'M3': mt5.TIMEFRAME_M3,      # 3 minutes
            'M4': mt5.TIMEFRAME_M4,      # 4 minutes
            'M5': mt5.TIMEFRAME_M5,      # 5 minutes
            'M6': mt5.TIMEFRAME_M6,      # 6 minutes
            'M10': mt5.TIMEFRAME_M10,    # 10 minutes
            'M12': mt5.TIMEFRAME_M12,    # 12 minutes
            'M15': mt5.TIMEFRAME_M15,    # 15 minutes
            'M20': mt5.TIMEFRAME_M20,    # 20 minutes
            'M30': mt5.TIMEFRAME_M30,    # 30 minutes
            'H1': mt5.TIMEFRAME_H1,      # 1 hour
            'H2': mt5.TIMEFRAME_H2,      # 2 hours
            'H3': mt5.TIMEFRAME_H3,      # 3 hours
            'H4': mt5.TIMEFRAME_H4,      # 4 hours
            'H6': mt5.TIMEFRAME_H6,      # 6 hours
            'H8': mt5.TIMEFRAME_H8,      # 8 hours
            'H12': mt5.TIMEFRAME_H12,    # 12 hours
            'D1': mt5.TIMEFRAME_D1,      # Daily
            'W1': mt5.TIMEFRAME_W1,      # Weekly
            'MN1': mt5.TIMEFRAME_MN1     # Monthly
        }
        
        # Validate and set timeframe
        if timeframe in self.timeframe_map:
            self.timeframe_str = timeframe
            self.timeframe = self.timeframe_map[timeframe]
        else:
            logging.warning(f"Invalid timeframe {timeframe}, defaulting to H1")
            self.timeframe_str = 'H1'
            self.timeframe = mt5.TIMEFRAME_H1
            
        # Log timeframe setting
        logging.info(f"Timeframe set to: {self.timeframe_str}")
        
        self.magic_number = magic_number
        
        # Timeframe-specific configurations
        self.configure_timeframe_settings()
        
        # Trading parameters
        self.risk_percent = 2.0  # Risk per trade
        self.max_spread = 20     # Maximum spread in points
        self.min_signal_strength = 4  # Minimum signal strength to trade
        
        # Power of 3 sessions (GMT)
        self.po3_sessions = {
            'london': {'start': 2, 'end': 6},      # Accumulation
            'new_york': {'start': 13, 'end': 17},  # Manipulation
            'asia': {'start': 21, 'end': 1}        # Distribution
        }
        
        # Trading state
        self.is_trading = False
        self.positions = {}
        self.last_analysis_time = None
        
        # Initialize MT5 connection
        self.connect_mt5()
    
    def configure_timeframe_settings(self):
        """Configure timeframe-specific parameters"""
        
        # Adjust parameters based on timeframe
        timeframe_configs = {
            'M1': {
                'swing_window': 3,
                'lookback_period': 10,
                'atr_multiplier_sl': 1.5,
                'atr_multiplier_tp': 3.0,
                'trailing_distance': 10,
                'min_profit_for_trail': 15
            },
            'M5': {
                'swing_window': 3,
                'lookback_period': 12,
                'atr_multiplier_sl': 1.5,
                'atr_multiplier_tp': 3.0,
                'trailing_distance': 15,
                'min_profit_for_trail': 20
            },
            'M15': {
                'swing_window': 4,
                'lookback_period': 15,
                'atr_multiplier_sl': 1.8,
                'atr_multiplier_tp': 3.5,
                'trailing_distance': 20,
                'min_profit_for_trail': 30
            },
            'M30': {
                'swing_window': 4,
                'lookback_period': 18,
                'atr_multiplier_sl': 2.0,
                'atr_multiplier_tp': 4.0,
                'trailing_distance': 25,
                'min_profit_for_trail': 40
            },
            'H1': {
                'swing_window': 5,
                'lookback_period': 20,
                'atr_multiplier_sl': 2.0,
                'atr_multiplier_tp': 4.0,
                'trailing_distance': 30,
                'min_profit_for_trail': 50
            },
            'H4': {
                'swing_window': 5,
                'lookback_period': 25,
                'atr_multiplier_sl': 2.5,
                'atr_multiplier_tp': 5.0,
                'trailing_distance': 50,
                'min_profit_for_trail': 80
            },
            'D1': {
                'swing_window': 7,
                'lookback_period': 30,
                'atr_multiplier_sl': 3.0,
                'atr_multiplier_tp': 6.0,
                'trailing_distance': 100,
                'min_profit_for_trail': 150
            }
        }
        
        # Get configuration for current timeframe or use H1 as default
        config = timeframe_configs.get(self.timeframe_str, timeframe_configs['H1'])
        
        # Set the configuration attributes
        self.swing_window = config['swing_window']
        self.lookback_period = config['lookback_period']
        self.atr_multiplier_sl = config['atr_multiplier_sl']
        self.atr_multiplier_tp = config['atr_multiplier_tp']
        self.trailing_distance = config['trailing_distance']
        self.min_profit_for_trail = config['min_profit_for_trail']
        
        logging.info(f"Configured for {self.timeframe_str}: swing_window={self.swing_window}, "
                    f"lookback={self.lookback_period}, SL_mult={self.atr_multiplier_sl}, "
                    f"TP_mult={self.atr_multiplier_tp}")
    
    def set_timeframe(self, new_timeframe: str):
        """Change timeframe dynamically"""
        if new_timeframe in self.timeframe_map:
            old_tf = self.timeframe_str
            self.timeframe_str = new_timeframe
            self.timeframe = self.timeframe_map[new_timeframe]
            self.configure_timeframe_settings()
            logging.info(f"Timeframe changed from {old_tf} to {new_timeframe}")
            return True
        else:
            logging.error(f"Invalid timeframe: {new_timeframe}")
            logging.info(f"Available timeframes: {list(self.timeframe_map.keys())}")
            return False
    
    def get_available_timeframes(self) -> list:
        """Get list of available timeframes"""
        return list(self.timeframe_map.keys())
    
    def get_timeframe_info(self) -> dict:
        """Get current timeframe configuration info"""
        return {
            'current_timeframe': self.timeframe_str,
            'swing_window': self.swing_window,
            'lookback_period': self.lookback_period,
            'atr_multiplier_sl': self.atr_multiplier_sl,
            'atr_multiplier_tp': self.atr_multiplier_tp,
            'trailing_distance': self.trailing_distance,
            'min_profit_for_trail': self.min_profit_for_trail,
            'available_timeframes': self.get_available_timeframes()
        }
        
    def connect_mt5(self) -> bool:
        """Initialize connection to MetaTrader5"""
        try:
            if not mt5.initialize():
                logging.error(f"MT5 initialization failed: {mt5.last_error()}")
                return False
                
            # Login to account
            if not mt5.login(self.login, password=self.password, server=self.server):
                logging.error(f"MT5 login failed: {mt5.last_error()}")
                return False
                
            logging.info(f"Connected to MT5 - Account: {self.login}, Server: {self.server}")
            
            # Get symbol info
            symbol_info = mt5.symbol_info(self.symbol)
            if symbol_info is None:
                logging.error(f"Symbol {self.symbol} not found")
                return False
                
            # Enable symbol for trading
            if not mt5.symbol_select(self.symbol, True):
                logging.error(f"Failed to select symbol {self.symbol}")
                return False
                
            self.symbol_info = symbol_info
            logging.info(f"Symbol {self.symbol} selected successfully")
            return True
            
        except Exception as e:
            logging.error(f"MT5 connection error: {e}")
            return False
    
    def disconnect_mt5(self):
        """Close MT5 connection"""
        mt5.shutdown()
        logging.info("MT5 connection closed")
    
    def get_historical_data(self, bars: int = 500) -> pd.DataFrame:
        """Get historical OHLC data from MT5"""
        try:
            rates = mt5.copy_rates_from_pos(self.symbol, self.timeframe, 0, bars)
            if rates is None:
                logging.error(f"Failed to get historical data: {mt5.last_error()}")
                return None
                
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            
            # Rename columns for consistency
            df.rename(columns={
                'open': 'open',
                'high': 'high', 
                'low': 'low',
                'close': 'close',
                'tick_volume': 'volume'
            }, inplace=True)
            
            # Add hour column for session identification
            df['hour'] = df.index.hour
            
            return df
            
        except Exception as e:
            logging.error(f"Error getting historical data: {e}")
            return None
    
    def identify_market_structure(self, df: pd.DataFrame) -> pd.DataFrame:
        """Identify market structure: HH, LL, BOS"""
        try:
            data = df.copy()
            
            # Calculate swing highs and lows (using dynamic window)
            data['swing_high'] = data['high'].rolling(window=self.swing_window, center=True).max() == data['high']
            data['swing_low'] = data['low'].rolling(window=self.swing_window, center=True).min() == data['low']
            
            # Initialize structure columns
            data['structure'] = None
            data['bos'] = False
            data['trend'] = None
            
            # Track previous swing points
            prev_high = None
            prev_low = None
            current_trend = None
            
            for i in range(len(data)):
                if data.iloc[i]['swing_high']:
                    if prev_high is not None:
                        if data.iloc[i]['high'] > prev_high:
                            data.iloc[i, data.columns.get_loc('structure')] = 'HH'
                            if current_trend != 'bullish':
                                data.iloc[i, data.columns.get_loc('bos')] = True
                                current_trend = 'bullish'
                        else:
                            data.iloc[i, data.columns.get_loc('structure')] = 'LH'
                            
                    prev_high = data.iloc[i]['high']
                    
                elif data.iloc[i]['swing_low']:
                    if prev_low is not None:
                        if data.iloc[i]['low'] < prev_low:
                            data.iloc[i, data.columns.get_loc('structure')] = 'LL'
                            if current_trend != 'bearish':
                                data.iloc[i, data.columns.get_loc('bos')] = True
                                current_trend = 'bearish'
                        else:
                            data.iloc[i, data.columns.get_loc('structure')] = 'HL'
                            
                    prev_low = data.iloc[i]['low']
                
                data.iloc[i, data.columns.get_loc('trend')] = current_trend
                
            return data
            
        except Exception as e:
            logging.error(f"Error identifying market structure: {e}")
            return df
    
    def identify_liquidity_zones(self, df: pd.DataFrame) -> pd.DataFrame:
        """Identify buy-side and sell-side liquidity zones"""
        try:
            data = df.copy()
            
            # Calculate recent highs and lows (using dynamic lookback)
            data['recent_high'] = data['high'].rolling(window=self.lookback_period).max()
            data['recent_low'] = data['low'].rolling(window=self.lookback_period).min()
            
            # Identify equal highs/lows
            data['equal_highs'] = False
            data['equal_lows'] = False
            
            # Dynamic tolerance based on ATR
            data['atr'] = self.calculate_atr(data, period=14)
            tolerance = data['atr'] * 0.3
            
            for i in range(self.lookback_period, len(data)):
                # Check for equal highs
                recent_highs = data['high'][i-self.lookback_period:i]
                max_high = recent_highs.max()
                equal_count = sum(abs(recent_highs - max_high) <= tolerance.iloc[i])
                
                if equal_count >= 2:
                    data.iloc[i, data.columns.get_loc('equal_highs')] = True
                    
                # Check for equal lows
                recent_lows = data['low'][i-self.lookback_period:i]
                min_low = recent_lows.min()
                equal_count = sum(abs(recent_lows - min_low) <= tolerance.iloc[i])
                
                if equal_count >= 2:
                    data.iloc[i, data.columns.get_loc('equal_lows')] = True
                    
            return data
            
        except Exception as e:
            logging.error(f"Error identifying liquidity zones: {e}")
            return df
    
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        try:
            high = df['high']
            low = df['low']
            close = df['close'].shift(1)
            
            tr1 = high - low
            tr2 = abs(high - close)
            tr3 = abs(low - close)
            
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = true_range.rolling(window=period).mean()
            
            return atr
            
        except Exception as e:
            logging.error(f"Error calculating ATR: {e}")
            return pd.Series([0] * len(df))
    
    def identify_po3_phases(self, df: pd.DataFrame) -> pd.DataFrame:
        """Identify Power of 3 phases"""
        try:
            data = df.copy()
            data['po3_phase'] = None
            
            for i in range(len(data)):
                hour = data.iloc[i]['hour']
                
                # London session - Accumulation
                if self.po3_sessions['london']['start'] <= hour <= self.po3_sessions['london']['end']:
                    data.iloc[i, data.columns.get_loc('po3_phase')] = 'accumulation'
                    
                # New York session - Manipulation
                elif self.po3_sessions['new_york']['start'] <= hour <= self.po3_sessions['new_york']['end']:
                    data.iloc[i, data.columns.get_loc('po3_phase')] = 'manipulation'
                    
                # Asian session - Distribution
                elif (hour >= self.po3_sessions['asia']['start'] or 
                      hour <= self.po3_sessions['asia']['end']):
                    data.iloc[i, data.columns.get_loc('po3_phase')] = 'distribution'
                    
            return data
            
        except Exception as e:
            logging.error(f"Error identifying Po3 phases: {e}")
            return df
    
    def detect_crt_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect CRT signals with full analysis"""
        try:
            # Run all analysis steps
            data = self.identify_market_structure(df)
            data = self.identify_liquidity_zones(data)
            data = self.identify_po3_phases(data)
            
            # Initialize signal columns
            data['crt_signal'] = None
            data['signal_strength'] = 0
            data['entry_price'] = None
            data['stop_loss'] = None
            data['take_profit'] = None
            
            # Generate CRT signals
            for i in range(self.lookback_period, len(data)):
                current = data.iloc[i]
                signal_strength = 0
                
                # Break of Structure (highest weight)
                if current['bos']:
                    signal_strength += 3
                    
                # Liquidity sweep
                if current['equal_highs'] or current['equal_lows']:
                    signal_strength += 2
                    
                # Po3 phase alignment
                if current['po3_phase'] == 'manipulation':
                    signal_strength += 2
                elif current['po3_phase'] in ['accumulation', 'distribution']:
                    signal_strength += 1
                    
                # ATR filter (avoid low volatility periods)
                if current['atr'] > data['atr'].rolling(50).mean().iloc[i]:
                    signal_strength += 1
                
                # Store signal strength
                data.iloc[i, data.columns.get_loc('signal_strength')] = signal_strength
                
                # Generate signal if threshold met
                if signal_strength >= self.min_signal_strength:
                    atr_value = current['atr']
                    
                    if current['trend'] == 'bullish' and current['bos']:
                        data.iloc[i, data.columns.get_loc('crt_signal')] = 'BUY'
                        data.iloc[i, data.columns.get_loc('entry_price')] = current['close']
                        data.iloc[i, data.columns.get_loc('stop_loss')] = current['close'] - (atr_value * self.atr_multiplier_sl)
                        data.iloc[i, data.columns.get_loc('take_profit')] = current['close'] + (atr_value * self.atr_multiplier_tp)
                        
                    elif current['trend'] == 'bearish' and current['bos']:
                        data.iloc[i, data.columns.get_loc('crt_signal')] = 'SELL'
                        data.iloc[i, data.columns.get_loc('entry_price')] = current['close']
                        data.iloc[i, data.columns.get_loc('stop_loss')] = current['close'] + (atr_value * self.atr_multiplier_sl)
                        data.iloc[i, data.columns.get_loc('take_profit')] = current['close'] - (atr_value * self.atr_multiplier_tp)
                        
            return data
            
        except Exception as e:
            logging.error(f"Error detecting CRT signals: {e}")
            return df
    
    def calculate_lot_size(self, entry_price: float, stop_loss: float) -> float:
        """Calculate lot size based on risk percentage"""
        try:
            account_info = mt5.account_info()
            if account_info is None:
                logging.error("Failed to get account info")
                return 0.01
                
            balance = account_info.balance
            risk_amount = balance * (self.risk_percent / 100)
            
            # Calculate pip value
            pip_size = self.symbol_info.point * 10 if 'JPY' not in self.symbol else self.symbol_info.point
            pip_risk = abs(entry_price - stop_loss) / pip_size
            
            if pip_risk <= 0:
                return 0.01
                
            # Calculate lot size
            pip_value = 1.0  # For major pairs, 1 pip = $1 per standard lot
            lot_size = risk_amount / (pip_risk * pip_value)
            
            # Round to broker's lot step
            lot_step = self.symbol_info.volume_step
            lot_size = round(lot_size / lot_step) * lot_step
            
            # Apply min/max limits
            lot_size = max(self.symbol_info.volume_min, lot_size)
            lot_size = min(self.symbol_info.volume_max, lot_size)
            
            return lot_size
            
        except Exception as e:
            logging.error(f"Error calculating lot size: {e}")
            return 0.01
    
    def check_spread(self) -> bool:
        """Check if current spread is acceptable"""
        try:
            tick = mt5.symbol_info_tick(self.symbol)
            if tick is None:
                return False
                
            spread = (tick.ask - tick.bid) / self.symbol_info.point
            return spread <= self.max_spread
            
        except Exception as e:
            logging.error(f"Error checking spread: {e}")
            return False
    
    def place_order(self, signal: str, entry_price: float, stop_loss: float, take_profit: float) -> bool:
        """Place buy/sell order in MT5"""
        try:
            if not self.check_spread():
                logging.warning(f"Spread too high, skipping trade")
                return False
                
            lot_size = self.calculate_lot_size(entry_price, stop_loss)
            
            # Prepare order request
            if signal == 'BUY':
                order_type = mt5.ORDER_TYPE_BUY
                price = mt5.symbol_info_tick(self.symbol).ask
            else:
                order_type = mt5.ORDER_TYPE_SELL  
                price = mt5.symbol_info_tick(self.symbol).bid
                
            request = {
                'action': mt5.TRADE_ACTION_DEAL,
                'symbol': self.symbol,
                'volume': lot_size,
                'type': order_type,
                'price': price,
                'sl': stop_loss,
                'tp': take_profit,
                'magic': self.magic_number,
                'comment': f'CRT_Po3_{signal}_{self.timeframe_str}',
                'type_time': mt5.ORDER_TIME_GTC,
                'type_filling': mt5.ORDER_FILLING_IOC
            }
            
            # Send order
            result = mt5.order_send(request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logging.error(f"Order failed: {result.retcode} - {result.comment}")
                return False
                
            logging.info(f"Order placed successfully: {signal} {lot_size} lots at {price} ({self.timeframe_str})")
            return True
            
        except Exception as e:
            logging.error(f"Error placing order: {e}")
            return False
    
    def get_open_positions(self) -> List:
        """Get open positions for this EA"""
        try:
            positions = mt5.positions_get(symbol=self.symbol)
            if positions is None:
                return []
                
            # Filter by magic number
            ea_positions = [pos for pos in positions if pos.magic == self.magic_number]
            return ea_positions
            
        except Exception as e:
            logging.error(f"Error getting positions: {e}")
            return []
    
    def manage_positions(self):
        """Manage open positions with timeframe-adjusted trailing stops"""
        try:
            positions = self.get_open_positions()
            
            for pos in positions:
                # Dynamic trailing stop based on timeframe
                current_price = mt5.symbol_info_tick(self.symbol).bid if pos.type == 0 else mt5.symbol_info_tick(self.symbol).ask
                
                if pos.type == 0:  # BUY position
                    new_sl = current_price - (self.trailing_distance * self.symbol_info.point)
                    min_profit_needed = self.min_profit_for_trail * self.symbol_info.point
                    
                    if new_sl > pos.sl and (current_price - pos.price_open) > min_profit_needed:
                        self.modify_position(pos.ticket, new_sl, pos.tp)
                        
                else:  # SELL position
                    new_sl = current_price + (self.trailing_distance * self.symbol_info.point)
                    min_profit_needed = self.min_profit_for_trail * self.symbol_info.point
                    
                    if new_sl < pos.sl and (pos.price_open - current_price) > min_profit_needed:
                        self.modify_position(pos.ticket, new_sl, pos.tp)
                        
        except Exception as e:
            logging.error(f"Error managing positions: {e}")
    
    def modify_position(self, ticket: int, stop_loss: float, take_profit: float):
        """Modify position stop loss and take profit"""
        try:
            request = {
                'action': mt5.TRADE_ACTION_SLTP,
                'position': ticket,
                'sl': stop_loss,
                'tp': take_profit
            }
            
            result = mt5.order_send(request)
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                logging.info(f"Position {ticket} modified successfully")
                
        except Exception as e:
            logging.error(f"Error modifying position: {e}")
    
    def analyze_and_trade(self):
        """Main analysis and trading function"""
        try:
            # Get fresh data
            df = self.get_historical_data()
            if df is None or len(df) < 50:
                logging.warning("Insufficient data for analysis")
                return
                
            # Run CRT analysis
            analyzed_df = self.detect_crt_signals(df)
            
            # Check latest signal
            latest_row = analyzed_df.iloc[-1]
            
            if pd.notna(latest_row['crt_signal']):
                signal = latest_row['crt_signal']
                entry_price = latest_row['entry_price']
                stop_loss = latest_row['stop_loss']
                take_profit = latest_row['take_profit']
                signal_strength = latest_row['signal_strength']
                
                logging.info(f"CRT Signal detected: {signal} (Strength: {signal_strength}) on {self.timeframe_str}")
                
                # Check if we already have positions
                open_positions = self.get_open_positions()
                if len(open_positions) == 0:  # No open positions
                    success = self.place_order(signal, entry_price, stop_loss, take_profit)
                    if success:
                        logging.info(f"Trade executed: {signal} on {self.timeframe_str}")
                else:
                    logging.info(f"Skipping signal - already have {len(open_positions)} open position(s)")
                    
            # Manage existing positions
            self.manage_positions()
            
            self.last_analysis_time = datetime.now()
            
        except Exception as e:
            logging.error(f"Error in analyze_and_trade: {e}")
    
    def start_trading(self, analysis_interval: int = 300):  # 5 minutes default
        """Start automated trading loop"""
        self.is_trading = True
        logging.info(f"Starting CRT Po3 automated trading on {self.timeframe_str}...")
        
        def trading_loop():
            while self.is_trading:
                try:
                    # Check if market is open
                    if self.is_market_open():
                        self.analyze_and_trade()
                    else:
                        logging.info("Market is closed")
                        
                    time.sleep(analysis_interval)
                    
                except Exception as e:
                    logging.error(f"Error in trading loop: {e}")
                    time.sleep(60)  # Wait 1 minute on error
                    
        # Start trading in separate thread
        self.trading_thread = threading.Thread(target=trading_loop)
        self.trading_thread.daemon = True
        self.trading_thread.start()
    
    def stop_trading(self):
        """Stop automated trading"""
        self.is_trading = False
        logging.info("Stopping automated trading...")
        
        if hasattr(self, 'trading_thread'):
            self.trading_thread.join(timeout=10)
    
    def is_market_open(self) -> bool:
        """Check if forex market is open (simplified)"""
        try:
            current_time = datetime.now(timezone.utc)
            weekday = current_time.weekday()
            
            # Forex market is closed on weekends
            if weekday == 5:  # Saturday
                return current_time.hour >= 22  # Opens Sunday 22:00 UTC
            elif weekday == 6:  # Sunday
                return current_time.hour >= 22  # Opens Sunday 22:00 UTC
            elif weekday == 4:  # Friday
                return current_time.hour < 22   # Closes Friday 22:00 UTC
            else:
                return True  # Monday-Thursday, market is open
                
        except Exception as e:
            logging.error(f"Error checking market hours: {e}")
            return True  # Default to open
    
    def get_trading_stats(self) -> Dict:
        """Get trading statistics"""
        try:
            # Get closed trades
            from_date = datetime.now() - timedelta(days=30)
            
            deals = mt5.history_deals_get(from_date, datetime.now())
            if deals is None:
                return {'error': 'No deals found'}
                
            # Filter by magic number
            ea_deals = [deal for deal in deals if deal.magic == self.magic_number]
            
            if not ea_deals:
                return {'message': 'No EA trades found'}
                
            total_profit = sum(deal.profit for deal in ea_deals)
            total_trades = len([d for d in ea_deals if d.entry == 1])  # Out deals only
            winning_trades = len([d for d in ea_deals if d.profit > 0])
            
            stats = {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': total_trades - winning_trades,
                'win_rate': (winning_trades / total_trades * 100) if total_trades > 0 else 0,
                'total_profit': total_profit,
                'open_positions': len(self.get_open_positions()),
                'timeframe': self.timeframe_str
            }
            
            return stats
            
        except Exception as e:
            logging.error(f"Error getting trading stats: {e}")
            return {'error': str(e)}


# Multi-Timeframe Analysis Functions
def quick_analysis_example(login: int, password: str, server: str):
    """Example of running analysis on different timeframes"""
    
    print("=== Quick Analysis with Multiple Timeframes ===\n")
    
    # Test different timeframes
    timeframes_to_test = ['M15', 'H1', 'H4']
    
    for tf in timeframes_to_test:
        print(f"Analyzing {tf} timeframe...")
        
        try:
            # Initialize trader with specific timeframe
            trader = MT5_CRT_PowerOfThree(
                login=login,
                password=password,
                server=server,
                symbol="EURUSD",
                timeframe=tf  # Easy timeframe setting
            )
            
            # Get timeframe info
            tf_info = trader.get_timeframe_info()
            print(f"Configuration: SL={tf_info['atr_multiplier_sl']}x ATR, TP={tf_info['atr_multiplier_tp']}x ATR")
            
            # Get historical data and analyze
            df = trader.get_historical_data(100)
            if df is not None:
                analyzed_df = trader.detect_crt_signals(df)
                
                # Find recent signals
                signals = analyzed_df[analyzed_df['crt_signal'].notna()]
                print(f"Found {len(signals)} signals")
                
                # Show latest signal if available
                if len(signals) > 0:
                    latest = signals.iloc[-1]
                    print(f"Latest Signal: {latest['crt_signal']} (Strength: {latest['signal_strength']})")
                else:
                    print("No recent signals found")
            else:
                print("Failed to get data")
            
            trader.disconnect_mt5()
            
        except Exception as e:
            print(f"Error analyzing {tf}: {e}")
            
        print("-" * 40)


class MultiTimeframeAnalyzer:
    """Analyze CRT Po3 signals across multiple timeframes simultaneously"""
    
    def __init__(self, login: int, password: str, server: str):
        self.login = login
        self.password = password
        self.server = server
        self.traders = {}
        
    def initialize_timeframes(self, symbol: str, timeframes: list):
        """Initialize traders for multiple timeframes"""
        
        print(f"Initializing {symbol} for timeframes: {', '.join(timeframes)}")
        
        for tf in timeframes:
            try:
                trader = MT5_CRT_PowerOfThree(
                    self.login, self.password, self.server,
                    symbol=symbol, timeframe=tf
                )
                
                # Test connection
                if hasattr(trader, 'symbol_info') and trader.symbol_info is not None:
                    self.traders[tf] = trader
                    config = trader.get_timeframe_info()
                    print(f"✓ {tf}: SL={config['atr_multiplier_sl']}x, TP={config['atr_multiplier_tp']}x ATR")
                else:
                    print(f"✗ {tf}: Connection failed")
                    if hasattr(trader, 'disconnect_mt5'):
                        trader.disconnect_mt5()
                        
            except Exception as e:
                print(f"✗ {tf}: Error - {e}")
    
    def analyze_all_timeframes(self) -> pd.DataFrame:
        """Analyze signals across all initialized timeframes"""
        
        results = []
        
        for tf, trader in self.traders.items():
            try:
                print(f"Analyzing {tf} timeframe...")
                
                df = trader.get_historical_data(100)
                if df is not None and len(df) > 50:
                    analyzed_df = trader.detect_crt_signals(df)
                    
                    # Get latest data
                    latest = analyzed_df.iloc[-1]
                    signals = analyzed_df[analyzed_df['crt_signal'].notna()]
                    
                    result = {
                        'timeframe': tf,
                        'current_price': latest['close'],
                        'signal': latest['crt_signal'] if pd.notna(latest['crt_signal']) else 'NONE',
                        'signal_strength': latest['signal_strength'],
                        'trend': latest['trend'],
                        'po3_phase': latest['po3_phase'],
                        'bos': latest['bos'],
                        'total_signals': len(signals),
                        'timestamp': latest.name,
                        'atr_sl': trader.atr_multiplier_sl,
                        'atr_tp': trader.atr_multiplier_tp
                    }
                    
                    results.append(result)
                    
            except Exception as e:
                print(f"Error analyzing {tf}: {e}")
        
        return pd.DataFrame(results)
    
    def get_consensus_signal(self, results_df: pd.DataFrame) -> dict:
        """Determine consensus signal across timeframes"""
        
        if results_df.empty:
            return {'consensus': 'NO_DATA', 'confidence': 0}
            
        # Filter active signals
        active_signals = results_df[results_df['signal'] != 'NONE']
        
        if active_signals.empty:
            return {'consensus': 'NO_SIGNAL', 'confidence': 0}
        
        # Count signals by type
        buy_signals = len(active_signals[active_signals['signal'] == 'BUY'])
        sell_signals = len(active_signals[active_signals['signal'] == 'SELL'])
        
        # Calculate consensus
        total_signals = len(active_signals)
        
        if buy_signals > sell_signals:
            consensus = 'BUY'
            confidence = (buy_signals / total_signals) * 100
        elif sell_signals > buy_signals:
            consensus = 'SELL'  
            confidence = (sell_signals / total_signals) * 100
        else:
            consensus = 'MIXED'
            confidence = 50
            
        # Weight by signal strength
        avg_strength = active_signals['signal_strength'].mean()
        
        return {
            'consensus': consensus,
            'confidence': confidence,
            'avg_strength': avg_strength,
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'total_active': total_signals,
            'timeframes': list(active_signals['timeframe'])
        }
    
    def cleanup(self):
        """Close all trader connections"""
        for trader in self.traders.values():
            trader.disconnect_mt5()


def display_timeframe_menu():
    """Display timeframe selection menu"""
    timeframes = {
        1: 'M1', 2: 'M5', 3: 'M15', 4: 'M30', 
        5: 'H1', 6: 'H4', 7: 'D1'
    }
    
    print("\nAvailable Timeframes:")
    for num, tf in timeframes.items():
        print(f"{num}. {tf}")
    
    return timeframes


def main():
    """Main function to run the CRT Po3 trading system"""
    
    # MT5 Account Configuration - REPLACE WITH YOUR ACTUAL CREDENTIALS
    MT5_CONFIG = {
        'login': 123456789,           # Your MT5 account number
        'password': 'your_password',  # Your MT5 password  
        'server': 'YourBroker-Demo'   # Your broker's server name
    }
    
    # Trading Configuration
    TRADING_CONFIG = {
        'symbol': 'EURUSD',
        'timeframe': 'H1',  # Easy to change: M1, M5, M15, M30, H1, H4, D1
        'magic_number': 234567,
        'risk_percent': 2.0,
        'max_spread': 20,
        'analysis_interval': 300  # 5 minutes
    }
    
    print("=== CRT Power of 3 Trading System ===")
    print("1. Start Automated Trading")
    print("2. Manual Analysis Only") 
    print("3. View Trading Statistics")
    print("4. Change Timeframe")
    print("5. Multi-Timeframe Analysis")
    print("6. View Timeframe Info")
    print("7. Quick Timeframe Test")
    print("8. Exit")
    
    choice = input("\nSelect option (1-8): ").strip()
    
    try:
        if choice == '1':
            # Start automated trading
            trader = MT5_CRT_PowerOfThree(
                login=MT5_CONFIG['login'],
                password=MT5_CONFIG['password'],
                server=MT5_CONFIG['server'],
                symbol=TRADING_CONFIG['symbol'],
                timeframe=TRADING_CONFIG['timeframe'],
                magic_number=TRADING_CONFIG['magic_number']
            )
            
            # Set trading parameters
            trader.risk_percent = TRADING_CONFIG['risk_percent']
            trader.max_spread = TRADING_CONFIG['max_spread']
            
            print(f"\nStarting automated trading on {trader.timeframe_str} timeframe...")
            trader.start_trading(TRADING_CONFIG['analysis_interval'])
            
            print("Automated trading started. Press Ctrl+C to stop...")
            try:
                while True:
                    time.sleep(60)
                    # Print periodic status
                    stats = trader.get_trading_stats()
                    current_time = datetime.now().strftime('%H:%M:%S')
                    open_pos = stats.get('open_positions', 0)
                    print(f"Status: {current_time} - Open Positions: {open_pos} - TF: {trader.timeframe_str}")
                    
            except KeyboardInterrupt:
                print("\nStopping trading...")
                trader.stop_trading()
                trader.disconnect_mt5()
                
        elif choice == '2':
            # Manual analysis
            trader = MT5_CRT_PowerOfThree(
                login=MT5_CONFIG['login'],
                password=MT5_CONFIG['password'],
                server=MT5_CONFIG['server'],
                symbol=TRADING_CONFIG['symbol'],
                timeframe=TRADING_CONFIG['timeframe'],
                magic_number=TRADING_CONFIG['magic_number']
            )
            
            print(f"Running manual analysis on {trader.timeframe_str}...")
            trader.analyze_and_trade()
            
            # Get recent data for display
            df = trader.get_historical_data(100)
            analyzed_df = trader.detect_crt_signals(df)
            
            print(f"\n=== Recent Signals on {trader.timeframe_str} ===")
            recent_signals = analyzed_df[analyzed_df['crt_signal'].notna()].tail(5)
            
            if len(recent_signals) > 0:
                for idx, row in recent_signals.iterrows():
                    print(f"{idx}: {row['crt_signal']} - Strength: {row['signal_strength']} - Price: {row['close']:.5f}")
            else:
                print("No recent signals found")
                
            trader.disconnect_mt5()
                
        elif choice == '3':
            # Show statistics
            trader = MT5_CRT_PowerOfThree(
                login=MT5_CONFIG['login'],
                password=MT5_CONFIG['password'],
                server=MT5_CONFIG['server'],
                symbol=TRADING_CONFIG['symbol'],
                timeframe=TRADING_CONFIG['timeframe'],
                magic_number=TRADING_CONFIG['magic_number']
            )
            
            stats = trader.get_trading_stats()
            print(f"\n=== Trading Statistics (Last 30 days) - {stats.get('timeframe', 'N/A')} ===")
            
            for key, value in stats.items():
                if key in ['total_profit']:
                    print(f"{key.replace('_', ' ').title()}: ${value:.2f}")
                elif key in ['win_rate']:
                    print(f"{key.replace('_', ' ').title()}: {value:.1f}%")
                else:
                    print(f"{key.replace('_', ' ').title()}: {value}")
                    
            trader.disconnect_mt5()
                    
        elif choice == '4':
            # Change timeframe
            trader = MT5_CRT_PowerOfThree(
                login=MT5_CONFIG['login'],
                password=MT5_CONFIG['password'],
                server=MT5_CONFIG['server'],
                symbol=TRADING_CONFIG['symbol'],
                timeframe=TRADING_CONFIG['timeframe'],
                magic_number=TRADING_CONFIG['magic_number']
            )
            
            print(f"\nCurrent timeframe: {trader.timeframe_str}")
            
            # Show available timeframes with numbers
            timeframe_options = display_timeframe_menu()
            
            try:
                selection = input("\nEnter timeframe number or code (e.g., 5 or H1): ").strip()
                
                # Check if it's a number (selection from menu)
                if selection.isdigit():
                    selection_num = int(selection)
                    if selection_num in timeframe_options:
                        new_tf = timeframe_options[selection_num]
                    else:
                        print("Invalid selection number")
                        trader.disconnect_mt5()
                        return
                else:
                    # Direct timeframe code entry
                    new_tf = selection.upper()
                
                if trader.set_timeframe(new_tf):
                    print(f"✓ Timeframe changed to {new_tf}")
                    TRADING_CONFIG['timeframe'] = new_tf  # Update config
                    
                    # Show new configuration
                    config = trader.get_timeframe_info()
                    print(f"New settings: SL={config['atr_multiplier_sl']}x ATR, TP={config['atr_multiplier_tp']}x ATR")
                else:
                    print("✗ Failed to change timeframe")
                    
            except ValueError:
                print("Invalid input")
                
            trader.disconnect_mt5()
            
        elif choice == '5':
            # Multi-timeframe analysis
            print("\n=== Multi-Timeframe Analysis ===")
            
            timeframes = ['M15', 'H1', 'H4']  # Analyze multiple timeframes
            
            analyzer = MultiTimeframeAnalyzer(
                MT5_CONFIG['login'],
                MT5_CONFIG['password'], 
                MT5_CONFIG['server']
            )
            
            analyzer.initialize_timeframes(TRADING_CONFIG['symbol'], timeframes)
            
            if analyzer.traders:
                results = analyzer.analyze_all_timeframes()
                
                if not results.empty:
                    print(f"\n=== Analysis Results for {TRADING_CONFIG['symbol']} ===")
                    
                    for _, row in results.iterrows():
                        tf = row['timeframe']
                        signal = row['signal']
                        strength = row['signal_strength']
                        trend = row['trend']
                        phase = row['po3_phase']
                        
                        status = f"{tf}: {signal}"
                        if signal != 'NONE':
                            status += f" (Strength: {strength})"
                        status += f" | Trend: {trend} | Phase: {phase}"
                        
                        print(status)
                    
                    # Get consensus
                    consensus = analyzer.get_consensus_signal(results)
                    print(f"\n=== CONSENSUS ===")
                    print(f"Signal: {consensus['consensus']}")
                    print(f"Confidence: {consensus['confidence']:.1f}%")
                    print(f"Average Strength: {consensus.get('avg_strength', 0):.1f}")
                    print(f"Active Timeframes: {', '.join(consensus.get('timeframes', []))}")
                else:
                    print("No analysis results available")
            else:
                print("No timeframes successfully initialized")
                
            analyzer.cleanup()
                
        elif choice == '6':
            # View timeframe info
            trader = MT5_CRT_PowerOfThree(
                login=MT5_CONFIG['login'],
                password=MT5_CONFIG['password'],
                server=MT5_CONFIG['server'],
                symbol=TRADING_CONFIG['symbol'],
                timeframe=TRADING_CONFIG['timeframe'],
                magic_number=TRADING_CONFIG['magic_number']
            )
            
            info = trader.get_timeframe_info()
            print(f"\n=== Timeframe Configuration ===")
            print(f"Current Timeframe: {info['current_timeframe']}")
            print(f"Swing Window: {info['swing_window']} periods")
            print(f"Lookback Period: {info['lookback_period']} periods")
            print(f"Stop Loss Multiplier: {info['atr_multiplier_sl']}x ATR")
            print(f"Take Profit Multiplier: {info['atr_multiplier_tp']}x ATR")
            print(f"Trailing Distance: {info['trailing_distance']} points")
            print(f"Min Profit for Trail: {info['min_profit_for_trail']} points")
            print(f"\nAvailable Timeframes: {', '.join(info['available_timeframes'])}")
            
            trader.disconnect_mt5()
            
        elif choice == '7':
            # Quick timeframe test
            print("Running quick analysis on multiple timeframes...")
            quick_analysis_example(
                MT5_CONFIG['login'],
                MT5_CONFIG['password'],
                MT5_CONFIG['server']
            )
            
        elif choice == '8':
            print("Exiting...")
        else:
            print("Invalid option selected")
            
    except Exception as e:
        print(f"Error: {e}")
        logging.error(f"Main function error: {e}")


if __name__ == "__main__":
    print("CRT Power of 3 MT5 Trading System")
    print("Make sure to update your MT5 credentials in the MT5_CONFIG section!")
    print("=" * 60)
    main()