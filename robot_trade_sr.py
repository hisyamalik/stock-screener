import MetaTrader5 as mt5 
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from enum import Enum

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TradingStrategy(Enum):
    """Available trading strategies"""
    SUPPORT_RESISTANCE_SWING = "sr_swing"
    SUPPORT_RESISTANCE_SCALP_1M = "sr_scalp_1m"
    SUPPORT_RESISTANCE_SCALP_5M = "sr_scalp_5m"
    SUPPORT_RESISTANCE_SCALP_15M = "sr_scalp_15m"
    CUSTOM = "custom"

class SupportResistanceRobot:
    """
    MT5 Forex Trading Robot using Support & Resistance Strategy
    - Identifies key support and resistance levels
    - Enters trades on bounces from S/R levels
    - Uses multiple timeframe analysis
    - Advanced risk management with drawdown protection
    """
    
    def __init__(self, strategy: TradingStrategy = TradingStrategy.SUPPORT_RESISTANCE_SCALP_1M,
                 custom_config: Dict = None):
        """
        Initialize Support & Resistance Trading Robot
        
        Args:
            strategy: Trading strategy configuration
            custom_config: Custom configuration overrides
        """
        self.strategy = strategy
        self.is_running = False
        self.trading_stopped_due_to_drawdown = False
        
        # Load strategy configuration
        self.config = self._load_strategy_config(strategy, custom_config)
        self._apply_config()
        
        # Initialize tracking variables
        self.equity_history = []
        self.initial_equity = 0.0
        self.peak_equity = 0.0
        self.support_resistance_cache = {} # Cache S/R levels
        
        # Initialize MT5
        self.initialize_mt5()
    
    def _load_strategy_config(self, strategy: TradingStrategy, custom_config: Dict = None) -> Dict:
        """Load configuration based on strategy"""
        
        if strategy == TradingStrategy.SUPPORT_RESISTANCE_SWING:
            config = {
                # Core Parameters
                'risk_per_trade': 0.02,
                'magic_number': 88888,
                'max_drawdown_percent': 20.0,
                'drawdown_period_hours': 24,
                
                # Support/Resistance Parameters
                'sr_lookback_periods': 50, # Bars to look back for S/R
                'sr_touch_threshold': 3, # Minimum touches to confirm S/R
                'sr_proximity_pips': 5, # How close price needs to be to S/R
                'sr_strength_periods': 20, # Recent periods to check S/R strength
                'pivot_strength': 2, # Bars each side for pivot identification
                
                # Entry Filters
                'bounce_confirmation_bars': 2, # Bars to confirm bounce
                'min_sr_age_bars': 5, # Minimum age of S/R level
                'max_sr_age_bars': 200, # Maximum age of S/R level
                
                # Risk Management
                'stop_loss_pips': 25, # Fixed pip stop loss
                'take_profit_ratio': 2.0, # Risk:Reward ratio
                'max_positions_per_symbol': 1,
                'max_total_positions': 4,
                
                # Timing
                'timeframe': mt5.TIMEFRAME_H1,
                'analysis_timeframe': mt5.TIMEFRAME_H4, # Higher TF for S/R
                'analysis_period': 200,
                'cycle_delay_seconds': 300, # 5 minutes
                
                # Trading Hours
                'trading_start_hour': 6, # 6 AM GMT
                'trading_end_hour': 22, # 10 PM GMT
                
                # Symbols
                'symbols': ['EURUSD', 'GBPUSD', 'USDCAD', 'AUDUSD'],
                
                # Filters
                'max_spread_points': 4,
                'min_distance_from_sr': 0.0010, # Minimum distance to avoid noise
                
                'strategy_name': 'S/R Swing Trading',
                'description': 'H1 entries based on H4 support/resistance levels'
            }
        
        elif strategy == TradingStrategy.SUPPORT_RESISTANCE_SCALP_1M:
            config = {
                'risk_per_trade': 0.005,
                'magic_number': 81111,
                'max_drawdown_percent': 8.0,
                'drawdown_period_hours': 2,
                'sr_lookback_periods': 100,
                'sr_touch_threshold': 2,
                'sr_proximity_pips': 2,
                'sr_strength_periods': 10,
                'pivot_strength': 1,
                'bounce_confirmation_bars': 1,
                'min_sr_age_bars': 3,
                'max_sr_age_bars': 50,
                'stop_loss_pips': 8,
                'take_profit_ratio': 1.5,
                'max_positions_per_symbol': 1,
                'max_total_positions': 2,
                'timeframe': mt5.TIMEFRAME_M1,
                'analysis_timeframe': mt5.TIMEFRAME_M15,
                'analysis_period': 150,
                'cycle_delay_seconds': 15,
                'trading_start_hour': 13,
                'trading_end_hour': 17,
                'symbols': ['XAUUSD'],
                'max_spread_points': 2,
                'min_distance_from_sr': 0.0005,
                'strategy_name': 'S/R 1M Scalping',
                'description': '1M entries on M15 support/resistance bounces'
            }
        
        elif strategy == TradingStrategy.SUPPORT_RESISTANCE_SCALP_5M:
            config = {
                'risk_per_trade': 0.002,
                'magic_number': 85555,
                'max_drawdown_percent': 12.0,
                'drawdown_period_hours': 4,
                'sr_lookback_periods': 80,
                'sr_touch_threshold': 2,
                'sr_proximity_pips': 3,
                'sr_strength_periods': 15,
                'pivot_strength': 2,
                'bounce_confirmation_bars': 1,
                'min_sr_age_bars': 5,
                'max_sr_age_bars': 100,
                'stop_loss_pips': 15,
                'take_profit_ratio': 2.0,
                'max_positions_per_symbol': 1,
                'max_total_positions': 3,
                'timeframe': mt5.TIMEFRAME_M5,
                'analysis_timeframe': mt5.TIMEFRAME_M30,
                'analysis_period': 120,
                'cycle_delay_seconds': 30,
                'trading_start_hour': 8,
                'trading_end_hour': 18,
                'symbols': ['XAUUSD'],
                'max_spread_points': 3,
                'min_distance_from_sr': 0.0008,
                'strategy_name': 'S/R 5M Scalping',
                'description': '5M entries on M30 support/resistance levels'
            }
        
        elif strategy == TradingStrategy.SUPPORT_RESISTANCE_SCALP_15M:
            config = {
                'risk_per_trade': 0.015,
                'magic_number': 81515,
                'max_drawdown_percent': 15.0,
                'drawdown_period_hours': 8,
                'sr_lookback_periods': 100,
                'sr_touch_threshold': 3,
                'sr_proximity_pips': 5,
                'sr_strength_periods': 20,
                'pivot_strength': 2,
                'bounce_confirmation_bars': 2,
                'min_sr_age_bars': 5,
                'max_sr_age_bars': 150,
                'stop_loss_pips': 20,
                'take_profit_ratio': 2.5,
                'max_positions_per_symbol': 1,
                'max_total_positions': 4,
                'timeframe': mt5.TIMEFRAME_M15,
                'analysis_timeframe': mt5.TIMEFRAME_H1,
                'analysis_period': 150,
                'cycle_delay_seconds': 60,
                'trading_start_hour': 7,
                'trading_end_hour': 20,
                'symbols': ['EURUSD', 'GBPUSD', 'USDCAD', 'AUDUSD'],
                'max_spread_points': 4,
                'min_distance_from_sr': 0.0010,
                'strategy_name': 'S/R 15M Scalping',
                'description': '15M entries on H1 support/resistance levels'
            }
        
        else: # CUSTOM
            config = custom_config or self._get_default_config()
        
        # Apply custom overrides
        if custom_config:
            config.update(custom_config)
        
        return config
    
    def _get_default_config(self) -> Dict:
        """Default configuration"""
        return {
            'risk_per_trade': 0.02,
            'magic_number': 87777,
            'max_drawdown_percent': 20.0,
            'drawdown_period_hours': 12,
            'sr_lookback_periods': 100,
            'sr_touch_threshold': 3,
            'sr_proximity_pips': 5,
            'sr_strength_periods': 20,
            'pivot_strength': 2,
            'bounce_confirmation_bars': 2,
            'min_sr_age_bars': 5,
            'max_sr_age_bars': 100,
            'stop_loss_pips': 20,
            'take_profit_ratio': 2.0,
            'max_positions_per_symbol': 1,
            'max_total_positions': 3,
            'timeframe': mt5.TIMEFRAME_M15,
            'analysis_timeframe': mt5.TIMEFRAME_H1,
            'analysis_period': 100,
            'cycle_delay_seconds': 60,
            'trading_start_hour': 8,
            'trading_end_hour': 18,
            'symbols': ['EURUSD', 'GBPUSD'],
            'max_spread_points': 3,
            'min_distance_from_sr': 0.0008,
            'strategy_name': 'Custom S/R Strategy',
            'description': 'Custom support/resistance configuration'
        }
    
    def _apply_config(self):
        """Apply configuration to robot parameters"""
        for key, value in self.config.items():
            setattr(self, key, value)
    
    def initialize_mt5(self) -> bool:
        """Initialize MT5 connection"""
        try:
            if not mt5.initialize():
                logger.error(f"MT5 initialization failed, error code = {mt5.last_error()}")
                return False
            
            account_info = mt5.account_info()
            if account_info is None:
                logger.error("Failed to get account info")
                return False
            
            logger.info(f"🎯 Support & Resistance Trading Robot Initialized")
            logger.info(f"📊 Strategy: {self.strategy_name}")
            logger.info(f"💰 Account: {account_info.login} | Balance: ${account_info.balance:.2f}")
            
            # Initialize equity tracking
            self.initial_equity = account_info.equity
            self.peak_equity = account_info.equity
            self.record_equity_snapshot()
            
            logger.info(f"🛡️ Drawdown Protection: {self.max_drawdown_percent}% over {self.drawdown_period_hours}h")
            logger.info(f"⚙️ Magic Number: {self.magic_number}")
            
            return True
            
        except Exception as e:
            logger.error(f"MT5 initialization failed: {e}")
            return False
    
    def get_market_data(self, symbol: str, timeframe=None, count: int = None) -> Optional[pd.DataFrame]:
        """Get market data from MT5"""
        try:
            tf = timeframe or self.timeframe
            cnt = count or self.analysis_period
            
            rates = mt5.copy_rates_from_pos(symbol, tf, 0, cnt)
            if rates is None:
                return None
            
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            return df
            
        except Exception as e:
            logger.error(f"Error getting market data for {symbol}: {e}")
            return None
    
    def identify_pivot_points(self, df: pd.DataFrame, strength: int = 2) -> Dict[str, List[Dict]]:
        """
        Identify pivot highs and lows in price data
        
        Args:
            df: DataFrame with OHLC data
            strength: Number of bars each side to confirm pivot
            
        Returns:
            Dictionary with 'highs' and 'lows' containing pivot points
        """
        try:
            highs = []
            lows = []
            
            for i in range(strength, len(df) - strength):
                # Check for pivot high
                is_pivot_high = True
                current_high = df.iloc[i]['high']
                
                for j in range(i - strength, i + strength + 1):
                    if j != i and df.iloc[j]['high'] >= current_high:
                        is_pivot_high = False
                        break
                
                if is_pivot_high:
                    highs.append({
                        'price': current_high,
                        'time': df.iloc[i]['time'],
                        'index': i,
                        'bar_count': len(df) - i # Age in bars
                    })
                
                # Check for pivot low
                is_pivot_low = True
                current_low = df.iloc[i]['low']
                
                for j in range(i - strength, i + strength + 1):
                    if j != i and df.iloc[j]['low'] <= current_low:
                        is_pivot_low = False
                        break
                
                if is_pivot_low:
                    lows.append({
                        'price': current_low,
                        'time': df.iloc[i]['time'],
                        'index': i,
                        'bar_count': len(df) - i # Age in bars
                    })
            
            return {'highs': highs, 'lows': lows}
            
        except Exception as e:
            logger.error(f"Error identifying pivot points: {e}")
            return {'highs': [], 'lows': []}
    
    def find_support_resistance_levels(self, df: pd.DataFrame, symbol: str) -> Dict[str, List[Dict]]:
        """
        Find strong support and resistance levels
        
        Args:
            df: DataFrame with OHLC data
            symbol: Currency pair symbol
            
        Returns:
            Dictionary with 'resistance' and 'support' levels
        """
        try:
            # Get pivot points
            pivots = self.identify_pivot_points(df, self.pivot_strength)
            
            # Group nearby pivot points into levels
            resistance_levels = self._group_pivot_levels(pivots['highs'], df, symbol)
            support_levels = self._group_pivot_levels(pivots['lows'], df, symbol)
            
            # Filter levels by strength and age
            resistance_levels = self._filter_sr_levels(resistance_levels, 'resistance')
            support_levels = self._filter_sr_levels(support_levels, 'support')
            
            return {
                'resistance': resistance_levels,
                'support': support_levels
            }
            
        except Exception as e:
            logger.error(f"Error finding S/R levels: {e}")
            return {'resistance': [], 'support': []}
    
    def _group_pivot_levels(self, pivots: List[Dict], df: pd.DataFrame, symbol: str,
                           tolerance_pips: int = None) -> List[Dict]:
        """Group nearby pivot points into support/resistance levels"""
        if not pivots:
            return []
        
        tolerance = tolerance_pips or self.sr_proximity_pips
        symbol_info = mt5.symbol_info(symbol)
        if not symbol_info:
            return []
        
        pip_value = symbol_info.point * 10 # Convert points to pips
        tolerance_price = tolerance * pip_value
        
        levels = []
        used_pivots = set()
        
        for i, pivot in enumerate(pivots):
            if i in used_pivots:
                continue
            
            # Find all pivots near this price level
            level_pivots = [pivot]
            used_pivots.add(i)
            
            for j, other_pivot in enumerate(pivots):
                if j in used_pivots or j == i:
                    continue
                
                if abs(pivot['price'] - other_pivot['price']) <= tolerance_price:
                    level_pivots.append(other_pivot)
                    used_pivots.add(j)
            
            # Create level if we have enough touches
            if len(level_pivots) >= self.sr_touch_threshold:
                avg_price = sum(p['price'] for p in level_pivots) / len(level_pivots)
                min_age = min(p['bar_count'] for p in level_pivots)
                max_age = max(p['bar_count'] for p in level_pivots)
                
                levels.append({
                    'price': avg_price,
                    'strength': len(level_pivots),
                    'touches': level_pivots,
                    'min_age_bars': min_age,
                    'max_age_bars': max_age,
                    'last_touch_age': min_age
                })
        
        return levels
    
    def _filter_sr_levels(self, levels: List[Dict], level_type: str) -> List[Dict]:
        """Filter S/R levels by age and strength"""
        filtered = []
        
        for level in levels:
            # Check age requirements
            if (level['min_age_bars'] >= self.min_sr_age_bars and 
                level['max_age_bars'] <= self.max_sr_age_bars):
                
                # Add additional strength metrics
                level['type'] = level_type
                level['quality_score'] = self._calculate_level_quality(level)
                filtered.append(level)
        
        # Sort by quality score (descending)
        filtered.sort(key=lambda x: x['quality_score'], reverse=True)
        
        return filtered[:10] # Keep top 10 levels
    
    def _calculate_level_quality(self, level: Dict) -> float:
        """Calculate quality score for S/R level"""
        try:
            # Base score from number of touches
            score = level['strength'] * 10
            
            # Bonus for recent activity
            if level['last_touch_age'] <= 20:
                score += 20
            elif level['last_touch_age'] <= 50:
                score += 10
            
            # Bonus for older, established levels
            if level['max_age_bars'] > 100:
                score += 15
            
            # Penalty for very old untested levels
            if level['last_touch_age'] > 100:
                score -= 10
            
            return max(score, 0)
            
        except:
            return 0
    
    def check_support_resistance_entry(self, symbol: str) -> str:
        """
        Check for support/resistance entry opportunities
        
        Returns:
            'BUY' for support bounce, 'SELL' for resistance rejection, 'HOLD' otherwise
        """
        try:
            # Get higher timeframe data for S/R identification
            df_higher = self.get_market_data(symbol, self.analysis_timeframe, self.analysis_period)
            if df_higher is None or len(df_higher) < self.sr_lookback_periods:
                return 'HOLD'
            
            # Get current timeframe data for entry signals
            df_current = self.get_market_data(symbol, self.timeframe, 1000)
            if df_current is None or len(df_current) < self.bounce_confirmation_bars + 1:
                return 'HOLD'
            
            # Find S/R levels on higher timeframe (pass symbol parameter)
            sr_levels = self.find_support_resistance_levels(df_higher, symbol)
            
            # Get current price
            tick = mt5.symbol_info_tick(symbol)
            if not tick:
                return 'HOLD'
            
            current_price = (tick.bid + tick.ask) / 2
            symbol_info = mt5.symbol_info(symbol)
            if not symbol_info:
                return 'HOLD'
                
            pip_value = symbol_info.point * 10
            
            # Check for support bounce (BUY signal)
            for support in sr_levels['support']:
                distance_pips = abs(current_price - support['price']) / pip_value
                
                if distance_pips <= self.sr_proximity_pips:
                    # Check if price is bouncing from support
                    if self._check_bounce_from_support(df_current, support['price']):
                        logger.info(f"🟢 Support bounce detected at {support['price']:.5f} "
                                  f"(Strength: {support['strength']}, Quality: {support['quality_score']:.1f})")
                        return 'BUY'
            
            # Check for resistance rejection (SELL signal)
            for resistance in sr_levels['resistance']:
                distance_pips = abs(current_price - resistance['price']) / pip_value
                
                if distance_pips <= self.sr_proximity_pips:
                    # Check if price is rejecting from resistance
                    if self._check_rejection_from_resistance(df_current, resistance['price']):
                        logger.info(f"🔴 Resistance rejection detected at {resistance['price']:.5f} "
                                  f"(Strength: {resistance['strength']}, Quality: {resistance['quality_score']:.1f})")
                        return 'SELL'
            
            return 'HOLD'
            
        except Exception as e:
            logger.error(f"Error checking S/R entry for {symbol}: {e}")
            return 'HOLD'
    
    def _check_bounce_from_support(self, df: pd.DataFrame, support_level: float) -> bool:
        """Check if price is bouncing from support level"""
        try:
            if len(df) < self.bounce_confirmation_bars + 1:
                return False
            
            recent_bars = df.tail(self.bounce_confirmation_bars + 1)
            
            # Check if price touched or went below support and then bounced back
            touched_support = False
            bounced_up = False
            
            for i, row in recent_bars.iterrows():
                # Check if low touched or broke support
                if row['low'] <= support_level + self.min_distance_from_sr:
                    touched_support = True
                
                # Check if close is now above support (bounce)
                if touched_support and row['close'] > support_level + self.min_distance_from_sr:
                    bounced_up = True
            
            # Additional confirmation: recent bar should be bullish
            last_bar = recent_bars.iloc[-1]
            is_bullish_bar = last_bar['close'] > last_bar['open']
            
            return touched_support and bounced_up and is_bullish_bar
            
        except Exception as e:
            logger.error(f"Error checking support bounce: {e}")
            return False
    
    def _check_rejection_from_resistance(self, df: pd.DataFrame, resistance_level: float) -> bool:
        """Check if price is rejecting from resistance level"""
        try:
            if len(df) < self.bounce_confirmation_bars + 1:
                return False
            
            recent_bars = df.tail(self.bounce_confirmation_bars + 1)
            
            # Check if price touched or went above resistance and then rejected
            touched_resistance = False
            rejected_down = False
            
            for i, row in recent_bars.iterrows():
                # Check if high touched or broke resistance
                if row['high'] >= resistance_level - self.min_distance_from_sr:
                    touched_resistance = True
                
                # Check if close is now below resistance (rejection)
                if touched_resistance and row['close'] < resistance_level - self.min_distance_from_sr:
                    rejected_down = True
            
            # Additional confirmation: recent bar should be bearish
            last_bar = recent_bars.iloc[-1]
            is_bearish_bar = last_bar['close'] < last_bar['open']
            
            return touched_resistance and rejected_down and is_bearish_bar
            
        except Exception as e:
            logger.error(f"Error checking resistance rejection: {e}")
            return False
    
    def calculate_sr_based_stops(self, symbol: str, entry_price: float, 
                                signal: str, sr_level: float) -> Tuple[float, float]:
        """Calculate stop loss and take profit based on S/R levels"""
        try:
            symbol_info = mt5.symbol_info(symbol)
            pip_value = symbol_info.point * 10
            
            if signal == 'BUY':
                # Stop loss below the support level
                stop_loss = sr_level - (self.stop_loss_pips * pip_value)
                # Take profit based on risk:reward ratio
                risk_pips = abs(entry_price - stop_loss) / pip_value
                take_profit = entry_price + (risk_pips * self.take_profit_ratio * pip_value)
                
            else: # SELL
                # Stop loss above the resistance level
                stop_loss = sr_level + (self.stop_loss_pips * pip_value)
                # Take profit based on risk:reward ratio
                risk_pips = abs(stop_loss - entry_price) / pip_value
                take_profit = entry_price - (risk_pips * self.take_profit_ratio * pip_value)
            
            return stop_loss, take_profit
            
        except Exception as e:
            logger.error(f"Error calculating S/R stops: {e}")
            return entry_price, entry_price
    
    def is_trading_hours(self) -> bool:
        """Check if within trading hours"""
        current_hour = datetime.now().hour
        return self.trading_start_hour <= current_hour <= self.trading_end_hour
    
    def check_spread_filter(self, symbol: str) -> bool:
        """Check spread filter"""
        try:
            tick = mt5.symbol_info_tick(symbol)
            symbol_info = mt5.symbol_info(symbol)
            if not tick or not symbol_info:
                return False
            
            spread_points = (tick.ask - tick.bid) / symbol_info.point
            return spread_points <= self.max_spread_points
        except:
            return False
    
    def get_current_positions(self) -> List:
        """Get current positions for this robot"""
        positions = mt5.positions_get()
        return [pos for pos in positions if pos.magic == self.magic_number] if positions else []
    
    def check_position_limits(self, symbol: str) -> bool:
        """Check position limits"""
        positions = self.get_current_positions()
        
        if len(positions) >= self.max_total_positions:
            return False
        
        symbol_positions = [pos for pos in positions if pos.symbol == symbol]
        return len(symbol_positions) < self.max_positions_per_symbol
    
    def calculate_position_size(self, symbol: str, entry_price: float, stop_loss: float) -> float:
        """Calculate position size based on risk"""
        try:
            account_info = mt5.account_info()
            if not account_info:
                return 0.0
            
            risk_amount = account_info.balance * self.risk_per_trade
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
    
    def place_order(self, symbol: str, order_type: int, volume: float, price: float,
                   stop_loss: float = None, take_profit: float = None) -> bool:
        """Place trading order"""
        try:
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": order_type,
                "price": price,
                "sl": stop_loss,
                "tp": take_profit,
                "magic": self.magic_number,
                "comment": f"{self.strategy_name} S/R Robot",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            result = mt5.order_send(request)
            return result.retcode == mt5.TRADE_RETCODE_DONE
            
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return False
    
    def execute_sr_trade(self, symbol: str, signal: str, sr_level: float) -> bool:
        """Execute trade based on support/resistance signal"""
        try:
            if (self.trading_stopped_due_to_drawdown or 
                not self.is_trading_hours() or 
                not self.check_spread_filter(symbol) or 
                not self.check_position_limits(symbol)):
                return False
            
            tick = mt5.symbol_info_tick(symbol)
            if not tick:
                return False
            
            if signal == 'BUY':
                entry_price = tick.ask
                stop_loss, take_profit = self.calculate_sr_based_stops(symbol, entry_price, signal, sr_level)
                order_type = mt5.ORDER_TYPE_BUY
                
            elif signal == 'SELL':
                entry_price = tick.bid
                stop_loss, take_profit = self.calculate_sr_based_stops(symbol, entry_price, signal, sr_level)
                order_type = mt5.ORDER_TYPE_SELL
            else:
                return False
            
            volume = self.calculate_position_size(symbol, entry_price, stop_loss)
            
            if volume > 0:
                success = self.place_order(symbol, order_type, volume, entry_price, stop_loss, take_profit)
                
                if success:
                    risk_pips = abs(entry_price - stop_loss) / mt5.symbol_info(symbol).point / 10
                    reward_pips = abs(take_profit - entry_price) / mt5.symbol_info(symbol).point / 10
                    rr_ratio = reward_pips / risk_pips if risk_pips > 0 else 0
                    
                    logger.info(f"✅ {signal} {symbol} @ S/R {sr_level:.5f}")
                    logger.info(f" 📊 Entry: {entry_price:.5f} | SL: {stop_loss:.5f} | TP: {take_profit:.5f}")
                    logger.info(f" 💰 Volume: {volume:.2f} | Risk: {risk_pips:.1f} pips | R:R = 1:{rr_ratio:.1f}")
                
                return success
            
            return False
            
        except Exception as e:
            logger.error(f"Error executing S/R trade: {e}")
            return False
    
    def record_equity_snapshot(self):
        """Record equity for drawdown monitoring"""
        try:
            account_info = mt5.account_info()
            if not account_info:
                return
                
            current_equity = account_info.equity
            current_time = datetime.now()
            
            self.equity_history.append({
                'timestamp': current_time,
                'equity': current_equity,
                'balance': account_info.balance,
                'profit': account_info.profit
            })
            
            if current_equity > self.peak_equity:
                self.peak_equity = current_equity
            
            # Clean old records
            cutoff_time = current_time - timedelta(hours=self.drawdown_period_hours)
            self.equity_history = [r for r in self.equity_history if r['timestamp'] > cutoff_time]
            
        except Exception as e:
            logger.error(f"Error recording equity: {e}")
    
    def check_drawdown_protection(self) -> bool:
        """Check if drawdown protection should trigger"""
        try:
            if not self.equity_history:
                return False
            
            account_info = mt5.account_info()
            if not account_info:
                return False
                
            current_equity = account_info.equity
            period_peak = max(r['equity'] for r in self.equity_history)
            
            drawdown_from_peak = ((period_peak - current_equity) / period_peak) * 100
            drawdown_from_initial = ((self.initial_equity - current_equity) / self.initial_equity) * 100
            
            max_drawdown = max(drawdown_from_peak, drawdown_from_initial)
            
            logger.info(f"💹 Equity: ${current_equity:.2f} | DD: {max_drawdown:.1f}% (Max: {self.max_drawdown_percent}%)")
            
            if max_drawdown >= self.max_drawdown_percent:
                logger.critical(f"🚨 DRAWDOWN PROTECTION TRIGGERED: {max_drawdown:.1f}% >= {self.max_drawdown_percent}%")
                self.trading_stopped_due_to_drawdown = True
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking drawdown: {e}")
            return False
    
    def close_all_positions(self) -> bool:
        """Emergency close all positions"""
        try:
            positions = self.get_current_positions()
            if not positions:
                return True
            
            logger.info(f"🚨 Emergency closing {len(positions)} positions...")
            
            for position in positions:
                try:
                    tick = mt5.symbol_info_tick(position.symbol)
                    if not tick:
                        continue
                    
                    if position.type == mt5.POSITION_TYPE_BUY:
                        order_type = mt5.ORDER_TYPE_SELL
                        price = tick.bid
                    else:
                        order_type = mt5.ORDER_TYPE_BUY
                        price = tick.ask
                    
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
                        logger.info(f"✅ Closed {position.symbol}: P&L = ${position.profit:.2f}")
                    else:
                        logger.error(f"❌ Failed to close {position.symbol}")
                        
                except Exception as e:
                    logger.error(f"Error closing position {position.symbol}: {e}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error in emergency close: {e}")
            return False
    
    def monitor_positions(self):
        """Monitor existing positions"""
        try:
            positions = self.get_current_positions()
            
            if positions:
                total_profit = sum(pos.profit for pos in positions)
                logger.info(f"📊 Active Positions: {len(positions)} | Total P&L: ${total_profit:.2f}")
                
                for pos in positions:
                    pos_type = "BUY" if pos.type == mt5.POSITION_TYPE_BUY else "SELL"
                    logger.info(f" {pos.symbol}: {pos_type} | Vol: {pos.volume} | P&L: ${pos.profit:.2f}")
            
        except Exception as e:
            logger.error(f"Error monitoring positions: {e}")
    
    def check_and_handle_drawdown(self) -> bool:
        """Check drawdown and handle emergency stop"""
        self.record_equity_snapshot()
        
        if self.check_drawdown_protection():
            self.close_all_positions()
            self.is_running = False
            logger.critical("🛑 TRADING STOPPED - DRAWDOWN PROTECTION ACTIVATED")
            return False
        
        return True
    
    def display_sr_levels(self, symbol: str):
        """Display current support and resistance levels for analysis"""
        try:
            df = self.get_market_data(symbol, self.analysis_timeframe, self.analysis_period)
            if df is None:
                return
            
            sr_levels = self.find_support_resistance_levels(df, symbol)
            
            logger.info(f"📈 {symbol} Support & Resistance Levels ({self.analysis_timeframe} TF):")
            
            if sr_levels['resistance']:
                logger.info("🔴 RESISTANCE LEVELS:")
                for i, level in enumerate(sr_levels['resistance'][:5], 1):
                    logger.info(f" {i}. {level['price']:.5f} | Strength: {level['strength']} | "
                              f"Quality: {level['quality_score']:.1f} | Age: {level['last_touch_age']} bars")
            
            if sr_levels['support']:
                logger.info("🟢 SUPPORT LEVELS:")
                for i, level in enumerate(sr_levels['support'][:5], 1):
                    logger.info(f" {i}. {level['price']:.5f} | Strength: {level['strength']} | "
                              f"Quality: {level['quality_score']:.1f} | Age: {level['last_touch_age']} bars")
            
            if not sr_levels['resistance'] and not sr_levels['support']:
                logger.info(" No significant S/R levels found")
                
        except Exception as e:
            logger.error(f"Error displaying S/R levels: {e}")
    
    def run_sr_trading_session(self, duration_minutes: int = 240, show_levels: bool = False):
        """
        Run support & resistance trading session
        
        Args:
            duration_minutes: Session duration in minutes
            show_levels: Whether to display S/R levels at start
        """
        logger.info(f"🎯 Starting Support & Resistance Trading Session")
        logger.info(f"📊 Strategy: {self.strategy_name}")
        logger.info(f"📝 {self.description}")
        logger.info(f"⏱️ Duration: {duration_minutes} minutes")
        logger.info(f"🎪 Symbols: {self.symbols}")
        logger.info(f"📈 Entry TF: {self.timeframe} | Analysis TF: {self.analysis_timeframe}")
        
        # Display S/R levels if requested
        if show_levels:
            for symbol in self.symbols:
                self.display_sr_levels(symbol)
        
        self.is_running = True
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=duration_minutes)
        cycle_count = 0
        
        try:
            while self.is_running and datetime.now() < end_time:
                cycle_count += 1
                logger.info(f"\n")
                logger.info(f"--- S/R Trading Cycle #{cycle_count} ---")
                if show_levels:
                    for symbol in self.symbols:
                        self.display_sr_levels(symbol)
                
                logger.info(f"------------------------------")
                # Priority: Check drawdown protection
                if not self.check_and_handle_drawdown():
                    break
                
                # Monitor existing positions
                self.monitor_positions()
                
                # Process each symbol
                if not self.trading_stopped_due_to_drawdown:
                    for symbol in self.symbols:
                        try:
                            # Check for S/R entry signals
                            signal = self.check_support_resistance_entry(symbol)
                            logger.info(f"📝 {symbol}: current signal is {signal}")
                            
                            if signal != 'HOLD':
                                logger.info(f"🎯 {symbol}: {signal} signal detected")
                                
                                # Find the relevant S/R level for entry
                                df_higher = self.get_market_data(symbol, self.analysis_timeframe, self.analysis_period)
                                if df_higher is not None:
                                    sr_levels = self.find_support_resistance_levels(df_higher, symbol)
                                    
                                    # Get current price to find nearest S/R level
                                    tick = mt5.symbol_info_tick(symbol)
                                    if tick:
                                        current_price = (tick.bid + tick.ask) / 2
                                        symbol_info = mt5.symbol_info(symbol)
                                        if symbol_info:
                                            pip_value = symbol_info.point * 10
                                            
                                            nearest_level = None
                                            min_distance = float('inf')
                                            
                                            # Find nearest relevant S/R level
                                            levels_to_check = sr_levels['support'] if signal == 'BUY' else sr_levels['resistance']
                                            
                                            for level in levels_to_check:
                                                distance = abs(current_price - level['price']) / pip_value
                                                if distance <= self.sr_proximity_pips and distance < min_distance:
                                                    min_distance = distance
                                                    nearest_level = level['price']
                                            
                                            if nearest_level:
                                                self.execute_sr_trade(symbol, signal, nearest_level)
                            else:
                                logger.debug(f"{symbol}: No S/R signal (HOLD)")
                                
                        except Exception as e:
                            logger.error(f"Error processing {symbol}: {e}")
                else:
                    logger.warning("⚠️ New trades disabled due to drawdown protection")
                
                # Wait before next cycle
                if self.is_running:
                    time.sleep(self.cycle_delay_seconds)
                
        except KeyboardInterrupt:
            logger.info("Trading session stopped by user")
        except Exception as e:
            logger.error(f"Trading session error: {e}")
        finally:
            # Final summary
            if self.trading_stopped_due_to_drawdown:
                account_info = mt5.account_info()
                if account_info:
                    final_equity = account_info.equity
                    total_loss = self.initial_equity - final_equity
                    loss_percentage = (total_loss / self.initial_equity) * 100
                    
                    logger.critical("=== DRAWDOWN PROTECTION SUMMARY ===")
                    logger.critical(f"Initial Equity: ${self.initial_equity:.2f}")
                    logger.critical(f"Final Equity: ${final_equity:.2f}")
                    logger.critical(f"Total Loss: ${total_loss:.2f} ({loss_percentage:.2f}%)")
            
            self.is_running = False
            logger.info(f"Support & Resistance trading session completed ({cycle_count} cycles)")
    
    def get_trading_statistics(self) -> Dict:
        """Get comprehensive trading statistics"""
        try:
            # Get deals history
            deals = mt5.history_deals_get(
                datetime.now() - timedelta(days=30),
                datetime.now()
            )
            
            if not deals:
                return {"message": "No trading history found"}
            
            robot_deals = [deal for deal in deals if deal.magic == self.magic_number]
            
            if not robot_deals:
                return {"message": f"No deals found for magic number {self.magic_number}"}
            
            total_profit = sum(deal.profit for deal in robot_deals)
            total_trades = len(robot_deals)
            winning_trades = len([deal for deal in robot_deals if deal.profit > 0])
            losing_trades = total_trades - winning_trades
            win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
            
            account_info = mt5.account_info()
            current_balance = account_info.balance if account_info else 0
            
            # Calculate additional metrics
            if winning_trades > 0:
                avg_win = sum(deal.profit for deal in robot_deals if deal.profit > 0) / winning_trades
            else:
                avg_win = 0
                
            if losing_trades > 0:
                avg_loss = sum(deal.profit for deal in robot_deals if deal.profit < 0) / losing_trades
            else:
                avg_loss = 0
            
            profit_factor = abs(avg_win * winning_trades / (avg_loss * losing_trades)) if avg_loss != 0 else 0
            
            return {
                "strategy": self.strategy_name,
                "total_trades": total_trades,
                "winning_trades": winning_trades,
                "losing_trades": losing_trades,
                "win_rate": round(win_rate, 2),
                "total_profit": round(total_profit, 2),
                "average_win": round(avg_win, 2),
                "average_loss": round(avg_loss, 2),
                "profit_factor": round(profit_factor, 2),
                "current_balance": round(current_balance, 2),
                "magic_number": self.magic_number
            }
            
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {"error": str(e)}
    
    def shutdown(self):
        """Shutdown robot and MT5 connection"""
        self.is_running = False
        mt5.shutdown()
        logger.info("🔌 MT5 connection closed - Robot shutdown complete")

# Example usage and configuration showcase
if __name__ == "__main__":
    """
  
    print("🎯 Support & Resistance Trading Robot Configurations")
    print("=" * 60)
    
    # Example 1: Swing Trading with H1 entries on H4 S/R levels
    print("\n1️⃣ SWING TRADING CONFIGURATION")
    print("Entry: H1 | Analysis: H4 | Conservative approach")
    
    swing_robot = SupportResistanceRobot(TradingStrategy.SUPPORT_RESISTANCE_SWING)
    
    # Example 2: 15M Scalping with entries on H1 S/R levels  
    print("\n2️⃣ 15-MINUTE SCALPING CONFIGURATION")
    print("Entry: M15 | Analysis: H1 | Balanced scalping")
    
    scalp_15m_robot = SupportResistanceRobot(TradingStrategy.SUPPORT_RESISTANCE_SCALP_15M)
    
    # Example 3: 5M Scalping with entries on M30 S/R levels
    print("\n3️⃣ 5-MINUTE SCALPING CONFIGURATION") 
    print("Entry: M5 | Analysis: M30 | Active scalping")
    
    scalp_5m_robot = SupportResistanceRobot(TradingStrategy.SUPPORT_RESISTANCE_SCALP_5M)
    
    # Example 4: 1M Ultra-fast scalping
    print("\n4️⃣ 1-MINUTE ULTRA SCALPING CONFIGURATION")
    print("Entry: M1 | Analysis: M15 | High-frequency")
    
    scalp_1m_robot = SupportResistanceRobot(TradingStrategy.SUPPORT_RESISTANCE_SCALP_1M)
    
    # Example 5: Custom configuration
    print("\n5️⃣ CUSTOM CONFIGURATION EXAMPLE")
    
    custom_config = {
        'risk_per_trade': 0.001,
        'max_drawdown_percent': 10.0,
        'symbols': ['XAUUSD'],
        'sr_touch_threshold': 2,
        'stop_loss_pips': 15,
        'take_profit_ratio': 2.5,
        'strategy_name': 'My Custom S/R Strategy'
    }
    
    custom_robot = SupportResistanceRobot(TradingStrategy.CUSTOM, custom_config)
    """

    
    # Uncomment to run a demo session:
    # swing_robot.run_sr_trading_session(duration_minutes=60, show_levels=True)
    
    # Example of running with statistics
    scalp_robot = SupportResistanceRobot(TradingStrategy.SUPPORT_RESISTANCE_SCALP_1M)
    try:
        # Run trading session
        scalp_robot.run_sr_trading_session(duration_minutes=480, show_levels=True)
        
        # Get and display statistics
        stats = scalp_robot.get_trading_statistics()
        print("\n📊 TRADING STATISTICS:")
        for key, value in stats.items():
            print(f"{key}: {value}")
            
    except Exception as e:
        logger.error(f"Example execution error: {e}")
    finally:
        scalp_robot.shutdown()

    
