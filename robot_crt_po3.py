# MT5 CRT Power of 3 - Usage Examples and Additional Scripts

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import json

# Example 1: Quick Setup and Single Analysis
def quick_analysis_example():
    """Example of running a single analysis without automated trading"""
    
    # Initialize trader with demo account
    trader = MT5_CRT_PowerOfThree(
        login=123456789,
        password="demo_password",
        server="Demo-Server",
        symbol="EURUSD"
    )
    
    # Get historical data and analyze
    df = trader.get_historical_data(200)
    analyzed_df = trader.detect_crt_signals(df)
    
    # Find recent signals
    signals = analyzed_df[analyzed_df['crt_signal'].notna()]
    print(f"Found {len(signals)} signals in recent data")
    
    # Show latest signal details
    if len(signals) > 0:
        latest = signals.iloc[-1]
        print(f"Latest Signal: {latest['crt_signal']}")
        print(f"Entry Price: {latest['entry_price']:.5f}")
        print(f"Stop Loss: {latest['stop_loss']:.5f}")
        print(f"Take Profit: {latest['take_profit']:.5f}")
        print(f"Signal Strength: {latest['signal_strength']}")
    
    trader.disconnect_mt5()

# Example 2: Multi-Symbol Analysis
class MultiSymbolCRTAnalyzer:
    """Analyze CRT Po3 signals across multiple forex pairs"""
    
    def __init__(self, login: int, password: str, server: str):
        self.login = login
        self.password = password
        self.server = server
        self.symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD']
        
    def analyze_all_symbols(self) -> pd.DataFrame:
        """Analyze CRT signals for all configured symbols"""
        
        results = []
        
        # Connect to MT5
        if not mt5.initialize():
            print("MT5 initialization failed")
            return pd.DataFrame()
            
        if not mt5.login(self.login, password=self.password, server=self.server):
            print("MT5 login failed")
            return pd.DataFrame()
        
        for symbol in self.symbols:
            try:
                # Initialize trader for this symbol
                trader = MT5_CRT_PowerOfThree(
                    self.login, self.password, self.server, symbol
                )
                
                # Get data and analyze
                df = trader.get_historical_data(100)
                if df is not None and len(df) > 50:
                    analyzed_df = trader.detect_crt_signals(df)
                    
                    # Get latest signal info
                    latest = analyzed_df.iloc[-1]
                    
                    result = {
                        'symbol': symbol,
                        'current_price': latest['close'],
                        'signal': latest['crt_signal'] if pd.notna(latest['crt_signal']) else 'NONE',
                        'signal_strength': latest['signal_strength'],
                        'trend': latest['trend'],
                        'po3_phase': latest['po3_phase'],
                        'bos': latest['bos'],
                        'timestamp': latest.name
                    }
                    
                    results.append(result)
                    
            except Exception as e:
                print(f"Error analyzing {symbol}: {e}")
                
        mt5.shutdown()
        return pd.DataFrame(results)

# Example 3: Signal Alert System
class CRTSignalAlerter:
    """Send alerts when new CRT signals are detected"""
    
    def __init__(self, trader: MT5_CRT_PowerOfThree):
        self.trader = trader
        self.last_signal_time = None
        self.alert_methods = ['console', 'file']  # Can add 'email', 'telegram' etc.
        
    def check_for_new_signals(self):
        """Check for new signals and send alerts"""
        try:
            df = self.trader.get_historical_data(50)
            analyzed_df = self.trader.detect_crt_signals(df)
            
            # Get latest signal
            latest_signals = analyzed_df[analyzed_df['crt_signal'].notna()]
            
            if len(latest_signals) > 0:
                latest_signal = latest_signals.iloc[-1]
                signal_time = latest_signal.name
                
                # Check if this is a new signal
                if self.last_signal_time is None or signal_time > self.last_signal_time:
                    self.send_alert(latest_signal)
                    self.last_signal_time = signal_time
                    
        except Exception as e:
            print(f"Error checking for signals: {e}")
    
    def send_alert(self, signal_data):
        """Send alert using configured methods"""
        
        alert_message = f"""
🚨 CRT Po3 SIGNAL ALERT 🚨
Symbol: {self.trader.symbol}
Signal: {signal_data['crt_signal']}
Strength: {signal_data['signal_strength']}/7
Entry Price: {signal_data['entry_price']:.5f}
Stop Loss: {signal_data['stop_loss']:.5f}
Take Profit: {signal_data['take_profit']:.5f}
Phase: {signal_data['po3_phase']}
Time: {signal_data.name}
        """
        
        if 'console' in self.alert_methods:
            print(alert_message)
            
        if 'file' in self.alert_methods:
            with open('crt_alerts.txt', 'a') as f:
                f.write(f"{datetime.now()}: {alert_message}\n")

# Example 4: Performance Analytics
class CRTPerformanceAnalyzer:
    """Analyze CRT Po3 strategy performance in detail"""
    
    def __init__(self, trader: MT5_CRT_PowerOfThree):
        self.trader = trader
        
    def analyze_signal_accuracy(self, days_back: int = 30) -> dict:
        """Analyze historical signal accuracy"""
        
        try:
            # Get historical data
            bars_needed = days_back * 24  # Assuming H1 timeframe
            df = self.trader.get_historical_data(bars_needed)
            analyzed_df = self.trader.detect_crt_signals(df)
            
            # Find all historical signals
            signals = analyzed_df[analyzed_df['crt_signal'].notna()].copy()
            
            if len(signals) == 0:
                return {'error': 'No signals found in historical data'}
            
            # Analyze each signal's outcome
            results = []
            
            for idx, signal in signals.iterrows():
                # Look ahead to see if TP or SL was hit first
                future_data = analyzed_df[analyzed_df.index > idx].head(100)  # Look ahead 100 bars
                
                if len(future_data) == 0:
                    continue
                    
                entry_price = signal['entry_price']
                stop_loss = signal['stop_loss']
                take_profit = signal['take_profit']
                
                outcome = 'TIMEOUT'  # Default if neither TP nor SL hit
                
                for future_idx, future_row in future_data.iterrows():
                    if signal['crt_signal'] == 'BUY':
                        if future_row['low'] <= stop_loss:
                            outcome = 'LOSS'
                            break
                        elif future_row['high'] >= take_profit:
                            outcome = 'WIN'
                            break
                    else:  # SELL
                        if future_row['high'] >= stop_loss:
                            outcome = 'LOSS'
                            break
                        elif future_row['low'] <= take_profit:
                            outcome = 'WIN'
                            break
                
                results.append({
                    'signal_time': idx,
                    'signal': signal['crt_signal'],
                    'strength': signal['signal_strength'],
                    'phase': signal['po3_phase'],
                    'outcome': outcome
                })
            
            # Calculate statistics
            results_df = pd.DataFrame(results)
            
            if len(results_df) == 0:
                return {'error': 'No complete signal results found'}
            
            total_signals = len(results_df)
            wins = len(results_df[results_df['outcome'] == 'WIN'])
            losses = len(results_df[results_df['outcome'] == 'LOSS'])
            timeouts = len(results_df[results_df['outcome'] == 'TIMEOUT'])
            
            # Analyze by signal strength
            strength_analysis = results_df.groupby('signal_strength')['outcome'].apply(
                lambda x: (x == 'WIN').sum() / len(x) * 100 if len(x) > 0 else 0
            ).to_dict()
            
            # Analyze by Po3 phase
            phase_analysis = results_df.groupby('phase')['outcome'].apply(
                lambda x: (x == 'WIN').sum() / len(x) * 100 if len(x) > 0 else 0
            ).to_dict()
            
            return {
                'total_signals': total_signals,
                'wins': wins,
                'losses': losses,
                'timeouts': timeouts,
                'win_rate': (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0,
                'strength_analysis': strength_analysis,
                'phase_analysis': phase_analysis,
                'detailed_results': results_df
            }
            
        except Exception as e:
            return {'error': f'Analysis failed: {str(e)}'}
    
    def plot_performance_charts(self, analysis_results: dict):
        """Create performance visualization charts"""
        
        if 'error' in analysis_results:
            print(f"Cannot plot: {analysis_results['error']}")
            return
            
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Chart 1: Win/Loss Distribution
        outcomes = [analysis_results['wins'], analysis_results['losses'], analysis_results['timeouts']]
        labels = ['Wins', 'Losses', 'Timeouts']
        colors = ['green', 'red', 'orange']
        
        ax1.pie(outcomes, labels=labels, colors=colors, autopct='%1.1f%%')
        ax1.set_title('Signal Outcomes Distribution')
        
        # Chart 2: Win Rate by Signal Strength
        strength_data = analysis_results['strength_analysis']
        if strength_data:
            strengths = list(strength_data.keys())
            win_rates = list(strength_data.values())
            
            ax2.bar(strengths, win_rates, color='blue', alpha=0.7)
            ax2.set_xlabel('Signal Strength')
            ax2.set_ylabel('Win Rate (%)')
            ax2.set_title('Win Rate by Signal Strength')
            ax2.grid(True, alpha=0.3)
        
        # Chart 3: Win Rate by Po3 Phase
        phase_data = analysis_results['phase_analysis']
        if phase_data:
            phases = list(phase_data.keys())
            phase_win_rates = list(phase_data.values())
            
            ax3.bar(phases, phase_win_rates, color='orange', alpha=0.7)
            ax3.set_xlabel('Po3 Phase')
            ax3.set_ylabel('Win Rate (%)')
            ax3.set_title('Win Rate by Po3 Phase')
            ax3.tick_params(axis='x', rotation=45)
            ax3.grid(True, alpha=0.3)
        
        # Chart 4: Signals Over Time
        results_df = analysis_results['detailed_results']
        if len(results_df) > 0:
            # Group by date and count signals
            results_df['date'] = pd.to_datetime(results_df['signal_time']).dt.date
            daily_signals = results_df.groupby('date').size()
            
            ax4.plot(daily_signals.index, daily_signals.values, marker='o', color='purple')
            ax4.set_xlabel('Date')
            ax4.set_ylabel('Number of Signals')
            ax4.set_title('Daily Signal Frequency')
            ax4.tick_params(axis='x', rotation=45)
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

# Example 5: Configuration Management
class CRTConfig:
    """Manage CRT Po3 system configuration"""
    
    def __init__(self, config_file: str = 'crt_config.json'):
        self.config_file = config_file
        self.default_config = {
            'mt5': {
                'login': 123456789,
                'password': 'demo_password',
                'server': 'Demo-Server'
            },
            'trading': {
                'symbols': ['EURUSD', 'GBPUSD', 'USDJPY'],
                'timeframe': 'H1',
                'risk_percent': 2.0,
                'max_spread': 20,
                'min_signal_strength': 4,
                'analysis_interval': 300
            },
            'po3_sessions': {
                'london': {'start': 2, 'end': 6},
                'new_york': {'start': 13, 'end': 17},
                'asia': {'start': 21, 'end': 1}
            },
            'alerts': {
                'enabled': True,
                'methods': ['console', 'file'],
                'min_strength': 5
            }
        }
        
    def load_config(self) -> dict:
        """Load configuration from file"""
        try:
            with open(self.config_file, 'r') as f:
                config = json.load(f)
            return config
        except FileNotFoundError:
            print(f"Config file {self.config_file} not found, creating default...")
            self.save_config(self.default_config)
            return self.default_config
        except Exception as e:
            print(f"Error loading config: {e}")
            return self.default_config
    
    def save_config(self, config: dict):
        """Save configuration to file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
            print(f"Configuration saved to {self.config_file}")
        except Exception as e:
            print(f"Error saving config: {e}")

# Example 6: Complete Trading Session
def complete_trading_session_example():
    """Example of a complete trading session with all features"""
    
    print("=== CRT Po3 Complete Trading Session ===\n")
    
    # Load configuration
    config_manager = CRTConfig()
    config = config_manager.load_config()
    
    # Initialize multi-symbol analyzer
    analyzer = MultiSymbolCRTAnalyzer(
        login=config['mt5']['login'],
        password=config['mt5']['password'],
        server=config['mt5']['server']
    )
    
    print("1. Multi-Symbol Analysis:")
    print("-" * 30)
    
    # Analyze all symbols
    results = analyzer.analyze_all_symbols()
    
    if not results.empty:
        # Display results
        for _, row in results.iterrows():
            if row['signal'] != 'NONE':
                print(f"{row['symbol']}: {row['signal']} (Strength: {row['signal_strength']}) - {row['po3_phase']}")
            else:
                print(f"{row['symbol']}: No signal - {row['trend']} trend")
    
    print("\n2. Performance Analysis:")
    print("-" * 30)
    
    # Select best signal for detailed analysis
    if not results.empty:
        best_signal = results[results['signal_strength'] == results['signal_strength'].max()].iloc[0]
        symbol = best_signal['symbol']
        
        print(f"Analyzing {symbol} (highest signal strength: {best_signal['signal_strength']})")
        
        # Initialize trader for detailed analysis
        trader = MT5_CRT_PowerOfThree(
            login=config['mt5']['login'],
            password=config['mt5']['password'],
            server=config['mt5']['server'],
            symbol=symbol
        )
        
        # Performance analysis
        perf_analyzer = CRTPerformanceAnalyzer(trader)
        perf_results = perf_analyzer.analyze_signal_accuracy(days_back=30)
        
        if 'error' not in perf_results:
            print(f"Historical Win Rate: {perf_results['win_rate']:.1f}%")
            print(f"Total Signals Analyzed: {perf_results['total_signals']}")
            print(f"Best Signal Strength: {max(perf_results['strength_analysis'], key=perf_results['strength_analysis'].get)}")
            
            # Plot performance charts
            perf_analyzer.plot_performance_charts(perf_results)
        
        trader.disconnect_mt5()
    
    print("\n3. Live Monitoring Setup:")
    print("-" * 30)
    print("To start live monitoring, run:")
    print("python crt_mt5_system.py")
    print("Then select option 1 for automated trading")

# Run examples
if __name__ == "__main__":
    print("CRT Po3 MT5 System - Examples")
    print("1. Quick Analysis Example")
    print("2. Multi-Symbol Analysis") 
    print("3. Complete Trading Session")
    print("4. Exit")
    
    choice = input("Select example (1-4): ").strip()
    
    if choice == '1':
        quick_analysis_example()
    elif choice == '2':
        analyzer = MultiSymbolCRTAnalyzer(123456789, "demo_password", "Demo-Server")
        results = analyzer.analyze_all_symbols()
        print(results)
    elif choice == '3':
        complete_trading_session_example()
    else:
        print("Exiting...")