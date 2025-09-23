import pandas as pd
import numpy as np
import yfinance as yf
import talib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import certifi, requests
import warnings
import re
import time
from bs4 import BeautifulSoup
import json
import urllib3
warnings.filterwarnings('ignore')
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class IndonesianStockScreener:
    def __init__(self):
        # Indonesian stock symbols typically end with .JK for Jakarta Stock Exchange
        self.idx_symbols = []
        self.broker_data = {}
        self.screened_stocks = []
        
    def fetch_all_idx_stocks_from_idx_website(self):
        """Fetch all Indonesian stocks from IDX official website"""
        try:
            # Try to fetch from IDX official stock list
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            # IDX provides stock list in JSON format
            url = "https://www.idx.co.id/umbraco/surface/ListedCompany/GetCompany"
            
            response = requests.post(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    symbols = []
                    
                    # Extract symbols from JSON response
                    if 'data' in data:
                        for item in data['data']:
                            if 'Kode' in item:
                                symbols.append(item['Kode'])
                    elif isinstance(data, list):
                        for item in data:
                            if isinstance(item, dict) and 'Kode' in item:
                                symbols.append(item['Kode'])
                    
                    if symbols:
                        print(f"Successfully fetched {len(symbols)} stocks from IDX website")
                        return symbols
                        
                except json.JSONDecodeError:
                    print("Failed to parse JSON from IDX website")
                    
        except Exception as e:
            print(f"Error fetching from IDX website: {e}")
        
        return None
    
    def get_comprehensive_stock_symbols(self):
        """Get comprehensive list using predefined major stocks + fetched symbols"""
        # Start with major known Indonesian stocks
        major_idx_stocks = [
            # Banking
            'BBCA', 'BBRI', 'BMRI', 'BBNI', 'BDMN', 'BRIS', 'NISP', 'MEGA', 'PNBN', 'BJBR',
            
            # Telecommunications
            'TLKM', 'ISAT', 'EXCL', 'FREN',
            
            # Consumer Goods
            'UNVR', 'ICBP', 'INDF', 'KLBF', 'KAEF', 'TSPC', 'MYOR', 'SIDO', 'PYFA', 'TCID',
            'DLTA', 'SKBM', 'MLBI', 'TBLA', 'PSDN', 'ULTJ', 'CEKA', 'ALTO', 'JPFA', 'STTP',
            
            # Automotive
            'ASII', 'AUTO', 'IMAS', 'SMSM', 'GDYR', 'MASA', 'NIPS', 'PRAS', 'BOLT', 'LPIN',
            
            # Cement & Construction
            'SMGR', 'INTP', 'SMBR', 'WTON', 'ADHI', 'WSKT', 'WIKA', 'PTPP', 'DGIK', 'TOTL',
            
            # Mining & Energy
            'ADRO', 'PTBA', 'ITMG', 'HRUM', 'TOBA', 'DEWA', 'COAL', 'GEMS', 'MYOH', 'BUMI',
            'AKRA', 'PTRO', 'MEDC', 'RUIS', 'ELSA',
            
            # Tobacco
            'HMSP', 'GGRM', 'RMBA', 'WIIM',
            
            # Property
            'BSDE', 'LPKR', 'ASRI', 'PWON', 'APLN', 'SMRA', 'CTRA', 'DILD', 'BEST', 'GPRA',
            'KIJA', 'MTSM', 'MTLA', 'BKSL', 'COWL', 'ELTY', 'FMII', 'GMTD', 'JRPT', 'KPIG',
            'LCGP', 'MDLN', 'MKPI', 'NIRO', 'OMRE', 'PPRO', 'PUDP', 'RBMS', 'RDTX', 'RODA',
            
            # Finance & Investment
            'BBTN', 'BTPN', 'MAYA', 'BFIN', 'ADMF', 'BBLD', 'BBYB', 'BCIC', 'BEKS', 'BINA',
            'BKSW', 'BNBA', 'BNII', 'BNLI', 'BSIM', 'BSWD', 'BTPS', 'BVIC', 'DNAR', 'INPC',
            'MAYA', 'MCOR', 'PNBS', 'SDRA',
            
            # Technology & Media
            'GOTO', 'BUKA', 'MTEL', 'VIVA', 'TMPO', 'FILM', 'SCMA', 'ABBA', 'BMTR', 'CTBN',
            'EMTK', 'FAST', 'FORU', 'GSMF', 'HART', 'HMSP', 'ITIC', 'KOBX', 'LINK', 'LPGI',
            'MAPI', 'META', 'MIDI', 'MPPA', 'NETWORK', 'OASA', 'PANS', 'RANC', 'SKYB', 'SOCI',
            'TRIO', 'WOOD',
            
            # Plantation
            'AALI', 'LSIP', 'UNSP', 'SIMP', 'SGRO', 'SSMS', 'TBMS', 'ANJT', 'BWPT', 'CPRO',
            'DSNG', 'GOLL', 'JAWA', 'MAGP', 'PALM', 'PPKS', 'SAMF', 'SMAR', 'TUAH',
            
            # Transportation & Logistics
            'BIRD', 'CMPP', 'GIAA', 'IPCM', 'JSMR', 'KARW', 'LRNA', 'MIRA', 'NELY', 'SAFE',
            'SMDR', 'SOCI', 'TKIM', 'TOPS', 'WEHA', 'WINS', 'ZBRA',
            
            # Healthcare & Pharmaceuticals
            'KAEF', 'KLBF', 'MERK', 'PYFA', 'SIDO', 'SQMI', 'TSPC', 'DVLA', 'INAF', 'KBLF',
            'MDRN', 'PRDA', 'SAIP', 'SILO', 'SOHO',
            
            # Retail & Trading
            'ACES', 'AMRT', 'CSAP', 'ECII', 'ERAA', 'HERO', 'LPPF', 'MAPI', 'MIDI', 'MKNT',
            'MPPA', 'RALS', 'RANC', 'SKYB', 'SORO', 'SPORT', 'TRIO'
        ]
        
        # Remove duplicates
        major_idx_stocks = list(set(major_idx_stocks))
        
        print(f"Starting with {len(major_idx_stocks)} major Indonesian stocks...")
        
        # Try to fetch additional symbols from various sources
        all_symbols = set(major_idx_stocks)
        
        # Try multiple sources
        sources = [
            self.fetch_all_idx_stocks_from_idx_website,
            self.fetch_all_idx_stocks_from_sectors_app,
            self.fetch_all_idx_stocks_from_stockanalysis,
        ]
        
        for source_func in sources:
            try:
                print(f"Trying {source_func.__name__}...")
                symbols = source_func()
                if symbols:
                    # Clean symbols and add to set
                    clean_symbols = [s.strip().upper().replace('.JK', '') for s in symbols 
                                   if s and len(s.strip()) <= 6 and s.strip().isalnum()]
                    all_symbols.update(clean_symbols)
                    print(f"Added {len(clean_symbols)} symbols, total: {len(all_symbols)}")
                    time.sleep(1)  # Be respectful to APIs
                    
            except Exception as e:
                print(f"Error with {source_func.__name__}: {e}")
                continue
        
        # Convert to sorted list
        final_symbols = sorted(list(all_symbols))
        
        print(f"Final comprehensive list: {len(final_symbols)} Indonesian stocks")
        return final_symbols
    
    def load_all_idx_symbols(self, use_comprehensive=True):
        """Load all Indonesian stock symbols automatically"""
        if use_comprehensive:
            # Get comprehensive list from multiple sources
            all_symbols = self.get_comprehensive_stock_symbols()
        else:
            # Use basic predefined list
            all_symbols = [
                'BBCA', 'BBRI', 'BMRI', 'TLKM', 'ASII', 'UNVR', 'ICBP', 'KLBF',
                'SMGR', 'INDF', 'GGRM', 'HMSP', 'CPIN', 'ADRO', 'PTBA', 'BSDE'
            ]
        
        # Add .JK suffix for Yahoo Finance
        self.idx_symbols = [symbol + '.JK' if not symbol.endswith('.JK') else symbol 
                           for symbol in all_symbols]
        
        print(f"Loaded {len(self.idx_symbols)} Indonesian stocks with .JK suffix")
        return self.idx_symbols
    
    def validate_symbols(self, sample_size=10):
        """Validate a sample of symbols to ensure they work with yfinance"""
        if not self.idx_symbols:
            print("No symbols loaded. Please run load_all_idx_symbols() first.")
            return
        
        print(f"Validating {sample_size} random symbols...")
        
        import random
        sample_symbols = random.sample(self.idx_symbols, min(sample_size, len(self.idx_symbols)))
        
        valid_symbols = []
        invalid_symbols = []
        
        for symbol in sample_symbols:
            try:
                stock = yf.Ticker(symbol)
                data = stock.history(period='5d')
                
                if len(data) > 0:
                    valid_symbols.append(symbol)
                    print(f"✓ {symbol} - Valid")
                else:
                    invalid_symbols.append(symbol)
                    print(f"✗ {symbol} - No data")
                    
            except Exception as e:
                invalid_symbols.append(symbol)
                print(f"✗ {symbol} - Error: {str(e)[:50]}...")
        
        print(f"\nValidation Results:")
        print(f"Valid: {len(valid_symbols)}/{len(sample_symbols)}")
        print(f"Invalid: {len(invalid_symbols)}/{len(sample_symbols)}")
        
        if invalid_symbols:
            print(f"Invalid symbols: {invalid_symbols[:5]}{'...' if len(invalid_symbols) > 5 else ''}")
        
        return valid_symbols, invalid_symbols
    
    def fetch_all_idx_stocks_from_sectors_app(self):
        """Fetch Indonesian stocks from Sectors.app API"""
        try:
            url = "https://api.sectors.app/v1/companies/"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10, verify=certifi.where())
            
            if response.status_code == 200:
                data = response.json()
                symbols = []
                
                if isinstance(data, list):
                    for company in data:
                        if isinstance(company, dict) and 'symbol' in company:
                            symbols.append(company['symbol'])
                elif isinstance(data, dict) and 'data' in data:
                    for company in data['data']:
                        if isinstance(company, dict) and 'symbol' in company:
                            symbols.append(company['symbol'])
                
                if symbols:
                    print(f"Successfully fetched {len(symbols)} stocks from Sectors.app")
                    return symbols
                    
        except Exception as e:
            print(f"Error fetching from Sectors.app: {e}")
        
        return None
    
    def fetch_all_idx_stocks_from_stockanalysis(self):
        """Fetch Indonesian stocks by scraping stockanalysis.com"""
        try:
            url = "https://stockanalysis.com/list/indonesia-stock-exchange/"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10, verify=certifi.where())
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                symbols = []
                
                # Look for table with stock symbols
                table = soup.find('table')
                if table:
                    rows = table.find_all('tr')[1:]  # Skip header
                    for row in rows:
                        cells = row.find_all('td')
                        if cells:
                            # First cell usually contains the symbol
                            symbol_link = cells[0].find('a')
                            if symbol_link:
                                symbol = symbol_link.text.strip()
                                if symbol and len(symbol) <= 6:  # IDX symbols are typically 4-6 chars
                                    symbols.append(symbol)
                
                if symbols:
                    print(f"Successfully scraped {len(symbols)} stocks from StockAnalysis.com")
                    return symbols
                    
        except Exception as e:
            print(f"Error scraping from StockAnalysis: {e}")
        
        return None
    
    def fetch_all_idx_stocks_yahoo_components(self):
        """Fetch Indonesian stocks from Yahoo Finance IDX Composite components"""
        try:
            # Get IDX Composite components
            idx_ticker = yf.Ticker("^JKSE")
            
            # Try to get institutional holders or similar data that might contain symbols
            info = idx_ticker.info
            
            # This approach might not work directly, but we can try
            if 'holdings' in info:
                symbols = []
                for holding in info['holdings']:
                    if 'symbol' in holding:
                        symbols.append(holding['symbol'].replace('.JK', ''))
                
                if symbols:
                    print(f"Successfully fetched {len(symbols)} stocks from Yahoo Finance IDX components")
                    return symbols
                    
        except Exception as e:
            print(f"Error fetching from Yahoo Finance components: {e}")
        
        return None
    
    def get_comprehensive_stock_symbols(self):
        """Get comprehensive list using predefined major stocks + fetched symbols"""
        # Start with major known Indonesian stocks
        major_idx_stocks = [
            # Banking
            'BBCA', 'BBRI', 'BMRI', 'BBNI', 'BDMN', 'BRIS', 'NISP', 'MEGA', 'PNBN', 'BJBR',
            
            # Telecommunications
            'TLKM', 'ISAT', 'EXCL', 'FREN',
            
            # Consumer Goods
            'UNVR', 'ICBP', 'INDF', 'KLBF', 'KAEF', 'TSPC', 'MYOR', 'SIDO', 'PYFA', 'TCID',
            'DLTA', 'SKBM', 'MLBI', 'TBLA', 'PSDN', 'ULTJ', 'CEKA', 'ALTO', 'JPFA', 'STTP',
            
            # Automotive
            'ASII', 'AUTO', 'IMAS', 'SMSM', 'GDYR', 'MASA', 'NIPS', 'PRAS', 'BOLT', 'LPIN',
            
            # Cement & Construction
            'SMGR', 'INTP', 'SMBR', 'WTON', 'ADHI', 'WSKT', 'WIKA', 'PTPP', 'DGIK', 'TOTL',
            
            # Mining & Energy
            'ADRO', 'PTBA', 'ITMG', 'HRUM', 'TOBA', 'DEWA', 'COAL', 'GEMS', 'MYOH', 'BUMI',
            'AKRA', 'PTRO', 'MEDC', 'RUIS', 'ELSA',
            
            # Tobacco
            'HMSP', 'GGRM', 'RMBA', 'WIIM',
            
            # Property
            'BSDE', 'LPKR', 'ASRI', 'PWON', 'APLN', 'SMRA', 'CTRA', 'DILD', 'BEST', 'GPRA',
            'KIJA', 'MTSM', 'MTLA', 'BKSL', 'COWL', 'ELTY', 'FMII', 'GMTD', 'JRPT', 'KPIG',
            'LCGP', 'MDLN', 'MKPI', 'NIRO', 'OMRE', 'PPRO', 'PUDP', 'RBMS', 'RDTX', 'RODA',
            
            # Finance & Investment
            'BBTN', 'BTPN', 'MAYA', 'BFIN', 'ADMF', 'BBLD', 'BBYB', 'BCIC', 'BEKS', 'BINA',
            'BKSW', 'BNBA', 'BNII', 'BNLI', 'BSIM', 'BSWD', 'BTPS', 'BVIC', 'DNAR', 'INPC',
            'MCOR', 'PNBS', 'SDRA',
            
            # Technology & Media
            'GOTO', 'BUKA', 'MTEL', 'VIVA', 'TMPO', 'FILM', 'SCMA', 'ABBA', 'BMTR', 'CTBN',
            'EMTK', 'FAST', 'FORU', 'GSMF', 'HART', 'ITIC', 'KOBX', 'LINK', 'LPGI',
            'MAPI', 'META', 'MIDI', 'MPPA', 'OASA', 'PANS', 'RANC', 'SKYB', 'SOCI',
            'TRIO', 'WOOD',
            
            # Plantation
            'AALI', 'LSIP', 'UNSP', 'SIMP', 'SGRO', 'SSMS', 'TBMS', 'ANJT', 'BWPT', 'CPRO',
            'DSNG', 'GOLL', 'JAWA', 'MAGP', 'PALM', 'PPKS', 'SAMF', 'SMAR', 'TUAH',
            
            # Transportation & Logistics
            'BIRD', 'CMPP', 'GIAA', 'IPCM', 'JSMR', 'KARW', 'LRNA', 'MIRA', 'NELY', 'SAFE',
            'SMDR', 'TKIM', 'TOPS', 'WEHA', 'WINS', 'ZBRA',
            
            # Healthcare & Pharmaceuticals
            'MERK', 'DVLA', 'INAF', 'KBLF', 'MDRN', 'PRDA', 'SAIP', 'SILO', 'SOHO',
            
            # Retail & Trading
            'ACES', 'AMRT', 'CSAP', 'ECII', 'ERAA', 'HERO', 'LPPF', 'MKNT',
            'RALS', 'SORO', 'SPORT',
            
            # Additional Major Stocks
            'CPIN', 'ANTM', 'WSKT', 'WIKA', 'PGAS', 'JSMR', 'PWON', 'TINS',
            'BUMI', 'MDKA', 'BRMS', 'PNLF', 'BNBR', 'SCCO', 'TPIA'
        ]
        
        # Remove duplicates
        major_idx_stocks = list(set(major_idx_stocks))
        
        print(f"Starting with {len(major_idx_stocks)} major Indonesian stocks...")
        
        # Try to fetch additional symbols from various sources
        all_symbols = set(major_idx_stocks)
        
        # Try multiple sources
        sources = [
            self.fetch_all_idx_stocks_from_idx_website,
            self.fetch_all_idx_stocks_from_sectors_app,
            self.fetch_all_idx_stocks_from_stockanalysis,
        ]
        
        for source_func in sources:
            try:
                print(f"Trying {source_func.__name__}...")
                symbols = source_func()
                if symbols:
                    # Clean symbols and add to set
                    clean_symbols = [s.strip().upper().replace('.JK', '') for s in symbols 
                                   if s and len(s.strip()) <= 6 and s.strip().isalnum()]
                    all_symbols.update(clean_symbols)
                    print(f"Added {len(clean_symbols)} symbols, total: {len(all_symbols)}")
                    time.sleep(1)  # Be respectful to APIs
                    
            except Exception as e:
                print(f"Error with {source_func.__name__}: {e}")
                continue
        
        # Convert to sorted list
        final_symbols = sorted(list(all_symbols))
        
        print(f"Final comprehensive list: {len(final_symbols)} Indonesian stocks")
        return final_symbols
    
    def get_comprehensive_stock_symbols(self):
        """Get comprehensive list using predefined major stocks + fetched symbols"""
        # Start with major known Indonesian stocks
        major_idx_stocks = [
            # Banking
            'BBCA', 'BBRI', 'BMRI', 'BBNI', 'BDMN', 'BRIS', 'NISP', 'MEGA', 'PNBN', 'BJBR',
            
            # Telecommunications
            'TLKM', 'ISAT', 'EXCL', 'FREN',
            
            # Consumer Goods
            'UNVR', 'ICBP', 'INDF', 'KLBF', 'KAEF', 'TSPC', 'MYOR', 'SIDO', 'PYFA', 'TCID',
            'DLTA', 'SKBM', 'MLBI', 'TBLA', 'PSDN', 'ULTJ', 'CEKA', 'ALTO', 'JPFA', 'STTP',
            
            # Automotive
            'ASII', 'AUTO', 'IMAS', 'SMSM', 'GDYR', 'MASA', 'NIPS', 'PRAS', 'BOLT', 'LPIN',
            
            # Cement & Construction
            'SMGR', 'INTP', 'SMBR', 'WTON', 'ADHI', 'WSKT', 'WIKA', 'PTPP', 'DGIK', 'TOTL',
            
            # Mining & Energy
            'ADRO', 'PTBA', 'ITMG', 'HRUM', 'TOBA', 'DEWA', 'COAL', 'GEMS', 'MYOH', 'BUMI',
            'AKRA', 'PTRO', 'MEDC', 'RUIS', 'ELSA',
            
            # Tobacco
            'HMSP', 'GGRM', 'RMBA', 'WIIM',
            
            # Property
            'BSDE', 'LPKR', 'ASRI', 'PWON', 'APLN', 'SMRA', 'CTRA', 'DILD', 'BEST', 'GPRA',
            'KIJA', 'MTSM', 'MTLA', 'BKSL', 'COWL', 'ELTY', 'FMII', 'GMTD', 'JRPT', 'KPIG',
            'LCGP', 'MDLN', 'MKPI', 'NIRO', 'OMRE', 'PPRO', 'PUDP', 'RBMS', 'RDTX', 'RODA',
            
            # Finance & Investment
            'BBTN', 'BTPN', 'MAYA', 'BFIN', 'ADMF', 'BBLD', 'BBYB', 'BCIC', 'BEKS', 'BINA',
            'BKSW', 'BNBA', 'BNII', 'BNLI', 'BSIM', 'BSWD', 'BTPS', 'BVIC', 'DNAR', 'INPC',
            'MAYA', 'MCOR', 'PNBS', 'SDRA',
            
            # Technology & Media
            'GOTO', 'BUKA', 'MTEL', 'VIVA', 'TMPO', 'FILM', 'SCMA', 'ABBA', 'BMTR', 'CTBN',
            'EMTK', 'FAST', 'FORU', 'GSMF', 'HART', 'HMSP', 'ITIC', 'KOBX', 'LINK', 'LPGI',
            'MAPI', 'META', 'MIDI', 'MPPA', 'NETWORK', 'OASA', 'PANS', 'RANC', 'SKYB', 'SOCI',
            'TRIO', 'WOOD',
            
            # Plantation
            'AALI', 'LSIP', 'UNSP', 'SIMP', 'SGRO', 'SSMS', 'TBMS', 'ANJT', 'BWPT', 'CPRO',
            'DSNG', 'GOLL', 'JAWA', 'MAGP', 'PALM', 'PPKS', 'SAMF', 'SMAR', 'TUAH',
            
            # Transportation & Logistics
            'BIRD', 'CMPP', 'GIAA', 'IPCM', 'JSMR', 'KARW', 'LRNA', 'MIRA', 'NELY', 'SAFE',
            'SMDR', 'SOCI', 'TKIM', 'TOPS', 'WEHA', 'WINS', 'ZBRA',
            
            # Healthcare & Pharmaceuticals
            'KAEF', 'KLBF', 'MERK', 'PYFA', 'SIDO', 'SQMI', 'TSPC', 'DVLA', 'INAF', 'KBLF',
            'MDRN', 'PRDA', 'SAIP', 'SILO', 'SOHO',
            
            # Retail & Trading
            'ACES', 'AMRT', 'CSAP', 'ECII', 'ERAA', 'HERO', 'LPPF', 'MAPI', 'MIDI', 'MKNT',
            'MPPA', 'RALS', 'RANC', 'SKYB', 'SORO', 'SPORT', 'TRIO'
        ]
        
        # Remove duplicates
        major_idx_stocks = list(set(major_idx_stocks))
        
        print(f"Starting with {len(major_idx_stocks)} major Indonesian stocks...")
        
        # Try to fetch additional symbols from various sources
        all_symbols = set(major_idx_stocks)
        
        # Try multiple sources
        sources = [
            self.fetch_all_idx_stocks_from_idx_website,
            self.fetch_all_idx_stocks_from_sectors_app,
            self.fetch_all_idx_stocks_from_stockanalysis,
        ]
        
        for source_func in sources:
            try:
                print(f"Trying {source_func.__name__}...")
                symbols = source_func()
                if symbols:
                    # Clean symbols and add to set
                    clean_symbols = [s.strip().upper().replace('.JK', '') for s in symbols 
                                   if s and len(s.strip()) <= 6 and s.strip().isalnum()]
                    all_symbols.update(clean_symbols)
                    print(f"Added {len(clean_symbols)} symbols, total: {len(all_symbols)}")
                    time.sleep(1)  # Be respectful to APIs
                    
            except Exception as e:
                print(f"Error with {source_func.__name__}: {e}")
                continue
        
        # Convert to sorted list
        final_symbols = sorted(list(all_symbols))
        
        print(f"Final comprehensive list: {len(final_symbols)} Indonesian stocks")
        return final_symbols
    
    def get_stock_data(self, symbol, period='6mo'):
        """Fetch stock data using yfinance"""
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period=period)
            return data
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return None
    
    def calculate_accumulation_distribution(self, data):
        """Calculate Accumulation/Distribution Line"""
        if data is None or len(data) < 20:
            return None
        
        high = data['High'].values
        low = data['Low'].values
        close = data['Close'].values
        volume = data['Volume'].values
        
        # Calculate Money Flow Multiplier
        mf_multiplier = ((close - low) - (high - close)) / (high - low)
        mf_multiplier = np.nan_to_num(mf_multiplier)  # Handle division by zero
        
        # Calculate Money Flow Volume
        mf_volume = mf_multiplier * volume
        
        # Calculate Accumulation/Distribution Line
        ad_line = np.cumsum(mf_volume)
        
        return ad_line
    
    def detect_chart_patterns(self, data):
        """Detect basic chart patterns using technical indicators"""
        if data is None or len(data) < 50:
            return {}
        
        close = data['Close'].values
        high = data['High'].values
        low = data['Low'].values
        
        patterns = {}
        
        # Moving averages
        sma_20 = talib.SMA(close, timeperiod=20)
        sma_50 = talib.SMA(close, timeperiod=50)
        ema_12 = talib.EMA(close, timeperiod=12)
        ema_26 = talib.EMA(close, timeperiod=26)
        
        # MACD
        macd, macd_signal, macd_hist = talib.MACD(close)
        
        # RSI
        rsi = talib.RSI(close, timeperiod=14)
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = talib.BBANDS(close)
        
        # Pattern detection
        patterns['golden_cross'] = sma_20[-1] > sma_50[-1] and sma_20[-2] <= sma_50[-2]
        patterns['death_cross'] = sma_20[-1] < sma_50[-1] and sma_20[-2] >= sma_50[-2]
        patterns['macd_bullish'] = macd[-1] > macd_signal[-1] and macd[-2] <= macd_signal[-2]
        patterns['macd_bearish'] = macd[-1] < macd_signal[-1] and macd[-2] >= macd_signal[-2]
        patterns['rsi_oversold'] = rsi[-1] < 30
        patterns['rsi_overbought'] = rsi[-1] > 70
        patterns['bb_squeeze'] = (bb_upper[-1] - bb_lower[-1]) / bb_middle[-1] < 0.1
        patterns['price_above_sma20'] = close[-1] > sma_20[-1]
        patterns['volume_surge'] = data['Volume'].iloc[-1] > data['Volume'].rolling(20).mean().iloc[-1] * 1.5
        
        return patterns
    
    def analyze_volume_profile(self, data):
        """Analyze volume patterns"""
        if data is None or len(data) < 20:
            return {}
        
        volume = data['Volume']
        close = data['Close']
        
        volume_analysis = {}
        
        # Volume moving averages
        vol_sma_20 = volume.rolling(20).mean()
        vol_sma_50 = volume.rolling(50).mean()
        
        # Volume trend
        volume_analysis['avg_volume_20d'] = vol_sma_20.iloc[-1]
        volume_analysis['avg_volume_50d'] = vol_sma_50.iloc[-1]
        volume_analysis['volume_trend'] = 'increasing' if vol_sma_20.iloc[-1] > vol_sma_50.iloc[-1] else 'decreasing'
        
        # Price-Volume correlation
        price_change = close.pct_change()
        volume_change = volume.pct_change()
        correlation = price_change.corr(volume_change)
        volume_analysis['price_volume_correlation'] = correlation
        
        # On Balance Volume (OBV)
        obv = talib.OBV(close.values, volume.values)
        volume_analysis['obv_trend'] = 'bullish' if obv[-1] > obv[-10] else 'bearish'
        
        return volume_analysis
    
    def simulate_broker_summary(self, symbol):
        """Simulate broker buy/sell accumulation data"""
        # In real implementation, you would fetch this from Phintraco API or data feed
        # This is a simulation for demonstration
        np.random.seed(hash(symbol) % 1000)
        
        broker_summary = {
            'net_foreign': np.random.uniform(-1000000, 1000000),
            'net_domestic': np.random.uniform(-500000, 500000),
            'total_buy_volume': np.random.uniform(1000000, 10000000),
            'total_sell_volume': np.random.uniform(1000000, 10000000),
            'broker_accumulation_score': np.random.uniform(-1, 1)  # -1 (selling) to 1 (buying)
        }
        
        return broker_summary
    
    def screen_stocks(self, 
                     min_accumulation_score=0.3,
                     required_patterns=None,
                     min_volume_trend='stable'):
        """Screen stocks based on specified criteria"""
        
        if required_patterns is None:
            required_patterns = ['price_above_sma20', 'volume_surge']
        
        screened_results = []
        
        print(f"Screening {len(self.idx_symbols)} stocks...")
        
        for symbol in self.idx_symbols:
            try:
                print(f"Processing {symbol}...")
                
                # Get stock data
                stock_data = self.get_stock_data(symbol)
                if stock_data is None or len(stock_data) < 50:
                    continue
                
                # Calculate technical indicators
                ad_line = self.calculate_accumulation_distribution(stock_data)
                patterns = self.detect_chart_patterns(stock_data)
                volume_analysis = self.analyze_volume_profile(stock_data)
                broker_data = self.simulate_broker_summary(symbol)
                
                # Apply filters
                stock_score = 0
                passed_filters = []
                
                # Broker accumulation filter
                if broker_data['broker_accumulation_score'] >= min_accumulation_score:
                    stock_score += 1
                    passed_filters.append('broker_accumulation')
                
                # Pattern filters
                pattern_score = sum([1 for pattern in required_patterns if patterns.get(pattern, False)])
                if pattern_score >= len(required_patterns) * 0.7:  # At least 70% of required patterns
                    stock_score += 1
                    passed_filters.append('chart_patterns')
                
                # Volume filter
                if (min_volume_trend == 'increasing' and volume_analysis.get('volume_trend') == 'increasing') or \
                   (min_volume_trend == 'stable'):
                    stock_score += 1
                    passed_filters.append('volume_trend')
                
                # Store results
                result = {
                    'symbol': symbol,
                    'score': stock_score,
                    'passed_filters': passed_filters,
                    'current_price': stock_data['Close'].iloc[-1],
                    'volume_20d_avg': volume_analysis.get('avg_volume_20d', 0),
                    'broker_accumulation_score': broker_data['broker_accumulation_score'],
                    'patterns': patterns,
                    'volume_analysis': volume_analysis,
                    'ad_line_trend': 'bullish' if len(ad_line) > 1 and ad_line[-1] > ad_line[-10] else 'bearish'
                }
                
                screened_results.append(result)
                
            except Exception as e:
                print(f"Error processing {symbol}: {e}")
                continue
        
        # Sort by score
        screened_results.sort(key=lambda x: x['score'], reverse=True)
        self.screened_stocks = screened_results
        
        return screened_results
    
    def display_results(self, top_n=10):
        """Display screening results"""
        if not self.screened_stocks:
            print("No screening results available. Run screen_stocks() first.")
            return
        
        print(f"\n{'='*80}")
        print(f"TOP {top_n} SCREENED INDONESIAN STOCKS")
        print(f"{'='*80}")
        
        for i, stock in enumerate(self.screened_stocks[:top_n]):
            print(f"\n{i+1}. {stock['symbol']} (Score: {stock['score']}/3)")
            print(f"   Current Price: Rp {stock['current_price']:,.2f}")
            print(f"   Broker Accumulation Score: {stock['broker_accumulation_score']:.3f}")
            print(f"   A/D Line Trend: {stock['ad_line_trend']}")
            print(f"   Volume Trend: {stock['volume_analysis'].get('volume_trend', 'N/A')}")
            print(f"   Passed Filters: {', '.join(stock['passed_filters'])}")
            
            # Show key patterns
            key_patterns = [k for k, v in stock['patterns'].items() if v]
            if key_patterns:
                print(f"   Active Patterns: {', '.join(key_patterns[:3])}")
    
    def plot_stock_analysis(self, symbol, save_chart=False):
        """Plot detailed analysis for a specific stock"""
        stock_data = self.get_stock_data(symbol, period='6mo')
        if stock_data is None:
            print(f"No data available for {symbol}")
            return
        
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        
        # Price and volume
        axes[0].plot(stock_data.index, stock_data['Close'], label='Close Price', linewidth=2)
        axes[0].set_title(f'{symbol} - Price Analysis')
        axes[0].set_ylabel('Price (Rp)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Volume
        axes[1].bar(stock_data.index, stock_data['Volume'], alpha=0.7, color='orange')
        axes[1].set_title('Volume Analysis')
        axes[1].set_ylabel('Volume')
        axes[1].grid(True, alpha=0.3)
        
        # A/D Line
        ad_line = self.calculate_accumulation_distribution(stock_data)
        if ad_line is not None:
            axes[2].plot(stock_data.index[-len(ad_line):], ad_line, 
                        label='A/D Line', color='purple', linewidth=2)
            axes[2].set_title('Accumulation/Distribution Line')
            axes[2].set_ylabel('A/D Value')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_chart:
            plt.savefig(f'{symbol}_analysis.png', dpi=300, bbox_inches='tight')
            print(f"Chart saved as {symbol}_analysis.png")
        
        plt.show()

# Usage Example
if __name__ == "__main__":
    # Initialize the screener
    screener = IndonesianStockScreener()
    
    # Load ALL Indonesian stock symbols automatically
    print("Fetching all Indonesian stock symbols...")
    screener.load_all_idx_symbols(use_comprehensive=False)
    
    # Validate a sample of symbols
    print("\nValidating symbols...")
    screener.validate_symbols(sample_size=20)
    
    # Screen stocks with custom criteria
    print("\nStarting comprehensive stock screening...")
    results = screener.screen_stocks(
        min_accumulation_score=0.2,  # Minimum broker accumulation score
        required_patterns=['price_above_sma20', 'volume_surge'],  # Required patterns
        min_volume_trend='stable'  # Volume trend requirement
    )
    
    # Display results
    screener.display_results(top_n=15)
    
    # Plot analysis for top stock
    if results:
        top_stock = results[0]['symbol']
        print(f"\nGenerating detailed analysis for {top_stock}...")
        screener.plot_stock_analysis(top_stock)