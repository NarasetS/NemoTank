import yfinance as yf
import pandas as pd
import json
import os
import time
import logging
import requests
import re
from io import StringIO
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

# --- CONFIGURATION ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GlobalFundamentalPipeline: 
    """
    Acquires high-fidelity data for US and Thailand (SET) markets.
    Features: Flexible index scraping, DR filtering, and Shark Tank metrics.
    """
    def __init__(self, storage_dir="market_data", batch_size=5, delay=3.0):
        self.storage_dir = storage_dir
        self.batch_size = batch_size
        self.delay = delay
        self.tickers = []
        self.raw_data = []
        if not os.path.exists(self.storage_dir):
            os.makedirs(self.storage_dir)

    def discover_tickers(self, market='US'):
        """Scrapes tickers for the selected market with flexible column detection."""
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
        
        try:
            if market == 'US':
                # Scrape S&P 500
                url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
                resp = requests.get(url, headers=headers)
                sp500 = pd.read_html(StringIO(resp.text), flavor='lxml')[0]
                
                # Scrape Nasdaq 100 for broader US coverage
                nas_url = "https://en.wikipedia.org/wiki/Nasdaq-100#Components"
                nas_resp = requests.get(nas_url, headers=headers)
                nas100 = pd.read_html(StringIO(nas_resp.text), flavor='lxml')[4] # Usually index 4
                
                # Combine and clean
                raw_tickers = sp500['Symbol'].tolist() + nas100['Ticker'].tolist()
                self.tickers = list(set([t.replace('.', '-') for t in raw_tickers]))
                logger.info(f"Discovered {len(self.tickers)} US Tickers (S&P 500 + Nasdaq 100).")
                
            elif market == 'SET':
                # Primary Source: StockAnalysis.com (Complete SET List)
                # Backup Sources: Wikipedia General List + SET Indices
                urls = [
                    "https://stockanalysis.com/list/stock-exchange-of-thailand/",
                    "https://en.wikipedia.org/wiki/List_of_companies_of_Thailand",
                    "https://en.wikipedia.org/wiki/SET50_Index_and_SET100_Index"
                ]
                
                all_set_tickers = []
                for url in urls:
                    try:
                        response = requests.get(url, headers=headers)
                        if response.status_code != 200:
                            logger.warning(f"Skipping {url}: Status {response.status_code}")
                            continue
                            
                        tables = pd.read_html(StringIO(response.text), flavor='lxml')
                        
                        for t in tables:
                            symbol_col = next((c for c in t.columns if any(k in str(c).lower() for k in ['symbol', 'ticker', 'code'])), None)
                            
                            if symbol_col:
                                raw_symbols = t[symbol_col].dropna().astype(str).tolist()
                                for sym in raw_symbols:
                                    clean_sym = sym.strip().upper()
                                    if 2 <= len(clean_sym) <= 10 and not clean_sym.isdigit() and any(c.isalpha() for c in clean_sym):
                                        all_set_tickers.append(f"{clean_sym}.BK")
                                        
                    except Exception as e:
                        logger.warning(f"Failed to scrape {url}: {e}")

                # Filter out DRs
                regex_dr = re.compile(r'^[A-Z]+\d{2}\.BK$')
                unique_tickers = list(set([t for t in all_set_tickers if not t.startswith('TICKER')]))
                self.tickers = [t for t in unique_tickers if not regex_dr.match(t)]
                
                if self.tickers:
                    logger.info(f"Discovered {len(self.tickers)} Thai Tickers (Aggregated Sources).")
                else:
                    raise ValueError("Could not find any tables with ticker symbols for the Thai market.")

        except Exception as e:
            logger.error(f"Discovery failed for {market}: {e}")

    def _get_financial_components(self, stock):
        """Extracts specific line items for Manual EBIT and Greenblatt ROC."""
        try:
            inc = stock.income_stmt
            bal = stock.balance_sheet
            if inc.empty or bal.empty: return None, None, None, None
            
            latest_inc, latest_bal = inc.iloc[:, 0], bal.iloc[:, 0]
            
            # EBIT Calculation
            ebit = latest_inc.get('EBIT')
            if pd.isna(ebit):
                ebit = (latest_inc.get('Net Income', 0) + 
                        latest_inc.get('Interest Expense', 0) + 
                        latest_inc.get('Tax Provision', 0))
            
            # ROC Components
            nwc = latest_bal.get('Total Current Assets', 0) - latest_bal.get('Total Current Liabilities', 0)
            nfa = latest_bal.get('Net PPE', latest_bal.get('Properties', 0))
            total_assets = latest_bal.get('Total Assets', 0)
            
            return float(ebit), float(nwc), float(nfa), float(total_assets)
        except:
            return None, None, None, None

    def _fetch_single_ticker(self, ticker):
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            long_name = info.get('longName', '').upper()
            if 'DEPOSITARY RECEIPT' in long_name:
                return None

            ebit, nwc, nfa, total_assets = self._get_financial_components(stock)
            if ebit is None: return None

            return {
                'ticker': ticker,
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'price': info.get('currentPrice'),
                'ebit': ebit,
                'nwc': nwc,
                'nfa': nfa,
                'total_assets': total_assets,
                'market_cap': info.get('marketCap'), 
                'enterprise_value': info.get('enterpriseValue'),
                'net_income': info.get('netIncomeToCommon'),
                'fcf': info.get('freeCashflow'),
                'total_debt': info.get('totalDebt'),
                'ebitda': info.get('ebitda'),
                'forward_pe': info.get('forwardPE'),
                
                # --- SHARK TANK METRICS ---
                'revenue': info.get('totalRevenue'),
                'revenue_growth': info.get('revenueGrowth'), # Year-over-Year Growth
                'gross_margins': info.get('grossMargins'),
                'operating_margins': info.get('operatingMargins'),
                'return_on_equity': info.get('returnOnEquity'),
                
                'currency': info.get('currency', 'USD'),
                'scan_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        except:
            return None

    def run_acquisition(self, market_label='US'):
        self.raw_data = [] 
        for i in range(0, len(self.tickers), self.batch_size):
            batch = self.tickers[i:i + self.batch_size]
            with ThreadPoolExecutor(max_workers=2) as exec:
                self.raw_data.extend([r for r in exec.map(self._fetch_single_ticker, batch) if r])
            logger.info(f"Acquired {len(self.raw_data)} tickers in {market_label}...")
            time.sleep(self.delay)
            
        filename = f"market_{market_label}_{datetime.now().strftime('%Y-%m-%d')}.json"
        filepath = os.path.join(self.storage_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(self.raw_data, f, indent=4)
        logger.info(f"Saved {market_label} data to {filename}")

if __name__ == "__main__":
    pipeline = GlobalFundamentalPipeline()
    
    # Run SET
    pipeline.discover_tickers(market='SET')
    if pipeline.tickers:
        pipeline.run_acquisition(market_label='SET')
    
    # Run US
    pipeline.discover_tickers(market='US')
    if pipeline.tickers:
        pipeline.run_acquisition(market_label='US')