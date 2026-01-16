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
    Features: Index tagging (S&P500/Nasdaq100), Exchange detection, and Shark Tank metrics.
    """
    def __init__(self, storage_dir="market_data", batch_size=5, delay=3.0):
        self.storage_dir = storage_dir
        self.batch_size = batch_size
        self.delay = delay
        self.tickers = []
        self.raw_data = []
        
        # Index Sets for Tagging
        self.sp500_set = set()
        self.nasdaq100_set = set()
        
        if not os.path.exists(self.storage_dir):
            os.makedirs(self.storage_dir)

    def _fetch_indices(self, headers):
        """Helper to fetch constituent lists for tagging."""
        try:
            # S&P 500
            url_sp = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            sp_df = pd.read_html(StringIO(requests.get(url_sp, headers=headers).text), flavor='lxml')[0]
            self.sp500_set = set([t.replace('.', '-') for t in sp_df['Symbol'].tolist()])
            logger.info(f"Loaded {len(self.sp500_set)} S&P 500 constituents for tagging.")
        except Exception as e:
            logger.warning(f"Failed to load S&P 500 list: {e}")

        try:
            # Nasdaq 100
            url_nas = "https://en.wikipedia.org/wiki/Nasdaq-100#Components"
            nas_df = pd.read_html(StringIO(requests.get(url_nas, headers=headers).text), flavor='lxml')[4]
            self.nasdaq100_set = set([t.replace('.', '-') for t in nas_df['Ticker'].tolist()])
            logger.info(f"Loaded {len(self.nasdaq100_set)} Nasdaq 100 constituents for tagging.")
        except Exception as e:
            logger.warning(f"Failed to load Nasdaq 100 list: {e}")

    def discover_tickers(self, market='US'):
        """Scrapes tickers for the selected market."""
        # Generic headers
        browser_headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
        sec_headers = {"User-Agent": "NemoTankApp/1.0 (educational_research@example.com)"}
        
        if market == 'US':
            # Pre-fetch indices for tagging later
            self._fetch_indices(browser_headers)
            
            logger.info("Fetching complete US market list...")
            # STRATEGY 1: SEC Official List (Reliable)
            try:
                url = "https://www.sec.gov/files/company_tickers.json"
                r = requests.get(url, headers=sec_headers, timeout=15)
                data = r.json()
                sec_tickers = [val['ticker'] for val in data.values()]
                self.tickers = list(set([str(t).replace('.', '-') for t in sec_tickers]))
                logger.info(f"Discovered {len(self.tickers)} US Tickers via SEC.")
            except Exception as e:
                logger.warning(f"SEC source failed: {e}. Fallback to Nasdaq Trader...")
                # STRATEGY 2: Nasdaq Trader
                try:
                    url = "http://ftp.nasdaqtrader.com/dynamic/SymDir/nasdaqtraded.txt"
                    r = requests.get(url, headers=browser_headers, timeout=15)
                    df = pd.read_csv(StringIO(r.text), sep='|')
                    if len(df) > 0: df = df[:-1]
                    df = df[df['Test Issue'] == 'N']
                    if 'ETF' in df.columns: df = df[df['ETF'] == 'N']
                    self.tickers = list(set([str(t).replace('.', '-') for t in df['Symbol'].tolist()]))
                    logger.info(f"Discovered {len(self.tickers)} US Tickers via Nasdaq Trader.")
                except Exception as e2:
                    logger.error(f"All US sources failed: {e2}")

        elif market == 'SET':
            # Thai Market Discovery
            urls = [
                "https://stockanalysis.com/list/stock-exchange-of-thailand/",
                "https://en.wikipedia.org/wiki/List_of_companies_of_Thailand",
                "https://en.wikipedia.org/wiki/SET50_Index_and_SET100_Index"
            ]
            all_set_tickers = []
            for url in urls:
                try:
                    response = requests.get(url, headers=browser_headers, timeout=15)
                    if response.status_code != 200: continue
                    tables = pd.read_html(StringIO(response.text), flavor='lxml')
                    for t in tables:
                        symbol_col = next((c for c in t.columns if any(k in str(c).lower() for k in ['symbol', 'ticker', 'code'])), None)
                        if symbol_col:
                            raw_symbols = t[symbol_col].dropna().astype(str).tolist()
                            for sym in raw_symbols:
                                clean_sym = sym.strip().upper()
                                if 2 <= len(clean_sym) <= 10 and not clean_sym.isdigit() and any(c.isalpha() for c in clean_sym):
                                    all_set_tickers.append(f"{clean_sym}.BK")
                except: continue

            # DR Filter
            regex_dr = re.compile(r'^[A-Z]+\d{2}\.BK$')
            unique_tickers = list(set([t for t in all_set_tickers if not t.startswith('TICKER')]))
            self.tickers = [t for t in unique_tickers if not regex_dr.match(t)]
            logger.info(f"Discovered {len(self.tickers)} Thai Tickers.")

    def _get_financial_components(self, stock):
        try:
            inc = stock.income_stmt
            bal = stock.balance_sheet
            if inc.empty or bal.empty: return None, None, None, None
            
            latest_inc, latest_bal = inc.iloc[:, 0], bal.iloc[:, 0]
            ebit = latest_inc.get('EBIT')
            if pd.isna(ebit):
                ebit = (latest_inc.get('Net Income', 0) + latest_inc.get('Interest Expense', 0) + latest_inc.get('Tax Provision', 0))
            
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
            
            if 'DEPOSITARY RECEIPT' in info.get('longName', '').upper(): return None

            ebit, nwc, nfa, total_assets = self._get_financial_components(stock)
            if ebit is None: return None

            # --- Tagging Indices ---
            # Remove .BK suffix for checking sets if needed, but US tickers match exactly
            is_sp500 = ticker in self.sp500_set
            is_nas100 = ticker in self.nasdaq100_set

            return {
                'ticker': ticker,
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'exchange': info.get('exchange', 'Unknown'), # NYSE, NMS (Nasdaq), etc.
                'is_sp500': is_sp500,
                'is_nasdaq100': is_nas100,
                'price': info.get('currentPrice'),
                'ebit': ebit, 'nwc': nwc, 'nfa': nfa, 'total_assets': total_assets,
                'market_cap': info.get('marketCap'), 'enterprise_value': info.get('enterpriseValue'),
                'net_income': info.get('netIncomeToCommon'), 'fcf': info.get('freeCashflow'),
                'total_debt': info.get('totalDebt'), 'ebitda': info.get('ebitda'),
                'forward_pe': info.get('forwardPE'), 'revenue': info.get('totalRevenue'),
                'revenue_growth': info.get('revenueGrowth'), 'gross_margins': info.get('grossMargins'),
                'operating_margins': info.get('operatingMargins'), 'return_on_equity': info.get('returnOnEquity'),
                'accruals_ratio': 0, # Placeholder, calc in engine
                'scan_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        except:
            return None

    def run_acquisition(self, market_label='US'):
        self.raw_data = [] 
        total = len(self.tickers)
        for i in range(0, total, self.batch_size):
            batch = self.tickers[i:i + self.batch_size]
            with ThreadPoolExecutor(max_workers=2) as exec:
                self.raw_data.extend([r for r in exec.map(self._fetch_single_ticker, batch) if r])
            logger.info(f"Progress: {len(self.raw_data)}/{total} captured for {market_label}...")
            time.sleep(self.delay)
            
        filename = f"market_{market_label}_{datetime.now().strftime('%Y-%m-%d')}.json"
        filepath = os.path.join(self.storage_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(self.raw_data, f, indent=4)
        logger.info(f"Saved {market_label} data to {filename}")

if __name__ == "__main__":
    pipeline = GlobalFundamentalPipeline()
    #
    pipeline.discover_tickers(market='SET')
    if pipeline.tickers:
        pipeline.run_acquisition(market_label='SET')

    # Run US (Full Market with Tagging)
    pipeline.discover_tickers(market='US')
    if pipeline.tickers:
        pipeline.run_acquisition(market_label='US')

