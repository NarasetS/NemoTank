import yfinance as yf
import pandas as pd
import json
import os
import time
import logging
import requests
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- CONFIGURATION ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FundamentalDataPipeline:
    """
    Acquires data for all stocks in the S&P 500 with robust error handling 
    and manual fundamental calculations.
    """
    def __init__(self, storage_dir="market_data", batch_size=5, delay=3.5):
        self.storage_dir = storage_dir
        self.batch_size = batch_size
        self.delay = delay
        self.tickers = []
        self.raw_data = []
        self._init_storage()

    def _init_storage(self):
        """Ensures the storage directory exists."""
        if not os.path.exists(self.storage_dir):
            os.makedirs(self.storage_dir)

    def discover_tickers(self, source='sp500'):
        """
        Stage 1: Discovery. 
        Scrapes all current tickers from Wikipedia (S&P 500).
        """
        if source == 'sp500':
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            headers = {"User-Agent": "Mozilla/5.0"}
            try:
                response = requests.get(url, headers=headers)
                response.raise_for_status()
                table = pd.read_html(response.text)[0]
                # Replace '.' with '-' for yfinance compatibility (e.g., BRK.B -> BRK-B)
                self.tickers = [t.replace('.', '-') for t in table['Symbol'].tolist()]
                logger.info(f"Discovered all {len(self.tickers)} S&P 500 tickers.")
            except Exception as e:
                logger.error(f"Discovery failed: {e}")

    def _get_financial_components(self, stock):
        """Extracts specific line items for Manual EBIT and Pure Greenblatt ROC."""
        try:
            # Fetch statements - these take time and can trigger rate limits
            inc = stock.income_stmt
            bal = stock.balance_sheet
            
            if inc.empty or bal.empty: 
                return None, None, None, None
            
            # Use the most recent reported year
            latest_inc = inc.iloc[:, 0]
            latest_bal = bal.iloc[:, 0]
            
            # 1. Manual EBIT (Earnings Before Interest and Taxes)
            ebit = latest_inc.get('EBIT')
            if pd.isna(ebit):
                ebit = (latest_inc.get('Net Income', 0) + 
                        latest_inc.get('Interest Expense', 0) + 
                        latest_inc.get('Tax Provision', 0))
            
            # 2. Components for Greenblatt ROC
            # NWC = Current Assets - Current Liabilities
            curr_assets = latest_bal.get('Total Current Assets', 0)
            curr_liabs = latest_bal.get('Total Current Liabilities', 0)
            nwc = curr_assets - curr_liabs
            
            # Net Fixed Assets (PPE)
            nfa = latest_bal.get('Net PPE', latest_bal.get('Properties', 0))
            
            # Total Assets for Conventional ROIC
            total_assets = latest_bal.get('Total Assets', 0)
            
            return float(ebit), float(nwc), float(nfa), float(total_assets)
        except Exception:
            return None, None, None, None

    def _fetch_single_ticker(self, ticker):
        """Acquires all necessary fundamentals for one stock."""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Fetch granular data for our custom calculations
            ebit, nwc, nfa, total_assets = self._get_financial_components(stock)
            
            # If critical ebit is missing even after manual attempt, we skip
            if ebit is None: return None

            return {
                'ticker': ticker,
                'sector': info.get('sector'),
                'industry': info.get('industry'),
                'price': info.get('currentPrice'),
                'ebit': ebit,
                'nwc': nwc,
                'nfa': nfa,
                'total_assets': total_assets,
                'enterprise_value': info.get('enterpriseValue'),
                'net_income': info.get('netIncomeToCommon'),
                'fcf': info.get('freeCashflow'),
                'total_debt': info.get('totalDebt'),
                'ebitda': info.get('ebitda'),
                'forward_pe': info.get('forwardPE'),
                'scan_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        except Exception as e:
            logger.debug(f"Failed to fetch {ticker}: {e}")
            return None

    def run_acquisition(self):
        """
        Stage 3: Market-Wide Acquisition.
        Processes the entire discovered list in batches to prevent IP bans.
        """
        total = len(self.tickers)
        if total == 0:
            logger.error("Ticker list is empty. Run discovery first.")
            return

        logger.info(f"Starting full market scan for {total} stocks...")
        
        for i in range(0, total, self.batch_size):
            batch = self.tickers[i:i + self.batch_size]
            
            # We use a low number of workers to prevent concurrent rate-limiting
            with ThreadPoolExecutor(max_workers=2) as executor:
                results = list(executor.map(self._fetch_single_ticker, batch))
                self.raw_data.extend([r for r in results if r])
            
            logger.info(f"Progress: {len(self.raw_data)} tickers captured. Sleeping {self.delay}s...")
            time.sleep(self.delay)

    def save_to_json(self):
        """Stage 5: Persist the full market snapshot."""
        if not self.raw_data:
            logger.warning("No data acquired to save.")
            return
            
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        filename = f"full_market_scan_{timestamp}.json"
        filepath = os.path.join(self.storage_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(self.raw_data, f, indent=4)
        
        logger.info(f"Market-wide scan completed. Data saved to {filepath}")

if __name__ == "__main__":
    # Initialize the full-scale pipeline
    pipeline = FundamentalDataPipeline()
    
    # 1. Discover every ticker in the S&P 500
    pipeline.discover_tickers()
    
    # 2. Acquire data for the ENTIRE list (no limit)
    # WARNING: This will take ~30-40 minutes due to rate-limit respect
    pipeline.run_acquisition()
    
    # 3. Save the master snapshot
    pipeline.save_to_json()