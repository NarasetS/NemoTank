import yfinance as yf
import pandas as pd
import numpy as np

class BacktestEngine:
    """
    Simulates portfolio performance based on historical price action.
    Robust handling for yfinance data structures.
    """
    def __init__(self):
        pass

    def fetch_price_history(self, tickers, benchmark_ticker='^GSPC', period='3y'):
        """
        Fetches adjusted close prices for the portfolio and a benchmark.
        """
        # Ensure tickers is a list
        if not isinstance(tickers, list):
            tickers = [tickers]
            
        # Clean ticker list
        valid_tickers = [t for t in tickers if isinstance(t, str) and len(t) > 0]
        if not valid_tickers:
            return pd.DataFrame(), pd.Series()

        # Combine logic
        all_symbols = list(set(valid_tickers + [benchmark_ticker]))
        
        try:
            # auto_adjust=True gets the split/dividend adjusted price (better than Adj Close)
            # threads=False can sometimes be more stable for smaller batches
            data = yf.download(all_symbols, period=period, progress=False, auto_adjust=True)
            
            # Handle yfinance "No Data" or Empty response
            if data.empty:
                return pd.DataFrame(), pd.Series()

            # yfinance return structure varies:
            # 1. Single Ticker: DataFrame with cols [Open, High, Low, Close, Volume]
            # 2. Multi Ticker: DataFrame with MultiIndex columns (Price, Ticker)
            
            prices_df = pd.DataFrame()

            # Scenario A: Multiple Tickers (MultiIndex columns)
            if isinstance(data.columns, pd.MultiIndex):
                # We want the 'Close' price (which is adjusted because auto_adjust=True)
                try:
                    prices_df = data['Close'].copy()
                except KeyError:
                    # Fallback if 'Close' isn't top level (rare with auto_adjust)
                    return pd.DataFrame(), pd.Series()
            
            # Scenario B: Single Ticker
            else:
                # If only 1 symbol was requested (e.g. 1 stock + benchmark was duplicate or failed)
                # The columns are just Open, Close, etc.
                # We need to figure out WHICH ticker it returned.
                # Typically yfinance doesn't return the ticker name in columns for single downloads.
                # In this edge case, if we requested multiple and got single, it's ambiguous.
                # However, usually we request >1 (Portfolio + Benchmark).
                
                # If we requested 1 ticker total (e.g. portfolio of 1 stock and benchmark is same?)
                # This is rare. Let's try to parse 'Close'.
                if 'Close' in data.columns:
                    # We assume this data belongs to the single symbol requested
                    prices_df[all_symbols[0]] = data['Close']

            # Clean up data: Drop columns that are all NaN (failed downloads)
            prices_df = prices_df.dropna(axis=1, how='all')
            prices_df = prices_df.dropna(how='all') # Drop days with no data

            if prices_df.empty:
                return pd.DataFrame(), pd.Series()

            # Separate Benchmark
            if benchmark_ticker in prices_df.columns:
                benchmark = prices_df[benchmark_ticker].copy()
                # Drop benchmark from portfolio, but keep other stocks
                # Use errors='ignore' in case benchmark is not in columns (failed download)
                portfolio_prices = prices_df.drop(columns=[benchmark_ticker], errors='ignore')
            else:
                # Benchmark download failed
                benchmark = pd.Series(dtype='float64')
                portfolio_prices = prices_df

            # Ensure we still have portfolio stocks
            if portfolio_prices.empty:
                return pd.DataFrame(), pd.Series()

            return portfolio_prices, benchmark

        except Exception as e:
            print(f"Backtest Fetch Error: {e}")
            return pd.DataFrame(), pd.Series()

    def run_simulation(self, prices, initial_capital=10000):
        """
        Simulates an equal-weighted Buy-and-Hold strategy.
        """
        if prices.empty: return None
        
        # 1. Normalize prices to start at 100 (Indexed Performance)
        # Forward fill to handle small gaps in data
        prices_filled = prices.ffill().bfill()
        normalized = prices_filled / prices_filled.iloc[0] * 100
        
        # 2. Calculate Portfolio Value (Equal Weight)
        # We assume we put equal money into each stock at start
        portfolio_curve = normalized.mean(axis=1)
        
        # 3. Calculate Drawdown
        rolling_max = portfolio_curve.cummax()
        drawdown = (portfolio_curve - rolling_max) / rolling_max
        
        # 4. Statistics
        start_val = portfolio_curve.iloc[0]
        end_val = portfolio_curve.iloc[-1]
        
        # Total Return
        total_return = (end_val / start_val) - 1
        
        # CAGR (Annualized)
        days = (portfolio_curve.index[-1] - portfolio_curve.index[0]).days
        years = days / 365.25
        if years > 0:
            cagr = (end_val / start_val) ** (1 / years) - 1
        else:
            cagr = 0
            
        # Max Drawdown
        max_dd = drawdown.min()
        
        # Sharpe Ratio (assuming risk-free rate ~0 for simplicity or relative comparison)
        daily_returns = portfolio_curve.pct_change()
        if daily_returns.std() > 0:
            sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
        else:
            sharpe = 0
        
        return {
            'curve': portfolio_curve,
            'drawdown': drawdown,
            'total_return': total_return,
            'cagr': cagr,
            'max_drawdown': max_dd,
            'sharpe_ratio': sharpe,
            'individual_normalized': normalized
        }