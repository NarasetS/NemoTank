import pandas as pd
import numpy as np
import json
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

class IntelligenceEngine:
    """
    Global Intelligence layer. 
    Includes Shark Tank Logic for High-Growth/High-Margin discovery.
    """
    def __init__(self, data_dir="market_data"):
        self.data_dir = data_dir

    def load_market_data(self, market='US'):
        """Loads the most recent scan for the specified market."""
        if not os.path.exists(self.data_dir): return pd.DataFrame()
        files = [f for f in os.listdir(self.data_dir) if f.startswith(f"market_{market}")]
        if not files: return pd.DataFrame()
        
        latest_file = sorted(files)[-1]
        with open(os.path.join(self.data_dir, latest_file), 'r') as f:
            data = json.load(f)
            
        if not data: return pd.DataFrame()
            
        df = pd.DataFrame(data)
        
        # --- CRITICAL FIX: Force numeric types BEFORE calculations ---
        # yfinance can return 'None', strings, or NoneType which crashes arithmetic operations
        numeric_cols_to_clean = [
            'ebit', 'nwc', 'nfa', 'total_assets', 'enterprise_value',
            'net_income', 'fcf', 'total_debt', 'ebitda', 'forward_pe',
            'revenue_growth', 'gross_margins', 'operating_margins', 'return_on_equity',
            'market_cap'
        ]
        
        for col in numeric_cols_to_clean:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # --- Calculate Universal Derived Metrics ---
        # 1. Greenblatt ROC
        # Avoid division by zero or NaN
        greenblatt_capital = (df['nwc'] + df['nfa']).replace(0, np.nan)
        df['greenblatt_roc'] = (df['ebit'] / greenblatt_capital) * 100
        
        # 2. Conventional ROIC
        df['conventional_roic'] = (df['ebit'] / df['total_assets'].replace(0, np.nan)) * 100
        
        # 3. Earnings Yield
        df['earnings_yield'] = (df['ebit'] / df['enterprise_value'].replace(0, np.nan)) * 100
        
        # 4. Accruals Ratio
        df['accruals_ratio'] = (df['net_income'] - df['fcf']) / df['total_assets'].replace(0, np.nan)
        
        # 5. Debt/EBITDA
        df['debt_ebitda'] = df['total_debt'] / df['ebitda'].replace(0, np.nan)
        
        # 6. PEG Ratio
        # Ensure we don't divide by zero/NaN/negative for PEG logic usually
        # Here we just want a raw calculation, treating 0 ROIC as NaN to avoid Inf
        df['peg_ratio'] = df['forward_pe'] / df['conventional_roic'].replace(0, np.nan)
        
        # Convert Shark Metrics to Percentage (if not already)
        # Note: yfinance often provides 0.15 for 15%, so we multiply by 100
        shark_cols = ['revenue_growth', 'gross_margins', 'operating_margins', 'return_on_equity']
        for col in shark_cols:
            if col in df.columns:
                # If values are small (like 0.15), multiply by 100. 
                # If they are already 15, logic handles it or we assume ratio format.
                df[col] = df[col].fillna(0) * 100
        
        # Fill NaNs in derived columns to allow for ranking/sorting without crashing
        derived_cols = ['greenblatt_roc', 'conventional_roic', 'earnings_yield', 
                        'accruals_ratio', 'debt_ebitda', 'peg_ratio']
        for col in derived_cols:
            df[col] = df[col].fillna(0)

        # Unified Alpha (Z-Score Ensemble)
        z_df = df.copy()
        # Inverse PEG: We want low PEG, so 1/PEG is better. 
        # Handle 0 PEG (invalid) by replacing with NaN
        z_df['inv_peg'] = 1 / z_df['peg_ratio'].replace(0, np.nan)
        
        for col in ['earnings_yield', 'conventional_roic', 'inv_peg']:
            # Handle standard deviation of 0 (if all values are same)
            std_dev = z_df[col].std()
            if std_dev == 0 or pd.isna(std_dev):
                z_df[f'z_{col}'] = 0
            else:
                z_df[f'z_{col}'] = (z_df[col] - z_df[col].mean()) / std_dev
                
        # Fill Z-score NaNs with 0 (neutral)
        z_cols = ['z_earnings_yield', 'z_conventional_roic', 'z_inv_peg']
        for col in z_cols:
            z_df[col] = z_df[col].fillna(0)
            
        df['unified_alpha'] = z_df[z_cols].mean(axis=1)
        
        return df

    def get_greenblatt_rank(self, df, limit=30):
        temp = df[df['earnings_yield'] > 0].copy()
        temp['yield_rank'] = temp['earnings_yield'].rank(ascending=False)
        temp['quality_rank'] = temp['greenblatt_roc'].rank(ascending=False)
        temp['magic_score'] = temp['yield_rank'] + temp['quality_rank']
        return temp.sort_values('magic_score').head(limit)

    def get_buffett_leads(self, df, roic_threshold=15, debt_limit=2.5):
        return df[(df['conventional_roic'] >= roic_threshold) & (df['debt_ebitda'] <= debt_limit)].sort_values('conventional_roic', ascending=False)

    def get_shark_tank_leads(self, df, min_growth=15, min_gross_margin=30, min_roe=15):
        """
        Shark Tank Logic: High Growth, Scalability (Margins), and Efficient Equity Use.
        """
        return df[
            (df['revenue_growth'] >= min_growth) &
            (df['gross_margins'] >= min_gross_margin) &
            (df['return_on_equity'] >= min_roe)
        ].sort_values('revenue_growth', ascending=False)

    def find_optimal_k(self, data, max_k=8):
        inertias = []
        # Ensure data is numeric
        data = data.select_dtypes(include=[np.number])
        
        # CRITICAL FIX: Replace infinite values with NaN and drop them.
        # This prevents 'ValueError: Input contains infinity'
        data = data.replace([np.inf, -np.inf], np.nan).dropna()
        
        if len(data) < 2: return [], []
        
        # Scale data to ensure distances are calculated correctly (consistent with perform_clustering)
        scaler = StandardScaler()
        try:
            data_scaled = scaler.fit_transform(data)
        except ValueError:
            return [], [] # Handle edge cases
        
        K = range(1, min(len(data), max_k + 1))
        for k in K:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            km.fit(data_scaled)
            inertias.append(km.inertia_)
        return list(K), inertias

    def perform_clustering(self, df, k=4, feature_list=None):
        if feature_list is None: feature_list = ['forward_pe', 'conventional_roic', 'earnings_yield']
        
        # Data cleaning for clustering
        data = df.dropna(subset=feature_list).copy()
        
        # Ensure columns are numeric
        for col in feature_list:
            data[col] = pd.to_numeric(data[col], errors='coerce')
        data = data.dropna(subset=feature_list)

        # Cap outliers if necessary, but scaler usually handles it reasonably well for k-means
        # Just in case of inf
        data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=feature_list)

        if len(data) < k: return data
        
        scaler = StandardScaler()
        scaled = scaler.fit_transform(data[feature_list])
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        data['cluster_id'] = km.fit_predict(scaled)
        return data