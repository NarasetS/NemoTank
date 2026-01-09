import pandas as pd
import numpy as np
import json
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

class IntelligenceEngine:
    """
    Global Intelligence layer. Switches between US and Thai market data.
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
            df = pd.DataFrame(json.load(f))
        
        # Calculate Universal Derived Metrics
        df['greenblatt_roc'] = (df['ebit'] / (df['nwc'] + df['nfa']).replace(0, np.nan)) * 100
        df['conventional_roic'] = (df['ebit'] / df['total_assets'].replace(0, np.nan)) * 100
        df['earnings_yield'] = (df['ebit'] / df['enterprise_value'].replace(0, np.nan)) * 100
        df['accruals_ratio'] = (df['net_income'] - df['fcf']) / df['total_assets'].replace(0, np.nan)
        df['debt_ebitda'] = df['total_debt'] / df['ebitda'].replace(0, np.nan)
        df['peg_ratio'] = df['forward_pe'] / df['conventional_roic'].replace(0, np.nan)
        
        # Unified Alpha (Z-Score Ensemble)
        z_df = df.copy()
        z_df['inv_peg'] = 1 / z_df['peg_ratio'].replace(0, np.nan)
        for col in ['earnings_yield', 'conventional_roic', 'inv_peg']:
            z_df[f'z_{col}'] = (z_df[col] - z_df[col].mean()) / z_df[col].std()
        df['unified_alpha'] = z_df[['z_earnings_yield', 'z_conventional_roic', 'z_inv_peg']].mean(axis=1)
        
        return df

    def get_greenblatt_rank(self, df, limit=30):
        temp = df[df['earnings_yield'] > 0].copy()
        temp['yield_rank'] = temp['earnings_yield'].rank(ascending=False)
        temp['quality_rank'] = temp['greenblatt_roc'].rank(ascending=False)
        temp['magic_score'] = temp['yield_rank'] + temp['quality_rank']
        return temp.sort_values('magic_score').head(limit)

    def get_buffett_leads(self, df, roic_threshold=15, debt_limit=2.5):
        return df[(df['conventional_roic'] >= roic_threshold) & (df['debt_ebitda'] <= debt_limit)].sort_values('conventional_roic', ascending=False)

    def find_optimal_k(self, data, max_k=8):
        inertias = []
        K = range(1, min(len(data), max_k + 1))
        for k in K:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            km.fit(data)
            inertias.append(km.inertia_)
        return list(K), inertias

    def perform_clustering(self, df, k=4, feature_list=None):
        if feature_list is None: feature_list = ['forward_pe', 'conventional_roic', 'earnings_yield']
        data = df.dropna(subset=feature_list).copy()
        scaler = StandardScaler()
        scaled = scaler.fit_transform(data[feature_list])
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        data['cluster_id'] = km.fit_predict(scaled)
        return data