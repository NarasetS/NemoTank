import pandas as pd
import numpy as np
import json
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

class IntelligenceEngine:
    """
    Intelligence layer handling distinct Buffett/Greenblatt/Lynch logic
    with enhanced filtering for sectors and industries.
    """
    def __init__(self, data_dir="market_data"):
        self.data_dir = data_dir
        self.master_df = self._load_and_process()

    def _load_and_process(self):
        if not os.path.exists(self.data_dir): return pd.DataFrame()
        files = [f for f in os.listdir(self.data_dir) if f.endswith('.json')]
        if not files: return pd.DataFrame()
        
        # Load the latest scan
        latest_file = sorted(files)[-1]
        with open(os.path.join(self.data_dir, latest_file), 'r') as f:
            df = pd.DataFrame(json.load(f))
        
        # 1. Pure Greenblatt ROC: EBIT / (Net Working Capital + Net Fixed Assets)
        # We use .replace(0, np.nan) to avoid infinity
        df['greenblatt_roc'] = (df['ebit'] / (df['nwc'] + df['nfa']).replace(0, np.nan)) * 100
        
        # 2. Conventional ROIC: EBIT / Total Assets
        df['conventional_roic'] = (df['ebit'] / df['total_assets'].replace(0, np.nan)) * 100
        
        # 3. Earnings Yield: EBIT / EV
        df['earnings_yield'] = (df['ebit'] / df['enterprise_value'].replace(0, np.nan)) * 100
        
        # Clean numeric data for ML and Ranking
        numeric_cols = ['greenblatt_roc', 'conventional_roic', 'earnings_yield', 'forward_pe', 'total_debt', 'ebitda']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
        return df

    def get_filtered_df(self, sectors=None, industries=None):
        """Returns the master dataframe filtered by user selection."""
        df = self.master_df.copy()
        if sectors:
            df = df[df['sector'].isin(sectors)]
        if industries:
            df = df[df['industry'].isin(industries)]
        return df

    def get_greenblatt_rank(self, df):
        """Applies Magic Formula ranking to the provided (filtered) dataframe."""
        temp = df[df['earnings_yield'] > 0].copy()
        if temp.empty: return temp
        
        temp['yield_rank'] = temp['earnings_yield'].rank(ascending=False)
        temp['quality_rank'] = temp['greenblatt_roc'].rank(ascending=False)
        temp['magic_score'] = temp['yield_rank'] + temp['quality_rank']
        return temp.sort_values('magic_score')

    def get_buffett_leads(self, df):
        """Applies Buffett filters to the provided (filtered) dataframe."""
        temp = df.copy()
        temp['debt_coverage'] = temp['total_debt'] / temp['ebitda'].replace(0, np.nan)
        return temp[(temp['conventional_roic'] > 15) & (temp['debt_coverage'] < 2.5)].sort_values('conventional_roic', ascending=False)

    def find_optimal_k(self, data, max_k=8):
        """
        Uses the Elbow Method (Inertia) to suggest optimal k.
        Returns the inertia values for visualization.
        """
        inertias = []
        K = range(1, min(len(data), max_k + 1))
        for k in K:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            km.fit(data)
            inertias.append(km.inertia_)
        return list(K), inertias

    def perform_clustering(self, df, k=4):
        """Clusters the provided dataframe."""
        features = ['forward_pe', 'conventional_roic', 'earnings_yield']
        data = df.dropna(subset=features).copy()
        data = data[(data['forward_pe'] > 0) & (data['forward_pe'] < 150)]
        
        if len(data) < k: return data
        
        scaler = StandardScaler()
        scaled = scaler.fit_transform(data[features])
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        data['cluster_id'] = km.fit_predict(scaled)
        return data