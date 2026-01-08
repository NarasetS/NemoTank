import pandas as pd
import numpy as np
import json
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

class IntelligenceEngine:
    """
    Enhanced Intelligence layer for 2026 Multi-Factor Quantitative Analysis.
    Integrates Valuation, Quality, Earnings Truth, and Solvency into ML.
    """
    def __init__(self, data_dir="market_data"):
        self.data_dir = data_dir
        self.master_df = self._load_and_process()

    def _load_and_process(self):
        if not os.path.exists(self.data_dir): return pd.DataFrame()
        files = [f for f in os.listdir(self.data_dir) if f.endswith('.json')]
        if not files: return pd.DataFrame()
        
        latest_file = sorted(files)[-1]
        with open(os.path.join(self.data_dir, latest_file), 'r') as f:
            df = pd.DataFrame(json.load(f))
        
        # Derived Metrics
        df['greenblatt_roc'] = (df['ebit'] / (df['nwc'] + df['nfa']).replace(0, np.nan)) * 100
        df['conventional_roic'] = (df['ebit'] / df['total_assets'].replace(0, np.nan)) * 100
        df['earnings_yield'] = (df['ebit'] / df['enterprise_value'].replace(0, np.nan)) * 100
        df['accruals_ratio'] = (df['net_income'] - df['fcf']) / df['total_assets'].replace(0, np.nan)
        df['debt_ebitda'] = df['total_debt'] / df['ebitda'].replace(0, np.nan)
        
        # Clean numeric data
        cols = ['greenblatt_roc', 'conventional_roic', 'earnings_yield', 'accruals_ratio', 'debt_ebitda', 'forward_pe']
        for col in cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
        return df

    def get_filtered_df(self, sectors=None, industries=None):
        df = self.master_df.copy()
        if sectors: df = df[df['sector'].isin(sectors)]
        if industries: df = df[df['industry'].isin(industries)]
        return df

    def get_greenblatt_rank(self, df):
        temp = df[df['earnings_yield'] > 0].copy()
        if temp.empty: return temp
        temp['yield_rank'] = temp['earnings_yield'].rank(ascending=False)
        temp['quality_rank'] = temp['greenblatt_roc'].rank(ascending=False)
        temp['magic_score'] = temp['yield_rank'] + temp['quality_rank']
        return temp.sort_values('magic_score')

    def get_buffett_leads(self, df):
        return df[(df['conventional_roic'] > 15) & (df['debt_ebitda'] < 2.5)].sort_values('conventional_roic', ascending=False)

    def find_optimal_k(self, data, max_k=8):
        """
        Uses the Elbow Method (Inertia) to suggest optimal k.
        """
        inertias = []
        K = range(1, min(len(data), max_k + 1))
        for k in K:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            km.fit(data)
            inertias.append(km.inertia_)
        return list(K), inertias

    def perform_clustering(self, df, k=4, feature_list=None):
        if feature_list is None:
            feature_list = ['forward_pe', 'conventional_roic', 'earnings_yield', 'accruals_ratio']
            
        data = df.dropna(subset=feature_list).copy()
        # Cap outliers
        for feat in feature_list:
            if feat == 'forward_pe': data = data[data[feat] < 100]
            if feat == 'debt_ebitda': data = data[data[feat] < 20]

        if len(data) < k: return data
        
        scaler = StandardScaler()
        scaled = scaler.fit_transform(data[feature_list])
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        data['cluster_id'] = km.fit_predict(scaled)
        return data