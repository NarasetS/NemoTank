import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from analysis_engine import IntelligenceEngine

st.set_page_config(page_title="2026 Global Quant Terminal", layout="wide")
st.title("🌎 2026 Global Academic Quant Terminal")

# --- MARKET SWITCHER ---
market_selection = st.sidebar.radio("Select Market Territory", ["US (Broad Market)", "Thailand (SET 100)"])
market_key = "US" if "US" in market_selection else "SET"

engine = IntelligenceEngine()
master_df = engine.load_market_data(market=market_key)

if master_df.empty:
    st.error(f"No {market_key} data found. Please run the Stock Data Pipeline first.")
else:
    # --- GLOBAL FILTERS ---
    st.sidebar.header("🎯 Global Filters")
    
    # Sector Multi-select
    all_sectors = sorted(master_df['sector'].dropna().unique().tolist())
    selected_sectors = st.sidebar.multiselect("Sectors", all_sectors, default=all_sectors)
    
    # Market Value Filter
    st.sidebar.subheader("Liquidity & Size")
    min_mcap = st.sidebar.slider("Min Market Value (Millions)", 0, 2000, 100, help="Greenblatt recommends >$50M or >$1B depending on the strategy version.")
    
    # Data Compatibility Check
    if 'market_cap' not in master_df.columns:
        if 'enterprise_value' in master_df.columns:
            master_df['market_cap'] = master_df['enterprise_value']
            st.sidebar.warning("Note: Using Enterprise Value as proxy (re-run pipeline for true Market Cap).")
        else:
            master_df['market_cap'] = 0

    # Apply Filters
    working_df = master_df[
        (master_df['sector'].isin(selected_sectors)) & 
        (master_df['market_cap'] >= min_mcap * 1_000_000)
    ]
    
    st.sidebar.divider()
    st.sidebar.success(f"Market: {market_key} | {len(working_df)} Stocks Active")
    st.sidebar.info(f"💡 Filtering stocks smaller than {min_mcap}M to ensure liquidity.")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["✨ Magic Formula", "🏰 Buffett Quality", "🚀 Lynch Growth", "🏆 Unified Alpha", "🤖 ML Clusters"])

    with tab1:
        st.header(f"Greenblatt Magic Formula ({market_key})")
        with st.expander("📚 Theory: Value + Quality"):
            st.write("""
            Finds the highest quality companies at the lowest price relative to their earnings.
            **Note:** Per Greenblatt's instructions, we have applied a size filter via the sidebar to exclude illiquid micro-caps.
            """)
        limit = st.slider("Shortlist Limit", 5, 100, 20)
        ranked = engine.get_greenblatt_rank(working_df, limit=limit)
        st.dataframe(ranked[['ticker', 'earnings_yield', 'greenblatt_roc', 'magic_score']])
        st.plotly_chart(px.scatter(ranked, x="earnings_yield", y="greenblatt_roc", text="ticker", color="magic_score"), use_container_width=True)

    with tab2:
        st.header(f"Buffett 'Wonderful Companies' ({market_key})")
        with st.expander("🏰 Theory: Moats & Fortresses"):
            st.write("Identifies businesses with high internal compounding (ROIC) and low leverage.")
        roic_val = st.slider("Min ROIC (%)", 5, 40, 15)
        debt_val = st.slider("Max Debt/EBITDA", 0.5, 5.0, 2.5)
        buffett = engine.get_buffett_leads(working_df, roic_threshold=roic_val, debt_limit=debt_val)
        st.dataframe(buffett[['ticker', 'conventional_roic', 'debt_ebitda', 'forward_pe']])

    with tab3:
        st.header(f"Lynch Growth ({market_key})")
        with st.expander("🚀 Theory: Growth at a Reasonable Price"):
            st.write("Target: PEG < 1.0. Finding companies where growth outpaces the valuation multiple.")
        lynch_df = working_df[(working_df['peg_ratio'] > 0) & (working_df['peg_ratio'] < 5)].sort_values('peg_ratio')
        st.dataframe(lynch_df[['ticker', 'peg_ratio', 'forward_pe', 'conventional_roic']].head(50))
        fig = px.scatter(lynch_df, x="forward_pe", y="peg_ratio", text="ticker", color="conventional_roic")
        fig.add_hline(y=1.0, line_dash="dash", line_color="green")
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.header("🏆 Unified Alpha Leaderboard")
        st.info("Combined Z-Scores across Value, Quality, and Growth Efficiency.")
        alpha_df = working_df.sort_values('unified_alpha', ascending=False).head(20)
        st.dataframe(alpha_df[['ticker', 'unified_alpha', 'earnings_yield', 'conventional_roic', 'peg_ratio']])
        st.plotly_chart(px.bar(alpha_df, x='ticker', y='unified_alpha', color='unified_alpha'), use_container_width=True)

    with tab5:
        st.header("🤖 Multi-Factor ML Clusters")
        feat_map = {'P/E': 'forward_pe', 'ROIC': 'conventional_roic', 'Yield': 'earnings_yield', 'Alpha': 'unified_alpha'}
        selected_feats = st.multiselect("Clustering Features", list(feat_map.keys()), default=['Alpha', 'P/E', 'ROIC'])
        feat_keys = [feat_map[f] for f in selected_feats]
        
        if len(working_df) > 10 and len(feat_keys) >= 2:
            k = st.slider("K Clusters", 2, 6, 4)
            clustered = engine.perform_clustering(working_df, k=k, feature_list=feat_keys)
            st.plotly_chart(px.scatter(clustered, x=feat_keys[0], y=feat_keys[1], color="cluster_id", hover_data=['ticker']), use_container_width=True)
            
            # Automated Inference
            stats = clustered.groupby('cluster_id')[feat_keys].mean()
            cols = st.columns(k)
            for i in range(k):
                with cols[i]:
                    st.markdown(f"### Cluster {i}")
                    s = stats.loc[i]
                    if s.get('unified_alpha', -1) > 0.5: label = "Elite Tier"
                    elif s.get('forward_pe', 50) < 15: label = "Value Tier"
                    else: label = "Neutral"
                    st.write(f"**{label}**")
                    for f in feat_keys: st.write(f"{f}: {s[f]:.2f}")