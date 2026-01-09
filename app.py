import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from analysis_engine import IntelligenceEngine

st.set_page_config(page_title="Nemo Tank", layout="wide")
st.title("🐠 Nemo Tank: Global Quant Terminal")

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
    all_sectors = sorted(master_df['sector'].dropna().unique().tolist())
    selected_sectors = st.sidebar.multiselect("Sectors", all_sectors, default=all_sectors)
    
    st.sidebar.subheader("Liquidity & Size")
    min_mcap = st.sidebar.slider("Min Market Value (Millions)", 0, 2000, 50, help="Filters out micro-cap stocks to ensure liquidity.")
    
    # Ensure market_cap exists
    if 'market_cap' not in master_df.columns:
        master_df['market_cap'] = master_df.get('enterprise_value', 0)

    working_df = master_df[
        (master_df['sector'].isin(selected_sectors)) & 
        (master_df['market_cap'] >= min_mcap * 1_000_000)
    ]
    
    st.sidebar.divider()
    st.sidebar.success(f"Market: {market_key} | {len(working_df)} Stocks Active")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🦈 Shark Tank",
        "✨ Magic Formula", 
        "🏰 Buffett Quality", 
        "🚀 Lynch Growth", 
        "🏆 Unified Alpha", 
        "🤖 ML Clusters"
    ])

    with tab1:
        st.header(f"Shark Tank Deals ({market_key})")
        with st.expander("🦈 The Shark Philosophy (Hyper-Growth)", expanded=True):
            st.markdown("""
            **"I'm out if it doesn't scale."**
            Sharks don't just want value; they want **Explosive Growth** and **Scalability**.
            * **Revenue Growth:** Is the product flying off the shelves? (>15%)
            * **Gross Margins:** Is the product profitable to make? (>30%)
            * **ROE:** Are you efficient with equity? (>15%)
            """)
        
        col_shark, col_res = st.columns([1, 2])
        with col_shark:
            st.subheader("Deal Parameters")
            min_growth = st.slider("Min Revenue Growth (%)", 0, 100, 15)
            min_margin = st.slider("Min Gross Margin (%)", 0, 80, 30)
            min_roe = st.slider("Min ROE (%)", 0, 50, 15)
            
            sharks = engine.get_shark_tank_leads(working_df, min_growth, min_margin, min_roe)
            st.metric("Deals Found", len(sharks))
        
        with col_res:
            if not sharks.empty:
                st.dataframe(sharks[['ticker', 'sector', 'revenue_growth', 'gross_margins', 'return_on_equity', 'forward_pe']])
                fig_shark = px.scatter(sharks, x="revenue_growth", y="gross_margins", size="return_on_equity", 
                                       color="sector", hover_data=['ticker'],
                                       title="Shark Tank Matrix: Growth vs Margins (Size = ROE)")
                st.plotly_chart(fig_shark, width="stretch")
            else:
                st.warning("No deals meet your Shark Tank criteria. Try lowering the thresholds.")

    with tab2:
        st.header(f"Greenblatt Magic Formula ({market_key})")
        with st.expander("✨ Theory: Value + Quality", expanded=False):
            st.markdown("""
            **"Buy good companies at cheap prices."**
            Joel Greenblatt's formula ranks stocks based on two metrics:
            1.  **Earnings Yield (Value):** How cheap is the stock relative to its earnings? (EBIT / Enterprise Value)
            2.  **Return on Capital (Quality):** How good is the company at investing its own money? (EBIT / Tangible Assets)
            
            **The Score:** We rank every stock on both metrics. The stock with the lowest combined rank is #1.
            """)
            
        limit = st.slider("Shortlist Limit", 5, 100, 20)
        ranked = engine.get_greenblatt_rank(working_df, limit=limit)
        
        st.subheader("The 'Magic' Shortlist")
        st.dataframe(ranked[['ticker', 'earnings_yield', 'greenblatt_roc', 'magic_score']])
        
        st.subheader("Visual Frontier")
        st.caption("Look for stocks in the top-right corner (High Yield + High Quality).")
        st.plotly_chart(px.scatter(ranked, x="earnings_yield", y="greenblatt_roc", text="ticker", color="magic_score"), width="stretch")

    with tab3:
        st.header(f"Buffett 'Wonderful Companies' ({market_key})")
        with st.expander("🏰 Theory: Moats & Fortresses", expanded=False):
            st.markdown("""
            **"Time is the friend of the wonderful company, the enemy of the mediocre."**
            Warren Buffett looks for **Economic Moats**—advantages that competitors cannot easily copy.
            1.  **High ROIC (>15%):** Indicates a moat (Brand, Network Effect, Cost Advantage).
            2.  **Low Debt (<2.5x):** Ensures the company can survive recessions ("Financial Fortress").
            """)
            
        roic_val = st.slider("Min ROIC (%)", 5, 40, 15)
        debt_val = st.slider("Max Debt/EBITDA", 0.5, 5.0, 2.5)
        buffett = engine.get_buffett_leads(working_df, roic_threshold=roic_val, debt_limit=debt_val)
        
        if not buffett.empty:
            st.dataframe(buffett[['ticker', 'conventional_roic', 'debt_ebitda', 'forward_pe']])
            st.plotly_chart(px.bar(buffett.head(15), x='ticker', y='conventional_roic', color='debt_ebitda', title="Quality Leaders (ROIC vs Debt)"), width="stretch")
        else:
            st.warning("No companies match these strict criteria.")

    with tab4:
        st.header(f"Lynch Growth ({market_key})")
        with st.expander("🚀 Theory: Growth at a Reasonable Price (GARP)", expanded=False):
            st.markdown("""
            **"Stalwarts" vs "Fast Growers"**
            Peter Lynch popularized the **PEG Ratio** (P/E divided by Growth Rate).
            * **PEG < 1.0:** Undervalued (You pay less than 1 unit of price for 1 unit of growth).
            * **PEG > 2.0:** Overvalued.
            
            *Note: In this model, we use ROIC as a proxy for sustainable internal growth efficiency.*
            """)
            
        lynch_df = working_df[(working_df['peg_ratio'] > 0) & (working_df['peg_ratio'] < 5)].sort_values('peg_ratio')
        st.dataframe(lynch_df[['ticker', 'peg_ratio', 'forward_pe', 'conventional_roic']].head(50))
        
        st.subheader("The PEG Frontier")
        st.caption("Stocks BELOW the green line are in the Lynch 'Buy Zone'.")
        fig_lynch = px.scatter(lynch_df, x="forward_pe", y="peg_ratio", text="ticker", color="conventional_roic")
        fig_lynch.add_hline(y=1.0, line_dash="dash", line_color="green", annotation_text="Fair Value (PEG=1)")
        st.plotly_chart(fig_lynch, width="stretch")

    with tab5:
        st.header("🏆 Unified Alpha Leaderboard")
        with st.expander("🏆 Theory: Statistical Z-Scores", expanded=False):
            st.markdown("""
            **The 'Triple Crown' of Investing.**
            Instead of picking one philosophy, this model calculates a **Z-Score** (Standard Deviation) for every stock across:
            1.  **Value** (Earnings Yield)
            2.  **Quality** (ROIC)
            3.  **Efficiency** (PEG)
            
            A high **Unified Alpha** score means the stock is statistically superior to the market average across ALL three dimensions.
            """)
            
        alpha_df = working_df.sort_values('unified_alpha', ascending=False).head(20)
        st.dataframe(alpha_df[['ticker', 'unified_alpha', 'earnings_yield', 'conventional_roic', 'peg_ratio']])
        st.plotly_chart(px.bar(alpha_df, x='ticker', y='unified_alpha', color='unified_alpha', title="Top 20 Ensemble Winners"), width="stretch")

    with tab6:
        st.header("🤖 Multi-Factor ML Clusters")
        with st.expander("🤖 How Machine Learning helps", expanded=False):
            st.markdown("""
            **K-Means Clustering** groups stocks based on their fundamental similarity, not their industry.
            * **Why use this?** To find hidden gems. For example, a "Boring" industrial stock might be statistically identical to a "Hot" tech stock in terms of margins and growth, but priced much lower.
            * **Interpretation:** Check the automated insights below the chart to see what each cluster represents (e.g., "Elite Tier" vs "Value Traps").
            """)
            
        feat_map = {'P/E': 'forward_pe', 'ROIC': 'conventional_roic', 'Yield': 'earnings_yield', 'Alpha': 'unified_alpha', 'Growth': 'revenue_growth'}
        selected_feats = st.multiselect("Clustering Features", list(feat_map.keys()), default=['Alpha', 'P/E', 'ROIC'])
        feat_keys = [feat_map[f] for f in selected_feats]
        
        if len(working_df) > 10 and len(feat_keys) >= 2:
            k = st.slider("K Clusters", 2, 6, 4)
            clustered = engine.perform_clustering(working_df, k=k, feature_list=feat_keys)
            st.plotly_chart(px.scatter(clustered, x=feat_keys[0], y=feat_keys[1], color="cluster_id", hover_data=['ticker']), width="stretch")
            
            # Automated Inference
            st.subheader("Cluster Intelligence")
            stats = clustered.groupby('cluster_id')[feat_keys].mean()
            cols = st.columns(k)
            for i in range(k):
                with cols[i]:
                    st.markdown(f"### Cluster {i}")
                    s = stats.loc[i]
                    # Heuristic Labeling
                    if s.get('unified_alpha', -1) > 0.5: label = "💎 Elite Tier"
                    elif s.get('forward_pe', 50) < 15 and s.get('conventional_roic', 0) > 15: label = "💰 Value & Quality"
                    elif s.get('revenue_growth', 0) > 20: label = "🚀 Hyper Growth"
                    else: label = "😐 Neutral / Laggard"
                    
                    st.write(f"**{label}**")
                    for f in feat_keys: st.write(f"{f}: {s[f]:.2f}")