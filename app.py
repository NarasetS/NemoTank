import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from analysis_engine import IntelligenceEngine

st.set_page_config(page_title="Nemo Tank", layout="wide")
st.title("🐠 Nemo Tank: Global Quant Terminal")

# --- MARKET SWITCHER ---
market_selection = st.sidebar.radio("Select Market Territory", ["US (Broad Market)", "Thailand (SET Universe)"])
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
    
    # --- INFO SIDEBAR ---
    st.sidebar.divider()
    
    # Extract Scan Date
    scan_date = "Unknown"
    if 'scan_date' in master_df.columns:
        try:
            scan_date = pd.to_datetime(master_df['scan_date']).max().strftime('%Y-%m-%d %H:%M')
        except:
            scan_date = str(master_df['scan_date'].iloc[0])

    st.sidebar.success(f"Market: {market_key}")
    st.sidebar.info(f"📊 Stocks Active: {len(working_df)}\n\n📅 Data Date: {scan_date}")

    # Define standard numeric columns for axis selection
    numeric_cols = [
        'forward_pe', 'peg_ratio', 'earnings_yield', 'conventional_roic', 'greenblatt_roc',
        'revenue_growth', 'gross_margins', 'return_on_equity', 'debt_ebitda', 
        'accruals_ratio', 'unified_alpha', 'market_cap'
    ]
    # Filter to only cols present in df
    numeric_cols = [c for c in numeric_cols if c in working_df.columns]

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "🦈 Shark Tank",
        "✨ Magic Formula", 
        "🏰 Buffett Quality", 
        "🚀 Lynch Growth", 
        "🏆 Unified Alpha", 
        "🤖 ML Clusters",
        "📊 Explorer"
    ])

    with tab1:
        st.header(f"Shark Tank Deals ({market_key})")
        with st.expander("🦈 The Shark Philosophy (Hyper-Growth)", expanded=True):
            st.markdown("""
            **"I'm out if it doesn't scale."**
            
            Sharks investors focus on businesses that can grow rapidly while maintaining profitability. We filter for:
            * **Revenue Growth (>15%):** Proof of product-market fit and expanding market share.
            * **Gross Margins (>30%):** Indicates pricing power and scalability. Low margins mean you're just moving money around.
            * **ROE (>15%):** Efficiency. How well are you using the equity investors gave you?
            
            **Goal:** Find the "Rocket Ships" before they become mainstream giants.
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
                st.subheader("Visual Analysis")
                st.caption("Explore the trade-offs between Growth, Profitability, and Efficiency.")
                c1, c2, c3 = st.columns(3)
                x_axis = c1.selectbox("X Axis", numeric_cols, index=numeric_cols.index('revenue_growth') if 'revenue_growth' in numeric_cols else 0, key='shark_x')
                y_axis = c2.selectbox("Y Axis", numeric_cols, index=numeric_cols.index('gross_margins') if 'gross_margins' in numeric_cols else 0, key='shark_y')
                size_axis = c3.selectbox("Size By", numeric_cols, index=numeric_cols.index('return_on_equity') if 'return_on_equity' in numeric_cols else 0, key='shark_size')

                fig_shark = px.scatter(sharks, x=x_axis, y=y_axis, size=size_axis, 
                                       color="sector", hover_data=['ticker'],
                                       title=f"Shark Tank Matrix: {x_axis} vs {y_axis}")
                st.plotly_chart(fig_shark, width="stretch")
                
                st.dataframe(sharks[['ticker', 'sector', 'revenue_growth', 'gross_margins', 'return_on_equity', 'forward_pe']])
            else:
                st.warning("No deals meet your Shark Tank criteria. Try lowering the thresholds.")

    with tab2:
        st.header(f"Greenblatt Magic Formula ({market_key})")
        with st.expander("✨ Theory: Value + Quality", expanded=False):
            st.markdown("""
            **"Buy good companies at cheap prices."** - Joel Greenblatt
            
            The Magic Formula is a contrarian strategy that ranks stocks based on two combined factors:
            1.  **Earnings Yield (Value):** calculated as EBIT / Enterprise Value. This tells us how much the business earns relative to its purchase price. High yield = Cheap.
            2.  **Return on Capital (Quality):** calculated as EBIT / (Net Working Capital + Net Fixed Assets). This measures how efficiently management uses tangible capital to generate profit.
            
            **The Insight:** Buying high-quality businesses when they are temporarily on sale leads to outsized returns over time.
            """)
            
        limit = st.slider("Shortlist Limit", 5, 100, 20)
        ranked = engine.get_greenblatt_rank(working_df, limit=limit)
        
        st.subheader("Visual Frontier")
        st.caption("The 'Magic Frontier' is the top-right corner. These stocks offer the best combination of high earnings yield (Cheap) and high return on capital (Good).")
        c1, c2 = st.columns(2)
        gb_x = c1.selectbox("X Axis", numeric_cols, index=numeric_cols.index('earnings_yield') if 'earnings_yield' in numeric_cols else 0, key='gb_x')
        gb_y = c2.selectbox("Y Axis", numeric_cols, index=numeric_cols.index('greenblatt_roc') if 'greenblatt_roc' in numeric_cols else 0, key='gb_y')

        st.plotly_chart(px.scatter(ranked, x=gb_x, y=gb_y, text="ticker", color="magic_score", title=f"Magic Formula: {gb_x} vs {gb_y}"), width="stretch")
        
        st.subheader("The 'Magic' Shortlist")
        st.dataframe(ranked[['ticker', 'earnings_yield', 'greenblatt_roc', 'magic_score']])

    with tab3:
        st.header(f"Buffett 'Wonderful Companies' ({market_key})")
        with st.expander("🏰 Theory: Moats & Fortresses", expanded=False):
            st.markdown("""
            **"Time is the friend of the wonderful company, the enemy of the mediocre."**
            
            Warren Buffett doesn't look for 'cigar butts' (cheap bad companies); he looks for **Economic Moats**:
            1.  **High ROIC (>15%):** A persistent high return on capital suggests a durable competitive advantage (Brand, Network Effect, Switching Costs) that competitors cannot erode.
            2.  **Low Debt (<2.5x EBITDA):** A 'Financial Fortress' balance sheet ensures the company can survive recessions and make opportunistic acquisitions when others are struggling.
            """)
            
        roic_val = st.slider("Min ROIC (%)", 5, 40, 15)
        debt_val = st.slider("Max Debt/EBITDA", 0.5, 5.0, 2.5)
        buffett = engine.get_buffett_leads(working_df, roic_threshold=roic_val, debt_limit=debt_val)
        
        if not buffett.empty:
            st.subheader("Leaderboard Analysis")
            st.caption("These companies are the 'Quality Aristocrats' of the market. Prioritize those with consistent bars (High ROIC) and cool colors (Low Debt).")
            c1, c2 = st.columns(2)
            bf_y = c1.selectbox("Bar Height (Metric)", numeric_cols, index=numeric_cols.index('conventional_roic') if 'conventional_roic' in numeric_cols else 0, key='bf_y')
            bf_c = c2.selectbox("Bar Color (Metric)", numeric_cols, index=numeric_cols.index('debt_ebitda') if 'debt_ebitda' in numeric_cols else 0, key='bf_c')
            
            st.plotly_chart(px.bar(buffett.head(15), x='ticker', y=bf_y, color=bf_c, title=f"Top Quality Leaders: {bf_y} colored by {bf_c}"), width="stretch")
            st.dataframe(buffett[['ticker', 'conventional_roic', 'debt_ebitda', 'forward_pe']])
        else:
            st.warning("No companies match these strict criteria.")

    with tab4:
        st.header(f"Lynch Growth ({market_key})")
        with st.expander("🚀 Theory: Growth at a Reasonable Price (GARP)", expanded=False):
            st.markdown("""
            **"Stalwarts" vs "Fast Growers"**
            
            Peter Lynch popularized the **PEG Ratio** (P/E divided by Growth Rate) to solve the value investor's dilemma: "Is this stock expensive, or just growing fast?"
            * **PEG < 1.0 (Undervalued):** You are paying less than 1 unit of price for 1 unit of growth. This is the 'Sweet Spot'.
            * **PEG > 2.0 (Overvalued):** The growth is already priced in (or overpriced).
            
            *Note: We use ROIC/Forward PE as a proxy for the PEG ratio here to emphasize sustainable internal growth.*
            """)
            
        lynch_df = working_df[(working_df['peg_ratio'] > 0) & (working_df['peg_ratio'] < 5)].sort_values('peg_ratio')
        
        st.subheader("Growth Valuation Map")
        st.caption("Look for stocks below the green dashed line (PEG = 1.0). These are growing faster than their valuation multiple implies.")
        c1, c2 = st.columns(2)
        l_x = c1.selectbox("X Axis", numeric_cols, index=numeric_cols.index('forward_pe') if 'forward_pe' in numeric_cols else 0, key='l_x')
        l_y = c2.selectbox("Y Axis", numeric_cols, index=numeric_cols.index('peg_ratio') if 'peg_ratio' in numeric_cols else 0, key='l_y')

        fig_lynch = px.scatter(lynch_df, x=l_x, y=l_y, text="ticker", color="conventional_roic")
        if l_y == 'peg_ratio':
            fig_lynch.add_hline(y=1.0, line_dash="dash", line_color="green", annotation_text="Fair Value (PEG=1)")
        st.plotly_chart(fig_lynch, width="stretch")
        st.dataframe(lynch_df[['ticker', 'peg_ratio', 'forward_pe', 'conventional_roic']].head(50))

    with tab5:
        st.header("🏆 Unified Alpha Leaderboard")
        with st.expander("🏆 Theory: Statistical Z-Scores", expanded=False):
            st.markdown("""
            **The 'Triple Crown' of Investing.**
            
            Why choose between Value, Quality, and Growth? This model standardizes metrics across the entire market using **Z-Scores** (Standard Deviations from the mean).
            * A stock with a **High Unified Alpha** is statistically superior to the market average across ALL three dimensions simultaneously.
            * It identifies the "Perfect Storm" candidates that Greenblatt, Buffett, and Lynch might all agree on.
            """)
            
        alpha_df = working_df.sort_values('unified_alpha', ascending=False).head(20)
        
        c1, c2 = st.columns(2)
        a_y = c1.selectbox("Y Axis", numeric_cols, index=numeric_cols.index('unified_alpha') if 'unified_alpha' in numeric_cols else 0, key='a_y')
        a_c = c2.selectbox("Color", numeric_cols, index=numeric_cols.index('unified_alpha') if 'unified_alpha' in numeric_cols else 0, key='a_c')

        st.plotly_chart(px.bar(alpha_df, x='ticker', y=a_y, color=a_c, title="Ensemble Winners"), width="stretch")
        st.dataframe(alpha_df[['ticker', 'unified_alpha', 'earnings_yield', 'conventional_roic', 'peg_ratio']])

    with tab6:
        st.header("🤖 Multi-Factor ML Clusters")
        with st.expander("🤖 How Machine Learning helps", expanded=True):
            st.markdown("""
            **Uncovering Hidden Structure**
            
            Traditional sectors (Tech, Energy) are often misleading. K-Means Clustering groups stocks based on **Fundamental Behavior**.
            * **The Elbow Method:** Helps us scientifically determine the optimal number of groups by finding where adding more clusters yields diminishing returns.
            * **Cluster Profiles:** We automatically label these groups (e.g., "Elite Tier", "Value Traps") based on their statistical centroids to give you instant context.
            """)
            
        feat_map = {
            'P/E': 'forward_pe', 
            'ROIC': 'conventional_roic', 
            'Yield': 'earnings_yield', 
            'Alpha': 'unified_alpha', 
            'Growth': 'revenue_growth',
            'Accruals Ratio': 'accruals_ratio' # Added per request
        }
        selected_feats = st.multiselect("Clustering Features", list(feat_map.keys()), default=['Alpha', 'P/E', 'ROIC'])
        feat_keys = [feat_map[f] for f in selected_feats]
        
        if len(working_df) > 10 and len(feat_keys) >= 2:
            st.subheader("1. Optimization: The Elbow Method")
            st.caption("Look for the 'Elbow' or bend in the line. This indicates the optimal number of clusters (k) where distinct groups are formed without over-fitting.")
            
            cluster_base = working_df[feat_keys].dropna()
            k_range, inertias = engine.find_optimal_k(cluster_base)
            
            fig_elbow = go.Figure(go.Scatter(x=k_range, y=inertias, mode='lines+markers'))
            fig_elbow.update_layout(
                xaxis_title="Number of Clusters (k)", 
                yaxis_title="Inertia (Sum of Squared Distances)",
                margin=dict(l=20, r=20, t=20, b=20),
                height=300
            )
            st.plotly_chart(fig_elbow, width="stretch")

            k = st.slider("Select K Clusters (Based on Elbow)", 2, 6, 4)
            clustered = engine.perform_clustering(working_df, k=k, feature_list=feat_keys)
            
            st.subheader("2. Segmentation Plot")
            c1, c2 = st.columns(2)
            ml_x = c1.selectbox("X Axis", feat_keys, index=0, key='ml_x')
            ml_y = c2.selectbox("Y Axis", feat_keys, index=1 if len(feat_keys)>1 else 0, key='ml_y')
            
            st.plotly_chart(px.scatter(clustered, x=ml_x, y=ml_y, color="cluster_id", hover_data=['ticker']), width="stretch")
            
            # Automated Inference
            st.subheader("3. Cluster Intelligence")
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
                    
                    with st.expander("View Tickers"):
                        tickers_in_cluster = clustered[clustered['cluster_id'] == i]['ticker'].tolist()
                        st.write(", ".join(tickers_in_cluster))

    with tab7:
        st.header("📊 Master Data Explorer")
        st.markdown("""
        **Full Custom Analysis:** Use the tools below to explore the entire dataset.
        * **Visual Analysis:** Plot any two metrics against each other.
        * **Raw Data:** Sort, filter, and inspect the raw numbers.
        * **Earnings Quality:** Check the `accruals_ratio` column (lower is generally better, indicating cash-backed earnings).
        """)
        
        # Visual Analysis Section for Explorer
        st.subheader("Visual Analysis")
        c1, c2, c3 = st.columns(3)
        exp_x = c1.selectbox("X Axis", numeric_cols, index=numeric_cols.index('market_cap') if 'market_cap' in numeric_cols else 0, key='exp_x')
        exp_y = c2.selectbox("Y Axis", numeric_cols, index=numeric_cols.index('forward_pe') if 'forward_pe' in numeric_cols else 0, key='exp_y')
        exp_c = c3.selectbox("Color By", numeric_cols, index=numeric_cols.index('accruals_ratio') if 'accruals_ratio' in numeric_cols else 0, key='exp_c')
        
        fig_explorer = px.scatter(working_df, x=exp_x, y=exp_y, color=exp_c, hover_data=['ticker', 'sector'], 
                                  title=f"Custom Plot: {exp_x} vs {exp_y}")
        st.plotly_chart(fig_explorer, width="stretch")

        st.subheader("Raw Data Table")
        st.dataframe(working_df)