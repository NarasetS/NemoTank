import streamlit as st
import pandas as pd
import numpy as np
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

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "🔎 Ticker Deep Dive",
        "🦈 Nemo Tank",
        "✨ Magic Formula", 
        "🏰 Buffett Quality", 
        "🚀 Lynch Growth", 
        "🏆 Unified Alpha", 
        "🤖 ML Clusters",
        "📊 Explorer"
    ])

    with tab1:
        st.header("🔎 Single Ticker Investigation")
        st.markdown("Select a stock to run a full **360° Forensic Audit** against all our quantitative models.")
        
        # Ticker Selector
        ticker_list = sorted(working_df['ticker'].unique().tolist())
        selected_ticker = st.selectbox("Select Ticker to Investigate", ticker_list)
        
        if selected_ticker:
            stock = working_df[working_df['ticker'] == selected_ticker].iloc[0]
            
            # --- 1. VITAL SIGNS ---
            st.subheader(f"1. Vital Signs: {selected_ticker} ({stock['sector']})")
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Forward P/E", f"{stock.get('forward_pe', 0):.2f}x")
            c2.metric("ROIC (Quality)", f"{stock.get('conventional_roic', 0):.2f}%")
            c3.metric("Earn. Yield (Value)", f"{stock.get('earnings_yield', 0):.2f}%")
            c4.metric("PEG Ratio", f"{stock.get('peg_ratio', 0):.2f}")
            c5.metric("Unified Alpha", f"{stock.get('unified_alpha', 0):.2f}")

            st.divider()

            # --- 2. THE SCORECARD ---
            st.subheader("2. Model Scorecard")
            col_score1, col_score2 = st.columns([1, 1.5])
            
            with col_score1:
                # -- GREENBLATT CALCULATION (Relative Rank) --
                # Calculate ranks dynamically for the current universe to see where this stock sits
                gb_df = working_df[working_df['earnings_yield'] > 0].copy()
                if not gb_df.empty:
                    gb_df['yield_rank'] = gb_df['earnings_yield'].rank(ascending=False)
                    gb_df['roc_rank'] = gb_df['greenblatt_roc'].rank(ascending=False)
                    gb_df['magic_score'] = gb_df['yield_rank'] + gb_df['roc_rank']
                    gb_df['final_rank'] = gb_df['magic_score'].rank(ascending=True)
                    
                    if selected_ticker in gb_df['ticker'].values:
                        s_gb = gb_df[gb_df['ticker'] == selected_ticker].iloc[0]
                        rank_val = int(s_gb['final_rank'])
                        total_count = len(gb_df)
                        # Logic: Top 20% is "Magic", Top 50% is "Average", Bottom is "Poor"
                        if rank_val <= total_count * 0.2: gb_status = "✅ TOP 20%"
                        elif rank_val <= total_count * 0.5: gb_status = "⚠️ AVERAGE"
                        else: gb_status = "❌ BOTTOM 50%"
                        
                        st.write(f"**✨ Magic Formula:** {gb_status}")
                        st.caption(f"Rank: #{rank_val}/{total_count} | Yield: {s_gb['earnings_yield']:.1f}% | ROC: {s_gb['greenblatt_roc']:.1f}%")
                    else:
                        st.write("**✨ Magic Formula:** ❌ N/A")
                        st.caption("Excluded (Negative Yield)")
                else:
                    st.write("**✨ Magic Formula:** ❌ No Data")

                # -- BUFFETT CHECK --
                buffett_pass = (stock.get('conventional_roic', 0) > 15) and (stock.get('debt_ebitda', 10) < 2.5)
                buffett_icon = "✅ PASS" if buffett_pass else "❌ FAIL"
                st.write(f"**🏰 Buffett Quality:** {buffett_icon}")
                st.caption(f"ROIC: {stock.get('conventional_roic', 0):.1f}% (Target >15) | Debt: {stock.get('debt_ebitda', 0):.1f}x (Target <2.5)")
                
                # -- LYNCH CHECK --
                peg = stock.get('peg_ratio', 10)
                if peg < 1.0 and peg > 0: lynch_status = "✅ BUY ZONE"
                elif peg < 2.0: lynch_status = "⚠️ HOLD/FAIR"
                else: lynch_status = "❌ OVERVALUED"
                st.write(f"**🚀 Lynch Growth:** {lynch_status}")
                st.caption(f"PEG Ratio: {peg:.2f} (Target < 1.0)")
                
                # -- SHARK CHECK --
                shark_pass = (stock.get('revenue_growth', 0) > 15) and (stock.get('gross_margins', 0) > 30)
                shark_icon = "✅ DEAL" if shark_pass else "❌ NO DEAL"
                st.write(f"**🦈 Shark Tank:** {shark_icon}")
                st.caption(f"Growth: {stock.get('revenue_growth', 0):.1f}% | Margins: {stock.get('gross_margins', 0):.1f}%")

            with col_score2:
                # --- 3. RADAR CHART (Percentile Rank within Selected Sector) ---
                subset = working_df.copy()
                pct_rank = {}
                metrics_map = {
                    'Value (Yield)': 'earnings_yield',
                    'Quality (ROIC)': 'conventional_roic',
                    'Growth (Rev)': 'revenue_growth',
                    'Safety (Debt)': 'debt_ebitda', # Lower is better, need to invert
                    'Efficiency (PEG)': 'peg_ratio' # Lower is better, need to invert
                }
                
                for label, key in metrics_map.items():
                    if key in subset.columns:
                        # Rank pct=True gives 0 to 1
                        if key in ['debt_ebitda', 'peg_ratio']:
                            # For Debt and PEG, Lower is Better, so we invert the rank
                            subset[f'{key}_rank'] = subset[key].rank(ascending=False, pct=True)
                        else:
                            subset[f'{key}_rank'] = subset[key].rank(ascending=True, pct=True)
                        
                        # Get value for selected stock
                        val = subset[subset['ticker'] == selected_ticker][f'{key}_rank'].iloc[0]
                        pct_rank[label] = val * 100 # Convert to 0-100 score
                
                # Plot Radar
                categories = list(pct_rank.keys())
                values = list(pct_rank.values())
                
                fig_radar = go.Figure()
                fig_radar.add_trace(go.Scatterpolar(
                    r=values,
                    theta=categories,
                    fill='toself',
                    name=selected_ticker
                ))
                fig_radar.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                    showlegend=False,
                    title=f"Relative Strength (vs Selected Universe)",
                    height=350,
                    margin=dict(t=30, b=30, l=40, r=40)
                )
                st.plotly_chart(fig_radar, width="stretch")
                st.caption("*Chart shows percentile rank (0-100) vs peers. Wider area = Better.*")

    with tab2:
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
                st.subheader("Visual Analysis")
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

    with tab3:
        st.header(f"Greenblatt Magic Formula ({market_key})")
        with st.expander("✨ Theory: Value + Quality", expanded=False):
            st.markdown("""
            **"Buy good companies at cheap prices."**
            Joel Greenblatt's formula ranks stocks based on two metrics:
            1.  **Earnings Yield (Value):** How cheap is the stock relative to its earnings?
            2.  **Return on Capital (Quality):** How good is the company at investing its own money?
            
            **Normalized Score:** We've added a 0-100 score where 100 represents the 'Most Magic' stock (Best combined Rank).
            """)
            
        limit = st.slider("Shortlist Limit", 5, 100, 20)
        ranked = engine.get_greenblatt_rank(working_df, limit=limit)
        
        # --- NORMALIZED SCORING ADDITION ---
        if not ranked.empty:
            max_score = ranked['magic_score'].max()
            min_score = ranked['magic_score'].min()
            if max_score != min_score:
                ranked['magic_normalized'] = 100 - ((ranked['magic_score'] - min_score) / (max_score - min_score) * 100)
            else:
                ranked['magic_normalized'] = 100 
        
        st.subheader("Visual Frontier")
        c1, c2 = st.columns(2)
        gb_x = c1.selectbox("X Axis", numeric_cols, index=numeric_cols.index('earnings_yield') if 'earnings_yield' in numeric_cols else 0, key='gb_x')
        gb_y = c2.selectbox("Y Axis", numeric_cols, index=numeric_cols.index('greenblatt_roc') if 'greenblatt_roc' in numeric_cols else 0, key='gb_y')

        st.plotly_chart(px.scatter(ranked, x=gb_x, y=gb_y, text="ticker", color="magic_normalized", 
                                   color_continuous_scale="Viridis",
                                   title=f"Magic Formula: {gb_x} vs {gb_y} (Color=Normalized Score)"), width="stretch")
        
        st.subheader("The 'Magic' Shortlist")
        st.dataframe(ranked[['ticker', 'earnings_yield', 'greenblatt_roc', 'magic_score', 'magic_normalized']].style.format({'magic_normalized': '{:.1f}'}))

    with tab4:
        st.header(f"Buffett 'Wonderful Companies' ({market_key})")
        with st.expander("🏰 Theory: Moats & Fortresses", expanded=False):
            st.markdown("""
            **"Time is the friend of the wonderful company."**
            Warren Buffett looks for **Economic Moats** and **Financial Fortresses**.
            """)
            
        roic_val = st.slider("Min ROIC (%)", 5, 40, 15)
        debt_val = st.slider("Max Debt/EBITDA", 0.5, 5.0, 2.5)
        buffett = engine.get_buffett_leads(working_df, roic_threshold=roic_val, debt_limit=debt_val)
        
        if not buffett.empty:
            st.subheader("Leaderboard Analysis")
            c1, c2 = st.columns(2)
            bf_y = c1.selectbox("Bar Height (Metric)", numeric_cols, index=numeric_cols.index('conventional_roic') if 'conventional_roic' in numeric_cols else 0, key='bf_y')
            bf_c = c2.selectbox("Bar Color (Metric)", numeric_cols, index=numeric_cols.index('debt_ebitda') if 'debt_ebitda' in numeric_cols else 0, key='bf_c')
            
            st.plotly_chart(px.bar(buffett.head(15), x='ticker', y=bf_y, color=bf_c, title=f"Top Quality Leaders: {bf_y} colored by {bf_c}"), width="stretch")
            st.dataframe(buffett[['ticker', 'conventional_roic', 'debt_ebitda', 'forward_pe']])
        else:
            st.warning("No companies match these strict criteria.")

    with tab5:
        st.header(f"Lynch Growth ({market_key})")
        with st.expander("🚀 Theory: Growth at a Reasonable Price (GARP)", expanded=False):
            st.markdown("""**"Stalwarts" vs "Fast Growers":** Target PEG < 1.0.""")
            
        lynch_df = working_df[(working_df['peg_ratio'] > 0) & (working_df['peg_ratio'] < 5)].sort_values('peg_ratio')
        
        st.subheader("Growth Valuation Map")
        c1, c2 = st.columns(2)
        l_x = c1.selectbox("X Axis", numeric_cols, index=numeric_cols.index('forward_pe') if 'forward_pe' in numeric_cols else 0, key='l_x')
        l_y = c2.selectbox("Y Axis", numeric_cols, index=numeric_cols.index('peg_ratio') if 'peg_ratio' in numeric_cols else 0, key='l_y')

        fig_lynch = px.scatter(lynch_df, x=l_x, y=l_y, text="ticker", color="conventional_roic")
        if l_y == 'peg_ratio':
            fig_lynch.add_hline(y=1.0, line_dash="dash", line_color="green", annotation_text="Fair Value (PEG=1)")
        st.plotly_chart(fig_lynch, width="stretch")
        st.dataframe(lynch_df[['ticker', 'peg_ratio', 'forward_pe', 'conventional_roic']].head(50))

    with tab6:
        st.header("🏆 Unified Alpha Leaderboard")
        with st.expander("🏆 Theory: Statistical Z-Scores", expanded=False):
            st.markdown("**The 'Triple Crown' of Investing:** Combining Value, Quality, and Efficiency.")
            
        alpha_df = working_df.sort_values('unified_alpha', ascending=False).head(20)
        
        c1, c2 = st.columns(2)
        a_y = c1.selectbox("Y Axis", numeric_cols, index=numeric_cols.index('unified_alpha') if 'unified_alpha' in numeric_cols else 0, key='a_y')
        a_c = c2.selectbox("Color", numeric_cols, index=numeric_cols.index('unified_alpha') if 'unified_alpha' in numeric_cols else 0, key='a_c')

        st.plotly_chart(px.bar(alpha_df, x='ticker', y=a_y, color=a_c, title="Ensemble Winners"), width="stretch")
        st.dataframe(alpha_df[['ticker', 'unified_alpha', 'earnings_yield', 'conventional_roic', 'peg_ratio']])

    with tab7:
        st.header("🤖 Multi-Factor ML Clusters")
        with st.expander("🤖 Machine Learning Insights", expanded=False):
            st.markdown("Groups stocks based on fundamental similarity.")
            
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

            k = st.slider("Select K Clusters", 2, 6, 4)
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

    with tab8:
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