import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from analysis_engine import IntelligenceEngine

# --- APP CONFIGURATION & THEME ---
st.set_page_config(
    page_title="Nemo Tank", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Dark Mode / High Contrast
st.markdown("""
    <style>
        .stApp { background-color: #0E1117; color: #FAFAFA; }
        div[data-testid="stMetricValue"] { font-size: 24px; color: #00CC96; }
        h1, h2, h3 { color: #FAFAFA !important; }
        section[data-testid="stSidebar"] { background-color: #262730; }
        th { background-color: #262730 !important; color: #FAFAFA !important; }
    </style>
""", unsafe_allow_html=True)

pio.templates.default = "plotly_dark"

st.title("🐠 Nemo Tank: Global Quant Terminal")

# --- MARKET SWITCHER ---
market_selection = st.sidebar.radio("Select Market Territory", ["US Market", "Thailand (SET)"])
market_key = "US" if "US" in market_selection else "SET"

engine = IntelligenceEngine()
master_df = engine.load_market_data(market=market_key)

if master_df.empty:
    st.error(f"No {market_key} data found. Please run the Stock Data Pipeline first.")
else:
    # --- GLOBAL FILTERS ---
    st.sidebar.header("🎯 Universe Filters")
    
    # 1. US Specific Sub-Filters (Index/Exchange)
    working_df = master_df.copy()
    
    if market_key == "US":
        st.sidebar.subheader("US Market Scope")
        us_scope = st.sidebar.selectbox(
            "Index / Filter", 
            ["Broad Market (All)", "S&P 500", "NASDAQ 100", "NYSE Listed", "NASDAQ Listed"]
        )
        
        # Apply Index Logic
        if us_scope == "S&P 500":
            if 'is_sp500' in working_df.columns:
                working_df = working_df[working_df['is_sp500'] == True]
            else:
                st.sidebar.warning("S&P 500 tags not found in data. Re-run pipeline.")
                
        elif us_scope == "NASDAQ 100":
            if 'is_nasdaq100' in working_df.columns:
                working_df = working_df[working_df['is_nasdaq100'] == True]
            else:
                st.sidebar.warning("Nasdaq 100 tags not found.")
                
        elif us_scope == "NYSE Listed":
            working_df = working_df[working_df['exchange'].isin(['NYQ', 'NYSE'])]
            
        elif us_scope == "NASDAQ Listed":
            working_df = working_df[working_df['exchange'].isin(['NMS', 'NGM', 'NCM', 'NASDAQ'])]

    # 2. Sector Filter
    all_sectors = sorted(working_df['sector'].dropna().unique().tolist())
    selected_sectors = st.sidebar.multiselect("Sectors", all_sectors, default=all_sectors)
    
    # 3. Industry Filter
    sector_subset = working_df[working_df['sector'].isin(selected_sectors)]
    all_industries = sorted(sector_subset['industry'].dropna().unique().tolist())
    selected_industries = st.sidebar.multiselect("Industries (Empty = All)", all_industries, default=[])
    filter_industries = selected_industries if selected_industries else all_industries
    
    # 4. Size Filter
    st.sidebar.subheader("Liquidity & Size")
    min_mcap = st.sidebar.number_input("Min Market Value (Millions)", min_value=0, value=50, step=50)
    
    if 'market_cap' not in working_df.columns:
        working_df['market_cap'] = working_df.get('enterprise_value', 0)

    # --- APPLY FINAL FILTERS ---
    working_df = working_df[
        (working_df['sector'].isin(selected_sectors)) & 
        (working_df['industry'].isin(filter_industries)) &
        (working_df['market_cap'] >= min_mcap * 1_000_000)
    ]
    
    # --- INFO SIDEBAR ---
    st.sidebar.divider()
    scan_date = "Unknown"
    if 'scan_date' in master_df.columns:
        try: scan_date = pd.to_datetime(master_df['scan_date']).max().strftime('%Y-%m-%d %H:%M')
        except: scan_date = str(master_df['scan_date'].iloc[0])

    st.sidebar.success(f"Active Universe: {len(working_df)} Stocks")
    st.sidebar.caption(f"Data Date: {scan_date}")

    # Numeric columns for axes
    numeric_cols = [
        'forward_pe', 'peg_ratio', 'earnings_yield', 'conventional_roic', 'greenblatt_roc',
        'revenue_growth', 'gross_margins', 'return_on_equity', 'debt_ebitda', 
        'accruals_ratio', 'unified_alpha', 'market_cap'
    ]
    numeric_cols = [c for c in numeric_cols if c in working_df.columns]

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "🔎 Ticker Deep Dive",
        "🦈 Shark Tank",
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
        
        ticker_list = sorted(working_df['ticker'].unique().tolist())
        selected_ticker = st.selectbox("Select Ticker", ticker_list)
        
        if selected_ticker:
            stock = working_df[working_df['ticker'] == selected_ticker].iloc[0]
            
            st.subheader(f"1. Vital Signs: {selected_ticker} ({stock['sector']})")
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Forward P/E", f"{stock.get('forward_pe', 0):.2f}x")
            c2.metric("ROIC (Quality)", f"{stock.get('conventional_roic', 0):.2f}%")
            c3.metric("Earn. Yield (Value)", f"{stock.get('earnings_yield', 0):.2f}%")
            c4.metric("PEG Ratio", f"{stock.get('peg_ratio', 0):.2f}")
            c5.metric("Unified Alpha", f"{stock.get('unified_alpha', 0):.2f}")

            st.divider()
            st.subheader("2. Model Scorecard")
            col_score1, col_score2 = st.columns([1, 1.5])
            
            with col_score1:
                # Greenblatt Rank Logic
                gb_df = working_df[working_df['earnings_yield'] > 0].copy()
                if not gb_df.empty:
                    gb_df['yield_rank'] = gb_df['earnings_yield'].rank(ascending=False)
                    gb_df['roc_rank'] = gb_df['greenblatt_roc'].rank(ascending=False)
                    gb_df['magic_score'] = gb_df['yield_rank'] + gb_df['roc_rank']
                    gb_df['final_rank'] = gb_df['magic_score'].rank(ascending=True)
                    
                    if selected_ticker in gb_df['ticker'].values:
                        s_gb = gb_df[gb_df['ticker'] == selected_ticker].iloc[0]
                        rank_val = int(s_gb['final_rank'])
                        total = len(gb_df)
                        if rank_val <= total * 0.2: gb_status = "✅ TOP 20%"
                        elif rank_val <= total * 0.5: gb_status = "⚠️ AVERAGE"
                        else: gb_status = "❌ BOTTOM 50%"
                        st.write(f"**✨ Magic Formula:** {gb_status}")
                        st.caption(f"Rank: #{rank_val}/{total}")
                    else:
                        st.write("**✨ Magic Formula:** ❌ N/A (Neg Yield)")
                
                # Logic Checks
                buffett_pass = (stock.get('conventional_roic', 0) > 15) and (stock.get('debt_ebitda', 10) < 2.5)
                peg = stock.get('peg_ratio', 10)
                shark_pass = (stock.get('revenue_growth', 0) > 15) and (stock.get('gross_margins', 0) > 30)

                st.write(f"**🏰 Buffett Quality:** {'✅ PASS' if buffett_pass else '❌ FAIL'}")
                st.caption(f"ROIC: {stock.get('conventional_roic',0):.1f}% | Debt: {stock.get('debt_ebitda',0):.1f}x")
                
                st.write(f"**🚀 Lynch Growth:** {'✅ BUY' if peg < 1.0 and peg > 0 else '⚠️ HOLD/SELL'}")
                st.caption(f"PEG Ratio: {peg:.2f}")
                
                st.write(f"**🦈 Shark Tank:** {'✅ DEAL' if shark_pass else '❌ NO DEAL'}")
                st.caption(f"Growth: {stock.get('revenue_growth',0):.1f}% | Margin: {stock.get('gross_margins',0):.1f}%")

            with col_score2:
                # Radar Chart
                subset = working_df.copy()
                pct_rank = {}
                metrics_map = {
                    'Value (Yield)': 'earnings_yield',
                    'Quality (ROIC)': 'conventional_roic',
                    'Growth (Rev)': 'revenue_growth',
                    'Safety (Debt)': 'debt_ebitda',
                    'Efficiency (PEG)': 'peg_ratio'
                }
                for label, key in metrics_map.items():
                    if key in subset.columns:
                        ascending = True if key not in ['debt_ebitda', 'peg_ratio'] else False
                        subset[f'{key}_rank'] = subset[key].rank(ascending=ascending, pct=True)
                        val = subset[subset['ticker'] == selected_ticker][f'{key}_rank'].iloc[0]
                        pct_rank[label] = val * 100
                
                fig_radar = go.Figure(go.Scatterpolar(r=list(pct_rank.values()), theta=list(pct_rank.keys()), fill='toself', name=selected_ticker))
                fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=False, title="Percentile Rank vs Peers", height=350, margin=dict(t=30, b=30, l=40, r=40))
                st.plotly_chart(fig_radar, width="stretch")

    with tab2:
        st.header(f"Shark Tank Deals ({market_key})")
        with st.expander("🦈 Philosophy", expanded=True):
            st.write("Explosive Growth + Scalability + Efficiency.")
        
        c1, c2 = st.columns([1, 2])
        with c1:
            min_growth = st.slider("Min Revenue Growth (%)", 0, 100, 15)
            min_margin = st.slider("Min Gross Margin (%)", 0, 80, 30)
            min_roe = st.slider("Min ROE (%)", 0, 50, 15)
            sharks = engine.get_shark_tank_leads(working_df, min_growth, min_margin, min_roe)
            st.metric("Deals Found", len(sharks))
        with c2:
            if not sharks.empty:
                cx, cy, cz = st.columns(3)
                sx = cx.selectbox("X", numeric_cols, index=numeric_cols.index('revenue_growth') if 'revenue_growth' in numeric_cols else 0, key='sx')
                sy = cy.selectbox("Y", numeric_cols, index=numeric_cols.index('gross_margins') if 'gross_margins' in numeric_cols else 0, key='sy')
                ss = cz.selectbox("Size", numeric_cols, index=numeric_cols.index('return_on_equity') if 'return_on_equity' in numeric_cols else 0, key='ss')
                
                fig = px.scatter(sharks, x=sx, y=sy, size=ss, color="sector", hover_data=['ticker'], title="Shark Tank Matrix")
                st.plotly_chart(fig, width="stretch")
                st.dataframe(sharks)
            else:
                st.warning("No deals found.")

    with tab3:
        st.header(f"Greenblatt Magic Formula ({market_key})")
        with st.expander("✨ Philosophy", expanded=False):
            st.write("Good Companies (High ROIC) at Cheap Prices (High Yield).")
        limit = st.slider("Limit", 5, 100, 20)
        ranked = engine.get_greenblatt_rank(working_df, limit=limit)
        
        # Normalized Score
        if not ranked.empty:
            mx, mn = ranked['magic_score'].max(), ranked['magic_score'].min()
            ranked['magic_norm'] = 100 if mx == mn else 100 - ((ranked['magic_score'] - mn) / (mx - mn) * 100)
            
            c1, c2 = st.columns(2)
            gx = c1.selectbox("X", numeric_cols, index=numeric_cols.index('earnings_yield') if 'earnings_yield' in numeric_cols else 0, key='gx')
            gy = c2.selectbox("Y", numeric_cols, index=numeric_cols.index('greenblatt_roc') if 'greenblatt_roc' in numeric_cols else 0, key='gy')
            
            st.plotly_chart(px.scatter(ranked, x=gx, y=gy, text="ticker", color="magic_norm", color_continuous_scale="Viridis", title="Magic Frontier"), width="stretch")
            st.dataframe(ranked)

    with tab4:
        st.header(f"Buffett Quality ({market_key})")
        with st.expander("🏰 Philosophy", expanded=False):
            st.write("Economic Moats (High ROIC) + Financial Safety (Low Debt).")
        roic_val = st.slider("Min ROIC", 5, 40, 15)
        debt_val = st.slider("Max Debt/EBITDA", 0.5, 5.0, 2.5)
        buffett = engine.get_buffett_leads(working_df, roic_threshold=roic_val, debt_limit=debt_val)
        
        if not buffett.empty:
            c1, c2 = st.columns(2)
            by = c1.selectbox("Height", numeric_cols, index=numeric_cols.index('conventional_roic') if 'conventional_roic' in numeric_cols else 0, key='by')
            bc = c2.selectbox("Color", numeric_cols, index=numeric_cols.index('debt_ebitda') if 'debt_ebitda' in numeric_cols else 0, key='bc')
            st.plotly_chart(px.bar(buffett.head(20), x='ticker', y=by, color=bc, title="Quality Leaders"), width="stretch")
            st.dataframe(buffett)
        else: st.warning("No matches.")

    with tab5:
        st.header(f"Lynch Growth ({market_key})")
        with st.expander("🚀 Philosophy", expanded=False):
            st.write("Growth at a Reasonable Price (PEG < 1.0).")
        lynch = working_df[(working_df['peg_ratio'] > 0) & (working_df['peg_ratio'] < 5)].sort_values('peg_ratio')
        
        c1, c2 = st.columns(2)
        lx = c1.selectbox("X", numeric_cols, index=numeric_cols.index('forward_pe') if 'forward_pe' in numeric_cols else 0, key='lx')
        ly = c2.selectbox("Y", numeric_cols, index=numeric_cols.index('peg_ratio') if 'peg_ratio' in numeric_cols else 0, key='ly')
        
        fig = px.scatter(lynch, x=lx, y=ly, text="ticker", color="conventional_roic")
        if ly == 'peg_ratio': fig.add_hline(y=1.0, line_dash="dash", line_color="green")
        st.plotly_chart(fig, width="stretch")
        st.dataframe(lynch.head(50))

    with tab6:
        st.header("🏆 Unified Alpha")
        with st.expander("🏆 Philosophy", expanded=False):
            st.write("Triple Crown: Z-Score average of Value, Quality, and Efficiency.")
        alpha = working_df.sort_values('unified_alpha', ascending=False).head(20)
        st.plotly_chart(px.bar(alpha, x='ticker', y='unified_alpha', color='unified_alpha', title="Ensemble Winners"), width="stretch")
        st.dataframe(alpha)

    with tab7:
        st.header("🤖 ML Clusters")
        with st.expander("🤖 Philosophy", expanded=False):
            st.write("K-Means grouping by fundamental similarity.")
        
        feats = st.multiselect("Features", list(numeric_cols), default=['unified_alpha', 'forward_pe', 'conventional_roic'])
        if len(working_df) > 10 and len(feats) >= 2:
            st.subheader("1. Elbow Method")
            base = working_df[feats].dropna()
            kr, inert = engine.find_optimal_k(base)
            st.plotly_chart(px.line(x=kr, y=inert, title="Elbow Curve"), width="stretch")
            
            k = st.slider("K Clusters", 2, 6, 4)
            clustered = engine.perform_clustering(working_df, k=k, feature_list=feats)
            
            c1, c2 = st.columns(2)
            mx = c1.selectbox("X", feats, index=0, key='mx')
            my = c2.selectbox("Y", feats, index=1 if len(feats)>1 else 0, key='my')
            st.plotly_chart(px.scatter(clustered, x=mx, y=my, color="cluster_id", hover_data=['ticker']), width="stretch")
            
            st.subheader("Cluster Intelligence")
            stats = clustered.groupby('cluster_id')[feats].mean()
            cols = st.columns(k)
            for i in range(k):
                with cols[i]:
                    st.markdown(f"**Cluster {i}**")
                    s = stats.loc[i]
                    for f in feats: st.write(f"{f}: {s[f]:.2f}")
                    with st.expander("Tickers"):
                        st.write(", ".join(clustered[clustered['cluster_id']==i]['ticker'].tolist()))

    with tab8:
        st.header("📊 Explorer")
        c1, c2, c3 = st.columns(3)
        ex = c1.selectbox("X", numeric_cols, index=numeric_cols.index('market_cap') if 'market_cap' in numeric_cols else 0, key='ex')
        ey = c2.selectbox("Y", numeric_cols, index=numeric_cols.index('forward_pe') if 'forward_pe' in numeric_cols else 0, key='ey')
        ec = c3.selectbox("Color", numeric_cols, index=numeric_cols.index('accruals_ratio') if 'accruals_ratio' in numeric_cols else 0, key='ec')
        st.plotly_chart(px.scatter(working_df, x=ex, y=ey, color=ec, hover_data=['ticker']), width="stretch")
        st.dataframe(working_df)