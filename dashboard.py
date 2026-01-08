import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from analysis_engine import IntelligenceEngine

# --- APP CONFIG ---
st.set_page_config(page_title="2026 Quant Terminal", layout="wide")
st.title("🛡️ 2026 Academic Quant Terminal")

# --- INITIALIZE ENGINE ---
engine = IntelligenceEngine()
master_df = engine.master_df

if master_df.empty:
    st.warning("No data found. Please run 'stock_pipeline.py' first to generate market data.")
else:
    # --- SIDEBAR FILTERS ---
    st.sidebar.header("🎯 Global Filters")
    
    # Sector Multi-select
    all_sectors = sorted(master_df['sector'].dropna().unique().tolist())
    selected_sectors = st.sidebar.multiselect("Select Sectors", all_sectors, default=all_sectors)
    
    # Industry Multi-select (Dependent on selected sectors)
    relevant_industries = sorted(master_df[master_df['sector'].isin(selected_sectors)]['industry'].dropna().unique().tolist())
    selected_industries = st.sidebar.multiselect("Select Industries", relevant_industries, default=[])

    # Apply Filters via Engine
    current_industries = selected_industries if selected_industries else relevant_industries
    working_df = engine.get_filtered_df(sectors=selected_sectors, industries=current_industries)

    st.sidebar.divider()
    st.sidebar.write(f"Showing **{len(working_df)}** of **{len(master_df)}** stocks.")

    # --- TABS ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "✨ Magic Formula", 
        "🏰 Buffett Quality", 
        "🚀 Lynch Growth", 
        "🤖 ML Clusters",
        "📊 Raw Data"
    ])

    with tab1:
        st.header("Greenblatt Magic Formula")
        st.caption("Ranked by Earnings Yield + Greenblatt ROC (EBIT / [NWC + NFA])")
        
        if not working_df.empty:
            ranked = engine.get_greenblatt_rank(working_df)
            if not ranked.empty:
                st.dataframe(ranked[['ticker', 'sector', 'industry', 'earnings_yield', 'greenblatt_roc', 'magic_score']].head(50), use_container_width=True)
                
                fig = px.scatter(ranked, x="earnings_yield", y="greenblatt_roc", text="ticker", 
                                 color="sector", hover_data=['industry'],
                                 title="Yield vs ROC Frontier (Selected Universe)")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No stocks in this selection have positive earnings yield.")
        else:
            st.error("No data available for the selected filters.")

    with tab2:
        st.header("Buffett's 'Wonderful' Companies")
        st.caption("Criteria: Conventional ROIC > 15% & Debt/EBITDA < 2.5x")
        buffett = engine.get_buffett_leads(working_df)
        if not buffett.empty:
            st.dataframe(buffett[['ticker', 'sector', 'industry', 'conventional_roic', 'total_debt', 'ebitda']], use_container_width=True)
        else:
            st.warning("No companies in the current selection meet Buffett's quality criteria.")

    with tab3:
        st.header("Lynch Growth at a Reasonable Price")
        st.caption("Sorted by Forward P/E (Lower is generally cheaper relative to expected earnings)")
        lynch = working_df[working_df['forward_pe'] > 0].sort_values('forward_pe')
        st.dataframe(lynch[['ticker', 'sector', 'industry', 'forward_pe', 'earnings_yield']], use_container_width=True)

    with tab4:
        st.header("Machine Learning Segmentation")
        st.markdown("""
        This module uses **K-Means Clustering** to group stocks based on three dimensions: 
        **Forward P/E**, **Conventional ROIC**, and **Earnings Yield**. 
        This helps identify "peer groups" based on fundamentals rather than just sectors.
        """)
        
        if len(working_df) > 10:
            # Elbow Method Explanation
            st.subheader("1. Optimal Cluster Detection (The Elbow Method)")
            features = ['forward_pe', 'conventional_roic', 'earnings_yield']
            cluster_base = working_df.dropna(subset=features)
            cluster_base = cluster_base[(cluster_base['forward_pe'] > 0) & (cluster_base['forward_pe'] < 100)]
            
            k_range, inertias = engine.find_optimal_k(cluster_base[features])
            elbow_fig = go.Figure(go.Scatter(x=k_range, y=inertias, mode='lines+markers', marker=dict(size=10, color='royalblue')))
            elbow_fig.update_layout(title="Inertia vs. Number of Clusters", xaxis_title="Number of Clusters (k)", yaxis_title="Inertia (Sum of Squared Distances)")
            st.plotly_chart(elbow_fig, use_container_width=True)
            
            st.info("""
            **Inference:** Look for the 'bend' in the curve above. This represents the point where adding more clusters 
            provides diminishing returns in explaining the variance of the data. Usually, k=3 or k=4 is optimal for this dataset.
            """)

            k = st.slider("Select Number of Clusters", 2, 8, 4)
            clustered = engine.perform_clustering(working_df, k=k)
            
            # Cluster Visualization
            st.subheader("2. Segmentation Visualizer")
            fig_ml = px.scatter(clustered, x="forward_pe", y="conventional_roic", 
                                color="cluster_id", symbol="sector",
                                hover_data=['ticker', 'industry', 'earnings_yield'],
                                title=f"K-Means Clusters: Valuation vs Quality (k={k})",
                                labels={"forward_pe": "Forward P/E", "conventional_roic": "ROIC (%)", "cluster_id": "Cluster"})
            st.plotly_chart(fig_ml, use_container_width=True)
            
            # CLUSTER INFERENCE AND SUMMARY
            st.subheader("3. Cluster Characteristics & Insights")
            
            # Calculate stats for each cluster
            cluster_stats = clustered.groupby('cluster_id')[['forward_pe', 'conventional_roic', 'earnings_yield']].mean()
            
            cols = st.columns(k)
            for i in range(k):
                stats = cluster_stats.loc[i]
                with cols[i]:
                    st.markdown(f"### Cluster {i}")
                    
                    # Heuristic inference based on averages
                    if stats['conventional_roic'] > 20 and stats['forward_pe'] > 25:
                        label, color = "High Quality / Premium Growth", "green"
                    elif stats['conventional_roic'] > 15 and stats['forward_pe'] < 15:
                        label, color = "The 'Magic' Zone (Value + Quality)", "blue"
                    elif stats['forward_pe'] < 10:
                        label, color = "Deep Value / Potential Traps", "orange"
                    elif stats['conventional_roic'] < 10:
                        label, color = "Low Quality / Speculative", "red"
                    else:
                        label, color = "Market Average / Neutral", "grey"
                    
                    st.markdown(f"**Profile:** :{color}[{label}]")
                    st.write(f"- Avg P/E: **{stats['forward_pe']:.1f}x**")
                    st.write(f"- Avg ROIC: **{stats['conventional_roic']:.1f}%**")
                    st.write(f"- Avg Yield: **{stats['earnings_yield']:.1f}%**")
                    
                    with st.expander("View Members"):
                        st.write(", ".join(clustered[clustered['cluster_id'] == i]['ticker'].tolist()))
        else:
            st.warning("Need at least 10 stocks in the selection to perform meaningful clustering.")

    with tab5:
        st.header("Master Data Explorer")
        st.dataframe(working_df)