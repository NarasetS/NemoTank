import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from analysis_engine import IntelligenceEngine

st.set_page_config(page_title="2026 Quant Terminal", layout="wide")
st.title("🛡️ 2026 Academic Quant Terminal")

engine = IntelligenceEngine()
master_df = engine.master_df

if master_df.empty:
    st.warning("Please run 'stock_pipeline.py' first.")
else:
    # --- SIDEBAR ---
    st.sidebar.header("🎯 Filters & Parameters")
    all_sectors = sorted(master_df['sector'].dropna().unique().tolist())
    selected_sectors = st.sidebar.multiselect("Sectors", all_sectors, default=all_sectors)
    
    working_df = engine.get_filtered_df(sectors=selected_sectors)
    st.sidebar.divider()

    tab1, tab2, tab3, tab4 = st.tabs(["✨ Magic Formula", "🏰 Buffett Quality", "🤖 ML Clusters", "📊 Explorer"])

    with tab1:
        st.header("Greenblatt Magic Formula")
        ranked = engine.get_greenblatt_rank(working_df)
        st.dataframe(ranked[['ticker', 'earnings_yield', 'greenblatt_roc', 'magic_score']].head(30))
        fig = px.scatter(ranked, x="earnings_yield", y="greenblatt_roc", text="ticker", color="magic_score", title="Yield vs Quality")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.header("Buffett Wonderful Companies")
        buffett = engine.get_buffett_leads(working_df)
        st.table(buffett[['ticker', 'conventional_roic', 'debt_ebitda']].head(15))

    with tab3:
        st.header("Multi-Factor Machine Learning Clusters")
        
        available_features = {
            'Forward P/E': 'forward_pe',
            'Earnings Yield': 'earnings_yield',
            'Conv. ROIC': 'conventional_roic',
            'Accruals Ratio': 'accruals_ratio',
            'Debt/EBITDA': 'debt_ebitda'
        }
        
        selected_feat_labels = st.multiselect("Select ML Features", list(available_features.keys()), default=['Forward P/E', 'Conv. ROIC', 'Accruals Ratio'])
        selected_feat_keys = [available_features[label] for label in selected_feat_labels]

        if len(working_df) > 10 and len(selected_feat_keys) >= 2:
            # 1. Elbow Plot
            st.subheader("1. Optimal K Selection")
            cluster_base = working_df[selected_feat_keys].dropna()
            k_range, inertias = engine.find_optimal_k(cluster_base)
            fig_elbow = go.Figure(go.Scatter(x=k_range, y=inertias, mode='lines+markers'))
            st.plotly_chart(fig_elbow, use_container_width=True)

            k = st.slider("Number of Clusters (k)", 2, 6, 4)
            clustered = engine.perform_clustering(working_df, k=k, feature_list=selected_feat_keys)
            
            # 2. Visualization
            fig_ml = px.scatter(clustered, x=selected_feat_keys[0], y=selected_feat_keys[1], 
                                color="cluster_id", hover_data=['ticker'],
                                title=f"K-Means Visualization ({selected_feat_labels[0]} vs {selected_feat_labels[1]})")
            st.plotly_chart(fig_ml, use_container_width=True)
            
            # 3. Automated Inference Summary
            st.subheader("3. Cluster Inference & Insights")
            cluster_stats = clustered.groupby('cluster_id')[selected_feat_keys].mean()
            
            cols = st.columns(k)
            for i in range(k):
                stats = cluster_stats.loc[i]
                with cols[i]:
                    st.markdown(f"### Cluster {i}")
                    
                    # Heuristic Labels
                    if 'forward_pe' in stats and stats['forward_pe'] < 15 and stats.get('conventional_roic', 0) > 15:
                        label, color = "The 'Magic' Zone (Value+Quality)", "blue"
                    elif 'forward_pe' in stats and stats['forward_pe'] > 30:
                        label, color = "High Growth / Premium", "green"
                    elif 'debt_ebitda' in stats and stats['debt_ebitda'] > 4:
                        label, color = "High Leverage Risk", "red"
                    else:
                        label, color = "Market Neutral", "grey"
                        
                    st.markdown(f"**Profile:** :{color}[{label}]")
                    for feat_label, feat_key in available_features.items():
                        if feat_key in stats:
                            st.write(f"- {feat_label}: **{stats[feat_key]:.2f}**")
                    
                    with st.expander("Member List"):
                        st.write(", ".join(clustered[clustered['cluster_id'] == i]['ticker'].tolist()))
        else:
            st.info("Select at least 2 features and filter a universe larger than 10 stocks.")

    with tab4:
        st.dataframe(working_df)