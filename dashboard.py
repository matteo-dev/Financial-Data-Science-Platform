# Python/Streamlit file to manage the Frontend part of the Datascience For Finance project

# Import necessary libraries
import streamlit as st # Used to create the interactive web interface of the dashboard
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from main import FinancialDataProcessor # Import the class from main.py

# Streamlit page configuration
st.set_page_config(page_title="Quantitative Finance Dashboard", layout="wide")

# Use st.cache_resource to avoid reloading data on each interaction
@st.cache_resource

# Function to initialize the financial data processor and prepare necessary DataFrames
def get_processor():
    p = FinancialDataProcessor("UniversInvestissement.xlsb", "Univers-Prix.parquet")
    p.load_and_pivot_data()
    p.fetch_external_data()
    p.convert_to_eur()
    p.compute_log_returns()
    return p

try:
    proc = get_processor()
    df_meta = proc.df_metadata
    df_prices = proc.df_prices
    name_map = df_meta.set_index('Code ISIN')['Nom'].to_dict()

    # Main dashboard title
    st.title("🚀 Financial Analysis Platform")

    # Sidebar for display configuration and asset selection
    st.sidebar.header("Configuration")
    categories = ["All"] + list(df_meta['Catégorie'].dropna().unique())
    selected_cat = st.sidebar.selectbox("Filter by Category", categories)
    
    # Depending on the selected category, display the corresponding assets in the dropdown menu
    if selected_cat == "All":
        available_isins = [t for t in df_prices.columns if "^" not in t and "=X" not in t]
    else:
        list_isins = df_meta[df_meta['Catégorie'] == selected_cat]['Code ISIN'].tolist()
        available_isins = [i for i in list_isins if i in df_prices.columns]
    
    # Dropdown menu to select an asset (ISIN) to analyze, displaying the asset name next to the ISIN code
    selected_isin = st.sidebar.selectbox(
        "Choose an asset (Tabs 1, 2 & 4)", 
        available_isins, 
        format_func=lambda x: f"{name_map.get(x, x)} ({x})"
    )

    # Create tabs to organize different analyses and visualizations
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Technical Analysis", 
        "🧮 Regression (Beta)", 
        "📍 Clustering & Groups",
        "🤖 Classification & Perfs"
    ])


# --- Tab 1: Technical Analysis ---


    # Display the selected asset price with technical indicators (MA20, Ichimoku Tenkan, RSI, Volatility)
    with tab1:
        st.subheader(f"Indicators: {name_map.get(selected_isin, selected_isin)}")
        df_ind = proc.add_technical_indicators(selected_isin)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind['Close'], name='Price EUR', line=dict(color='blue')))
        
        df_plot_ma = df_ind.dropna(subset=['MA20'])
        df_plot_ichi = df_ind.dropna(subset=['Ichimoku_Tenkan'])
        
        fig.add_trace(go.Scatter(x=df_plot_ma.index, y=df_plot_ma['MA20'], name='MA20', line=dict(dash='dot', color='orange')))
        fig.add_trace(go.Scatter(x=df_plot_ichi.index, y=df_plot_ichi['Ichimoku_Tenkan'], name='Ichimoku Tenkan', line=dict(color='green')))
        
        fig.update_layout(title="Price and Trend Indicators", template="plotly_white", hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
        
        # Display RSI on the same graph as overbought/oversold lines
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("RSI (14-day Momentum)")
            df_rsi = df_ind.dropna(subset=['RSI'])
            fig_rsi = go.Figure(go.Scatter(x=df_rsi.index, y=df_rsi['RSI'], line=dict(color='purple')))
            fig_rsi.add_hline(y=70, line_dash="dot", line_color="red", annotation_text="Overbought (70)")
            fig_rsi.add_hline(y=30, line_dash="dot", line_color="green", annotation_text="Oversold (30)")
            fig_rsi.update_layout(template="plotly_white", yaxis=dict(range=[0, 100]))
            st.plotly_chart(fig_rsi, use_container_width=True)
            
        # Display 20-day volatility
        with col2:
            st.subheader("20-day Volatility")
            st.area_chart(df_ind.dropna(subset=['Vol_20j'])['Vol_20j'])


# --- Tab 2: Regression ---

    # Display linear regression results
    with tab2:
        st.subheader("Exposure Model (Alpha / Beta)")
        benchmarks = st.multiselect("Benchmarks", ['^FCHI', '^GDAXI', '^GSPC', '^STOXX50E'], default=['^STOXX50E', '^GSPC'])
        
        # If benchmarks are selected, run the linear regression
        if benchmarks:
            res, df_reg = proc.run_linear_regression(selected_isin, benchmark_tickers=benchmarks)
            if res:
                c1, c2 = st.columns(2)
                c1.metric("R² Score", f"{res['r2_score']:.4f}")
                c2.metric("Alpha", f"{res['alpha']:.6f}")
                
                col_reg1, col_reg2 = st.columns(2)

                # Betas chart for each benchmark
                with col_reg1:
                    fig_b = px.bar(x=list(res['betas'].keys()), y=list(res['betas'].values()), 
                                   labels={'x': 'Benchmark', 'y': 'Beta'}, title="Exposures (Betas)",
                                   color=list(res['betas'].values()), color_continuous_scale='Blues')
                    st.plotly_chart(fig_b, use_container_width=True)
                
                # Scatter plot of real vs predicted returns to evaluate regression quality
                with col_reg2:
                    st.markdown("### Returns: Real vs Predicted")
                    fig_scatter = go.Figure()
                    fig_scatter.add_trace(go.Scatter(x=df_reg['Predicted_Return'], y=df_reg[selected_isin], 
                                                     mode='markers', name='Real points', marker=dict(opacity=0.5)))
                    
                    m_val = min(df_reg['Predicted_Return'].min(), df_reg[selected_isin].min())
                    M_val = max(df_reg['Predicted_Return'].max(), df_reg[selected_isin].max())
                    fig_scatter.add_trace(go.Scatter(x=[m_val, M_val], y=[m_val, M_val], 
                                                     mode='lines', name='Perfect Prediction', line=dict(color='red', dash='dash')))
                    
                    fig_scatter.update_layout(template="plotly_white", xaxis_title="Predicted Return", yaxis_title="Real Return")
                    st.plotly_chart(fig_scatter, use_container_width=True)


# --- Tab 3: Clustering ---

    # Display clustering results
    with tab3:
        st.header("Asset Segmentation by Profile")
        
        col_clust1, col_clust2 = st.columns([1, 3])

        # In the first column, place the slider to choose the number of clusters (K) and info to explain clustering
        with col_clust1:
            n_clust = st.slider("Number of clusters (K)", 2, 6, 4)
            st.info("K-Means groups assets based on their return, volatility, Beta, and Max Drawdown.")
        
        # In the second column, display clustering results
        df_clust = proc.run_clustering(n_clusters=n_clust)
        
        # Display a scatter plot of assets colored by cluster to visualize segmentation
        df_p = df_clust.reset_index()
        fig_clust = px.scatter(df_p, x='Volatilite_Ann', y='Rendement_Ann', color='Cluster',
                               hover_name='Ticker', title="Scatter plot: Risk vs Return",
                               labels={'Volatilite_Ann': 'Volatility (Risk)', 'Rendement_Ann': 'Return (Performance)'},
                               template="plotly_white")
        fig_clust.update_traces(marker=dict(size=12, line=dict(width=1, color='DarkSlateGrey')))
        st.plotly_chart(fig_clust, use_container_width=True)
        
        # Display a summary table of the average profile for each cluster
        st.subheader("📊 Average Profile by Cluster")
        st.table(df_clust.groupby('Cluster').mean().style.format("{:.2%}"))

        # Display a detailed section to see the composition of each cluster
        st.subheader("🔍 Group composition details")
        
        selected_view_cluster = st.selectbox("Select a cluster to view its associated assets:", 
                                             options=sorted(df_clust['Cluster'].unique()))
        
        assets_in_cluster = df_clust[df_clust['Cluster'] == selected_view_cluster].index.tolist()
        
        details = []
        for ticker in assets_in_cluster:
            details.append({
                "Code ISIN": ticker,
                "Name": name_map.get(ticker, "Unknown"),
                "Return": f"{df_clust.loc[ticker, 'Rendement_Ann']:.2%}",
                "Volatility": f"{df_clust.loc[ticker, 'Volatilite_Ann']:.2%}",
                "Beta": f"{df_clust.loc[ticker, 'Beta_Marche']:.2f}"
            })
        
        st.dataframe(pd.DataFrame(details), use_container_width=True, hide_index=True)


# --- Tab 4: Classification & AI Strategy Evaluation ---


    # Display the results of the classification and the evaluation of the AI-generated strategy
    with tab4:
        st.header("Classification and AI Strategy Evaluation")
        st.write("Training a model to predict up/down movements and evaluating financial performance against the market.")
        
        # In the first column, place the dropdown to choose the classification algorithm and the button to launch the backtest
        col_m1, col_m2 = st.columns([1, 3])
        with col_m1:
            model_choice = st.selectbox("Algorithm choice", ["RandomForest", "Logistique", "SVM"])
            run_btn = st.button("Run Backtest", type="primary")

        # When the user clicks the button, train the classification model and display modeling results
        if run_btn:
            with st.spinner("Training the model and calculating risk metrics..."):
                res_clf = proc.run_classification(selected_isin, model_type=model_choice)

                if res_clf is not None:
                    st.subheader("1. Modeling (Machine Learning)")
                    c_metric1, c_metric2 = st.columns(2)
                    c_metric1.metric("Global Accuracy", f"{res_clf['accuracy']:.2%}")
                    c_metric2.metric("F1 Score", f"{res_clf['f1']:.4f}")

                    # Display the confusion matrix
                    col_conf, col_chart = st.columns(2)
                    with col_conf:
                        st.markdown("### Confusion Matrix")
                        fig_cm = px.imshow(
                            res_clf['matrix'],
                            text_auto=True,
                            x=['Predicted Down', 'Predicted Up'],
                            y=['Actual Down', 'Actual Up'],
                            color_continuous_scale='Blues'
                        )
                        st.plotly_chart(fig_cm, use_container_width=True)

                    # Display the AI strategy yield curve compared to Buy & Hold
                    with col_chart:
                        st.markdown("### Yield Curve (Test Set)")
                        ret_test = res_clf['prices_test'].pct_change().dropna()
                        signals = pd.Series(res_clf['y_pred'], index=res_clf['y_test'].index).shift(1).fillna(0)
                        strat_ret = (ret_test * signals.loc[ret_test.index]).dropna()
                        
                        df_bt = pd.DataFrame({
                            'Market (Buy & Hold)': (1 + ret_test).cumprod(),
                            'AI Strategy': (1 + strat_ret).cumprod()
                        })
                        st.line_chart(df_bt)

                    # Display a comparative table of performance and risk indicators between the AI strategy and Buy & Hold
                    st.divider()
                    st.subheader("2. Performance and Risk Evaluation")
                    st.write("Strict comparison between the AI-generated strategy and passive asset holding (Buy & Hold).")
                    
                    s_perf = res_clf['strategy_performance']
                    b_perf = res_clf['buy_and_hold_performance']

                    # Create the comparative table
                    df_perfs = pd.DataFrame({
                        "Indicator": ["Total Performance", "Annualized Volatility", "Maximum Loss (Max Drawdown)", "Correlation with EuroStoxx50 Index"],
                        "AI Strategy": [
                            f"{s_perf['Performance_Totale']:.2%}",
                            f"{s_perf['Volatilite_Annuelle']:.2%}",
                            f"{s_perf['Max_Drawdown']:.2%}",
                            f"{s_perf['Correlation_Benchmark']:.2f}" if s_perf['Correlation_Benchmark'] else "N/A"
                        ],
                        "Market (Buy & Hold)": [
                            f"{b_perf['Performance_Totale']:.2%}",
                            f"{b_perf['Volatilite_Annuelle']:.2%}",
                            f"{b_perf['Max_Drawdown']:.2%}",
                            "1.00" 
                        ]
                    })
                    st.table(df_perfs.set_index("Indicator"))

                    # Display the classification model's conclusion
                    st.markdown("### 🧠 Model Verdict")
                    if 'conclusion_text' in res_clf:
                        st.info(res_clf['conclusion_text'])

                else:
                    st.warning("⚠️ The historical data for this asset is insufficient to train the model.")

# In case of an error during code execution, display an error message in the dashboard
except Exception as e:
    st.error(f"Execution error: {e}")
