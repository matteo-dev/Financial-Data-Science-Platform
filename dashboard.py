# Fichier en python/streamlit pour gérer la partie Frontend du projet de la matière Datascience For Finance

# Importation des bibliothèques nécessaires
import streamlit as st # Sert à créer l'interface web interactive du dashboard
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from main import FinancialDataProcessor # Importation de la classe de main.py

# Configuration de la page Streamlit
st.set_page_config(page_title="Dashboard Finance Quantitative", layout="wide")

# Utilisation de st.cache_resource pour éviter de recharger les données à chaque interaction
@st.cache_resource

# Fonction pour initialiser le processeur de données financières et préparer les DataFrames nécessaires
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

    # Titre principal du dashboard
    st.title("🚀 Plateforme d'Analyse Financière")

    # Sidebar pour la configuration de l'affichage et la sélection des actifs
    st.sidebar.header("Configuration")
    categories = ["Toutes"] + list(df_meta['Catégorie'].dropna().unique())
    selected_cat = st.sidebar.selectbox("Filtrer par Catégorie", categories)
    
    # En fonction de la catégorie sélectionnée, on affiche les actifs correspondants dans le menu déroulant
    if selected_cat == "Toutes":
        available_isins = [t for t in df_prices.columns if "^" not in t and "=X" not in t]
    else:
        list_isins = df_meta[df_meta['Catégorie'] == selected_cat]['Code ISIN'].tolist()
        available_isins = [i for i in list_isins if i in df_prices.columns]
    
    # Menu déroulant pour sélectionner un actif (ISIN) à analyser, avec affichage du nom de l'actif à côté du code ISIN
    selected_isin = st.sidebar.selectbox(
        "Choisir un actif (Tabs 1, 2 & 4)", 
        available_isins, 
        format_func=lambda x: f"{name_map.get(x, x)} ({x})"
    )

    # Création des onglets pour organiser les différentes analyses et visualisations
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Analyse Technique", 
        "🧮 Régression (Beta)", 
        "📍 Clustering & Groupes",
        "🤖 Classification & Perfs"
    ])


# --- Onglet 1 : Analyse Technique  ---


    # Affichage du prix de l'actif sélectionné avec les indicateurs techniques (MA20, Ichimoku Tenkan, RSI, Volatilité)
    with tab1:
        st.subheader(f"Indicateurs : {name_map.get(selected_isin, selected_isin)}")
        df_ind = proc.add_technical_indicators(selected_isin)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind['Close'], name='Prix EUR', line=dict(color='blue')))
        
        df_plot_ma = df_ind.dropna(subset=['MA20'])
        df_plot_ichi = df_ind.dropna(subset=['Ichimoku_Tenkan'])
        
        fig.add_trace(go.Scatter(x=df_plot_ma.index, y=df_plot_ma['MA20'], name='MA20', line=dict(dash='dot', color='orange')))
        fig.add_trace(go.Scatter(x=df_plot_ichi.index, y=df_plot_ichi['Ichimoku_Tenkan'], name='Ichimoku Tenkan', line=dict(color='green')))
        
        fig.update_layout(title="Prix et Indicateurs de Suivi", template="plotly_white", hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
        
        # Affichage du RSI sur le même graphique que les lignes de surachat/survente
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("RSI (Momentum 14j)")
            df_rsi = df_ind.dropna(subset=['RSI'])
            fig_rsi = go.Figure(go.Scatter(x=df_rsi.index, y=df_rsi['RSI'], line=dict(color='purple')))
            fig_rsi.add_hline(y=70, line_dash="dot", line_color="red", annotation_text="Surachat (70)")
            fig_rsi.add_hline(y=30, line_dash="dot", line_color="green", annotation_text="Survente (30)")
            fig_rsi.update_layout(template="plotly_white", yaxis=dict(range=[0, 100]))
            st.plotly_chart(fig_rsi, use_container_width=True)
            
        # Affichage de la volatilité sur une période de 20 jours
        with col2:
            st.subheader("Volatilité 20j")
            st.area_chart(df_ind.dropna(subset=['Vol_20j'])['Vol_20j'])


# --- Onglet 2 : Régression  ---

    # Affichage des résultats de la régression linéaire
    with tab2:
        st.subheader("Modèle d'Exposition (Alpha / Beta)")
        benchmarks = st.multiselect("Indices de référence", ['^FCHI', '^GDAXI', '^GSPC', '^STOXX50E'], default=['^STOXX50E', '^GSPC'])
        
        # Si des indices de référence sont sélectionnés, on lance la régression linéaire
        if benchmarks:
            res, df_reg = proc.run_linear_regression(selected_isin, benchmark_tickers=benchmarks)
            if res:
                c1, c2 = st.columns(2)
                c1.metric("R² Score", f"{res['r2_score']:.4f}")
                c2.metric("Alpha", f"{res['alpha']:.6f}")
                
                col_reg1, col_reg2 = st.columns(2)

                # Graphique des Betas pour chaque indice de référence
                with col_reg1:
                    fig_b = px.bar(x=list(res['betas'].keys()), y=list(res['betas'].values()), 
                                   labels={'x': 'Indice', 'y': 'Beta'}, title="Expositions (Betas)",
                                   color=list(res['betas'].values()), color_continuous_scale='Blues')
                    st.plotly_chart(fig_b, use_container_width=True)
                
                # Graphique de dispersion des rendements réels vs prédits pour évaluer la qualité de la régression
                with col_reg2:
                    st.markdown("### Rendements : Réels vs Prédits")
                    fig_scatter = go.Figure()
                    fig_scatter.add_trace(go.Scatter(x=df_reg['Predicted_Return'], y=df_reg[selected_isin], 
                                                     mode='markers', name='Points réels', marker=dict(opacity=0.5)))
                    
                    m_val = min(df_reg['Predicted_Return'].min(), df_reg[selected_isin].min())
                    M_val = max(df_reg['Predicted_Return'].max(), df_reg[selected_isin].max())
                    fig_scatter.add_trace(go.Scatter(x=[m_val, M_val], y=[m_val, M_val], 
                                                     mode='lines', name='Prédiction Parfaite', line=dict(color='red', dash='dash')))
                    
                    fig_scatter.update_layout(template="plotly_white", xaxis_title="Rendement Prédit", yaxis_title="Rendement Réel")
                    st.plotly_chart(fig_scatter, use_container_width=True)


# --- Onglet 3 : Clustering  ---

    # Affichage des résultats du clustering 
    with tab3:
        st.header("Segmentation des Actifs par Profil")
        
        col_clust1, col_clust2 = st.columns([1, 3])

        # Dans la première colonne, on place le slider pour choisir le nombre de clusters (K) et une info pour expliquer ce que fait le clustering
        with col_clust1:
            n_clust = st.slider("Nombre de clusters (K)", 2, 6, 4)
            st.info("K-Means groupe les actifs selon leur rendement, volatilité, Beta et Max Drawdown.")
        
        # Dans la seconde colonne, on affiche les résultats du clustering
        df_clust = proc.run_clustering(n_clusters=n_clust)
        
        # Affichage d'un graphique de dispersion des actifs colorés par cluster pour visualiser la segmentation
        df_p = df_clust.reset_index()
        fig_clust = px.scatter(df_p, x='Volatilite_Ann', y='Rendement_Ann', color='Cluster',
                               hover_name='Ticker', title="Nuage de points : Risque vs Rendement",
                               labels={'Volatilite_Ann': 'Volatilité (Risque)', 'Rendement_Ann': 'Rendement (Performance)'},
                               template="plotly_white")
        fig_clust.update_traces(marker=dict(size=12, line=dict(width=1, color='DarkSlateGrey')))
        st.plotly_chart(fig_clust, use_container_width=True)
        
        # Affichage d'un tableau récapitulatif du profil moyen de chaque cluster 
        st.subheader("📊 Profil moyen par Cluster")
        st.table(df_clust.groupby('Cluster').mean().style.format("{:.2%}"))

        # Affichage d'une section détaillée pour voir la composition de chaque cluster
        st.subheader("🔍 Détail de la composition des groupes")
        
        selected_view_cluster = st.selectbox("Sélectionnez un cluster pour voir les actifs associés :", 
                                             options=sorted(df_clust['Cluster'].unique()))
        
        assets_in_cluster = df_clust[df_clust['Cluster'] == selected_view_cluster].index.tolist()
        
        details = []
        for ticker in assets_in_cluster:
            details.append({
                "Code ISIN": ticker,
                "Nom": name_map.get(ticker, "Inconnu"),
                "Rendement": f"{df_clust.loc[ticker, 'Rendement_Ann']:.2%}",
                "Volatilité": f"{df_clust.loc[ticker, 'Volatilite_Ann']:.2%}",
                "Beta": f"{df_clust.loc[ticker, 'Beta_Marche']:.2f}"
            })
        
        st.dataframe(pd.DataFrame(details), use_container_width=True, hide_index=True)


# --- Onglet 4 : Classification & Évaluation de la Stratégie IA  ---


    # Affichage des résultats de la classification et de l'évaluation de la stratégie générée par l'IA
    with tab4:
        st.header("Classification et Évaluation de la Stratégie IA")
        st.write("Entraînement d'un modèle pour prédire la hausse/baisse et évaluation des performances financières face au marché.")
        
        # Dans la première colonne, on place le menu déroulant pour choisir l'algorithme de classification et le bouton pour lancer le backtest
        col_m1, col_m2 = st.columns([1, 3])
        with col_m1:
            model_choice = st.selectbox("Choix de l'algorithme", ["RandomForest", "Logistique", "SVM"])
            run_btn = st.button("Lancer le Backtest", type="primary")

        # Lorsque l'utilisateur clique sur le bouton, on entraîne le modèle de classification et on affiche les résultats de la modélisation 
        if run_btn:
            with st.spinner("Entraînement du modèle et calcul des métriques de risque..."):
                res_clf = proc.run_classification(selected_isin, model_type=model_choice)

                if res_clf is not None:
                    st.subheader("1. Modélisation (Machine Learning)")
                    c_metric1, c_metric2 = st.columns(2)
                    c_metric1.metric("Précision Globale (Accuracy)", f"{res_clf['accuracy']:.2%}")
                    c_metric2.metric("Score F1", f"{res_clf['f1']:.4f}")

                    # Affichage de la matrice de confusion 
                    col_conf, col_chart = st.columns(2)
                    with col_conf:
                        st.markdown("### Matrice de Confusion")
                        fig_cm = px.imshow(
                            res_clf['matrix'],
                            text_auto=True,
                            x=['Baisse Prédite', 'Hausse Prédite'],
                            y=['Baisse Réelle', 'Hausse Réelle'],
                            color_continuous_scale='Blues'
                        )
                        st.plotly_chart(fig_cm, use_container_width=True)

                    # Affichage de la courbe de rendement de la stratégie IA comparée au Buy & Hold
                    with col_chart:
                        st.markdown("### Courbe de Rendement (Test Set)")
                        ret_test = res_clf['prices_test'].pct_change().dropna()
                        signals = pd.Series(res_clf['y_pred'], index=res_clf['y_test'].index).shift(1).fillna(0)
                        strat_ret = (ret_test * signals.loc[ret_test.index]).dropna()
                        
                        df_bt = pd.DataFrame({
                            'Marché (Buy & Hold)': (1 + ret_test).cumprod(),
                            'Stratégie IA': (1 + strat_ret).cumprod()
                        })
                        st.line_chart(df_bt)

                    # Affichage d'un tableau comparatif des indicateurs de performance et de risque entre la stratégie IA et le Buy & Hold
                    st.divider()
                    st.subheader("2. Évaluation des Performances et des Risques")
                    st.write("Comparatif strict entre la stratégie générée par l'IA et la conservation passive de l'actif (Buy & Hold).")
                    
                    s_perf = res_clf['strategy_performance']
                    b_perf = res_clf['buy_and_hold_performance']

                    # Création du tableau comparatif
                    df_perfs = pd.DataFrame({
                        "Indicateur": ["Performance Totale", "Volatilité Annualisée", "Perte Maximale (Max Drawdown)", "Corrélation avec l'Indice EuroStoxx50"],
                        "Stratégie IA": [
                            f"{s_perf['Performance_Totale']:.2%}",
                            f"{s_perf['Volatilite_Annuelle']:.2%}",
                            f"{s_perf['Max_Drawdown']:.2%}",
                            f"{s_perf['Correlation_Benchmark']:.2f}" if s_perf['Correlation_Benchmark'] else "N/A"
                        ],
                        "Marché (Buy & Hold)": [
                            f"{b_perf['Performance_Totale']:.2%}",
                            f"{b_perf['Volatilite_Annuelle']:.2%}",
                            f"{b_perf['Max_Drawdown']:.2%}",
                            "1.00" 
                        ]
                    })
                    st.table(df_perfs.set_index("Indicateur"))

                    # Affichage de la conclusion du modèle de classification
                    st.markdown("### 🧠 Verdict du Modèle")
                    if 'conclusion_text' in res_clf:
                        st.info(res_clf['conclusion_text'])

                else:
                    st.warning("⚠️ L'historique de cet actif est insuffisant pour entraîner le modèle.")

# En cas d'erreur lors de l'exécution du code, on affiche un message d'erreur dans le dashboard
except Exception as e:
    st.error(f"Erreur d'exécution : {e}")
