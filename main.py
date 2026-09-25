# Fichier en python pour gérer la partie Backend du projet de la matière Datascience For Finance

# Importation des bibliothèques nécessaires
import pandas as pd
import numpy as np
import yfinance as yf # Sert à récupérer des données financières externes (indices, taux de change)
from sklearn.linear_model import LinearRegression, LogisticRegression # Sert à la régression linéaire et à la classification logistique
from sklearn.ensemble import RandomForestClassifier # Sert à la classification "Random Forest"
from sklearn.svm import SVC # Sert à la classification "Support Vector Machine"
from sklearn.cluster import KMeans # Sert au clustering des actifs
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score # Sert à l'évaluation des modèles de classification

# Classe principale pour le traitement des données financières et l'exécution des analyses
class FinancialDataProcessor:

    # Fonction d'initialisation de la classe qui initialise les dataframes et le mapping des devises et les chemins d'accès aux données
    def __init__(self, metadata_path, prices_path):
        self.metadata_path = metadata_path
        self.prices_path = prices_path
        self.df_metadata = None
        self.df_prices = None
        self.df_returns = None
        self.df_external = None
        self.currency_map = {}
        self.df_features = None 
        self.df_clusters = None 

    # Fonction pour charger les données et les pivoter pour avoir les dates en index et les tickers en colonnes
    def load_and_pivot_data(self):

        # Chargement des données à partir des fichiers Excel et Parquet
        self.df_metadata = pd.read_excel(self.metadata_path, engine='pyxlsb')
        df_raw = pd.read_parquet(self.prices_path)

        # Création du mapping ticker -> devise à partir des métadonnées
        self.currency_map = df_raw.drop_duplicates('ticker').set_index('ticker')['currency'].to_dict()
        df_raw['date'] = pd.to_datetime(df_raw['date']).dt.tz_localize(None)
        df_raw = df_raw.drop_duplicates(['date', 'ticker'])

        # Pivot des données pour avoir les dates en index et les tickers en colonnes
        self.df_prices = df_raw.pivot(index='date', columns='ticker', values='close')
        self.df_prices.sort_index(inplace=True)
        return self.df_metadata, self.df_prices

    # Fonction pour récupérer les données financières externes (indices et taux de change)
    def fetch_external_data(self):

        # Récupération des différents taux de change (EURUSD, EURGBP, EURCHF) et des indices boursiers (CAC40, DAX, S&P500, FTSE100, SMI, STOXX50E)
        tickers_to_get = ['EURUSD=X', 'EURGBP=X', 'EURCHF=X', '^FCHI', '^GDAXI', '^GSPC', '^FTSE', '^SSMI', '^STOXX50E']

        # Choix des dates de début et de fin pour la récupération des données, en ajoutant une marge de 7 jours 
        start_date = self.df_prices.index.min() - pd.Timedelta(days=7)
        end_date = self.df_prices.index.max() + pd.Timedelta(days=7)

        # Récupération des données de clôture pour les tickers spécifiés sur la période définie
        data = yf.download(tickers_to_get, start=start_date, end=end_date)['Close']
        data.index = pd.to_datetime(data.index).tz_localize(None)

        # Reindexation des données externes pour les aligner avec les dates de notre dataframe de prix
        self.df_external = data.reindex(self.df_prices.index).ffill().bfill()
        self.df_prices = pd.concat([self.df_prices, self.df_external], axis=1)
        return self.df_prices

    # Fonction pour convertir les prix des actifs dans la devise euro
    def convert_to_eur(self):

        # Création d'une copie du dataframe de prix 
        df_final = self.df_prices.copy()
        for fx in ['EURUSD=X', 'EURGBP=X', 'EURCHF=X']:
            if fx in df_final.columns:
                df_final[fx] = df_final[fx].ffill().bfill()

        # Boucle "IF" pour convertir les prix des actifs dans la devise euro en utilisant le taux de change approprié
        for ticker in df_final.columns:
            if ticker in self.df_external.columns: continue
            currency = self.currency_map.get(ticker)
            if currency == 'USD' and 'EURUSD=X' in df_final.columns:
                df_final[ticker] /= df_final['EURUSD=X']
            elif currency == 'GBP' and 'EURGBP=X' in df_final.columns:
                df_final[ticker] /= df_final['EURGBP=X']
            elif currency == 'CHF' and 'EURCHF=X' in df_final.columns:
                df_final[ticker] /= df_final['EURCHF=X']
        self.df_prices = df_final
        return self.df_prices

    # Fonction pour calculer les rendements logarithmiques 
    def compute_log_returns(self):

        # Remplissage des valeurs manquantes
        df_filled = self.df_prices.ffill().bfill()

        # Calcul des rendements logarithmiques 
        self.df_returns = np.log(df_filled / df_filled.shift(1)).dropna(how='all')
        return self.df_returns

    # Fonction pour ajouter des indicateurs techniques (MA20, RSI, Volatilité sur 20 jours, Performance glissante sur 20 jours, Ichimoku Tenkan)
    def add_technical_indicators(self, ticker):

        # Vérification que le ticker existe dans le dataframe de prix
        if ticker not in self.df_prices.columns: return None
        df = self.df_prices[[ticker]].copy()
        df.columns = ['Close']
        df.dropna(inplace=True)

        # Calcul de la moyenne mobile sur 20 jours (MA20)
        df['MA20'] = df['Close'].rolling(window=20).mean()

        # Calcul de l'indice de force relative (RSI) sur 14 jours
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = np.where(loss == 0, 100, 100 - (100 / (1 + rs)))

        # Calcul de la volatilité sur 20 jours 
        df['Vol_20j'] = df['Close'].pct_change().rolling(window=20).std()

        # Calcul de la performance glissante sur 20 jours
        df['Perf_glissante'] = df['Close'].pct_change(periods=20)

        # Calcul de la ligne Tenkan de l'Ichimoku (moyenne des plus hauts et plus bas sur 9 jours)
        high_9 = df['Close'].rolling(window=9).max()
        low_9 = df['Close'].rolling(window=9).min()
        df['Ichimoku_Tenkan'] = (high_9 + low_9) / 2
        return df

    # Fonction pour exécuter une régression linéaire entre les rendements d'un actif et ceux de plusieurs indices de référence
    def run_linear_regression(self, ticker, benchmark_tickers=['^STOXX50E', '^GSPC']):

        # Vérification que les rendements ont été calculés, sinon les calculer
        if self.df_returns is None: self.compute_log_returns()
        valid_benchmarks = [b for b in benchmark_tickers if b in self.df_returns.columns]
        if not valid_benchmarks or ticker not in self.df_returns.columns: return None, None
        
        # Préparation des données pour la régression linéaire 
        cols = [ticker] + valid_benchmarks
        df_reg = self.df_returns[cols].dropna()
        if len(df_reg) == 0: return None, None
        
        # Exécution de la régression linéaire
        X = df_reg[valid_benchmarks]
        y = df_reg[ticker]
        model = LinearRegression()
        model.fit(X, y)
        
       # Compilation des résultats de la régression linéaire
        results = {
            'ticker': ticker,
            'alpha': model.intercept_,
            'r2_score': model.score(X, y),
            'betas': dict(zip(valid_benchmarks, model.coef_))
        }
        # Ajout des rendements prédits par le modèle au dataframe de régression
        df_reg['Predicted_Return'] = model.predict(X)
        return results, df_reg

    # Fonction pour préparer les caractéristiques utilisées pour le clustering des actifs
    def prepare_clustering_features(self, benchmark_for_beta='^STOXX50E'):

        # Vérification que les rendements ont été calculés
        if self.df_returns is None:
            self.compute_log_returns()

        # Calcul des caractéristiques pour chaque actif
        features = []
        for ticker in self.df_returns.columns:
            if "^" in ticker or "=X" in ticker: continue
                
            # Calcul du rendement annualisé
            ret_ann = self.df_returns[ticker].mean() * 252

            # Calcul de la volatilité annualisée
            vol_ann = self.df_returns[ticker].std() * np.sqrt(252)

            # Calcul du Max Drawdown
            cum_ret = (1 + self.df_returns[ticker]).cumprod()
            rolling_max = cum_ret.cummax()
            drawdown = (cum_ret - rolling_max) / rolling_max
            max_dd = drawdown.min()

            # Calcul du Beta par rapport à un indice de référence   
            beta = 1.0
            if benchmark_for_beta in self.df_returns.columns:
                cov = self.df_returns[[ticker, benchmark_for_beta]].cov().iloc[0, 1]
                var_bench = self.df_returns[benchmark_for_beta].var()
                beta = cov / var_bench if var_bench != 0 else 1.0

            # Compilation des caractéristiques dans une liste de dictionnaires
            features.append({
                'Ticker': ticker,
                'Rendement_Ann': ret_ann,
                'Volatilite_Ann': vol_ann,
                'Max_Drawdown': max_dd,
                'Beta_Marche': beta
            })
          
        self.df_features = pd.DataFrame(features).set_index('Ticker').dropna()
        return self.df_features

    # Fonction pour exécuter le clustering des actifs en utilisant l'algorithme K-Means sur les caractéristiques préparées
    def run_clustering(self, n_clusters=4):

        # Vérification que les caractéristiques pour le clustering ont été préparées
        if self.df_features is None: self.prepare_clustering_features()
        X = self.df_features.copy()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Exécution du clustering K-Means et ajout des labels de cluster au dataframe des caractéristiques
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        X['Cluster'] = kmeans.fit_predict(X_scaled).astype(str)
        self.df_clusters = X
        return self.df_clusters

    # Fonction pour préparer les données de classification
    def prepare_classification_data(self, ticker):

        # Vérification que les rendements ont été calculés
        df = self.add_technical_indicators(ticker)
        if df is None: return None, None, None
        
        # Création de la variable cible "Target" qui indique si le prix de clôture du jour suivant est supérieur au prix de clôture actuel
        df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

        # Création de caractéristiques supplémentaires basées sur les rendements (rendement du jour, rendement du jour précédent)
        df['Ret_L1'] = df['Close'].pct_change()
        df['Ret_L2'] = df['Ret_L1'].shift(1)
        
        # Sélection des colonnes de caractéristiques et de la variable cible
        feature_cols = ['RSI', 'Vol_20j', 'Perf_glissante', 'Ret_L1', 'Ret_L2']
        data = df[feature_cols + ['Target', 'Close']].dropna()
        
        # Séparation des caractéristiques (X) et de la variable cible (y)
        X = data[feature_cols]
        y = data['Target']
        return X, y, data

    # Fonction pour évaluer les performances d'une stratégie de trading 
    def evaluate_performance(self, returns_series, benchmark_series=None):

        # Vérification que la série de rendements n'est pas vide
        if len(returns_series) == 0:
            return None
            
        # Calcul de la performance totale 
        cum_ret = (1 + returns_series).cumprod()
        total_perf = cum_ret.iloc[-1] - 1
        
        # calcul de la volatilité annualisée
        vol_ann = returns_series.std() * np.sqrt(252)
        
        # calcul du Max Drawdown
        rolling_max = cum_ret.cummax()
        drawdown = (cum_ret - rolling_max) / rolling_max
        max_dd = drawdown.min()
        
        # calcul de la corrélation avec un indice de référence 
        corr = None
        if benchmark_series is not None:
            aligned = pd.concat([returns_series, benchmark_series], axis=1).dropna()
            if len(aligned) > 1:
                corr = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])

        # Compilation des métriques de performance dans un dictionnaire        
        return {
            'Performance_Totale': total_perf,
            'Volatilite_Annuelle': vol_ann,
            'Max_Drawdown': max_dd,
            'Correlation_Benchmark': corr
        }

    # Fonction pour exécuter la classification en utilisant un modèle de machine learning
    def run_classification(self, ticker, model_type='RandomForest', benchmark_ticker='^STOXX50E'):

        # Préparation des données de classification pour le ticker spécifié
        X, y, full_data = self.prepare_classification_data(ticker)
        if X is None or len(X) < 50: return None
        
        # Normalisation des caractéristiques 
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), index=X.index, columns=X.columns)
        
        # Séparation des données en ensemble d'entraînement (80%) et de test (20%)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X_scaled.iloc[:split_idx], X_scaled.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Choix du modèle de classification et entraînement du modèle 
        if model_type == 'Logistique':
            model = LogisticRegression(random_state=42)
        elif model_type == 'SVM':
            model = SVC(random_state=42)
        else: 
            model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
            
        # Entraînement du modèle de classification et prédiction
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Calcul des rendements de la période de test 
        ret_test = full_data['Close'].iloc[split_idx:].pct_change().dropna()
        signals = pd.Series(y_pred, index=y_test.index).shift(1).fillna(0)
        
        # Calcul des rendements de la stratégie 
        strat_ret = (ret_test * signals.loc[ret_test.index]).dropna()
        
        # Récupération des rendements de l'indice de référence pour la période de test
        bench_ret = None
        if self.df_returns is not None and benchmark_ticker in self.df_returns.columns:
            bench_ret = self.df_returns[benchmark_ticker].loc[strat_ret.index]
            
        # Evaluation des performances de la stratégie et de la stratégie Buy & Hold sur la période de test
        strat_perf = self.evaluate_performance(strat_ret, bench_ret)
        bh_perf = self.evaluate_performance(ret_test, bench_ret)
        
        # Génération d'un texte de conclusion basé sur les performances de la stratégie par rapport à la stratégie Buy & Hold
        conclusion_text = ""
        if strat_perf and bh_perf:
            s_tot = strat_perf['Performance_Totale']
            b_tot = bh_perf['Performance_Totale']
            s_dd = strat_perf['Max_Drawdown']
            b_dd = bh_perf['Max_Drawdown']
            
            # Logique de génération du texte de conclusion basée sur la comparaison des performances totales et des Max Drawdown
            if s_tot > b_tot and s_dd > b_dd: 
                conclusion_text = " Excellente strategie : L'IA surperforme la strategie Buy & Hold tout en subissant une perte maximale (Max Drawdown) plus faible."
            elif s_tot > b_tot:
                conclusion_text = " Bonne strategie : Le modele genere une performance superieure au marche (Buy & Hold), bien que le risque associe puisse etre different."
            elif s_tot > 0:
                conclusion_text = " Strategie moyenne : Le modele genere du profit, mais sous-performe la simple strategie de conservation (Buy & Hold) sur la periode."
            else:
                conclusion_text = " Mauvaise strategie : Le modele detruit de la valeur sur la periode de test et sous-performe le marche."
        else:
            conclusion_text = "Donnees insuffisantes pour emettre une conclusion."

        # Compilation des résultats de la classification et de l'évaluation financière dans un dictionnaire
        results = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'matrix': confusion_matrix(y_test, y_pred),
            'y_test': y_test,
            'y_pred': y_pred,
            'prices_test': full_data['Close'].iloc[split_idx:],
            'strategy_performance': strat_perf, 
            'buy_and_hold_performance': bh_perf,
            'conclusion_text': conclusion_text 
        }
        return results

# Fonction pour exécuter une série de tests sur les différentes fonctionnalités du backend financier
def run_tests():

    # Affichage d'un message de démarrage des tests
    print("DEMARRAGE DES TESTS DU BACKEND FINANCIER")

    
    # Initialisation de la classe FinancialDataProcessor
    proc = FinancialDataProcessor("UniversInvestissement.xlsb", "Univers-Prix.parquet")
    print("\n Chargement et preparation des donnees...")

    # Test Fonctions load_and_pivot_data, fetch_external_data, convert_to_eur et compute_log_returns
    proc.load_and_pivot_data()
    proc.fetch_external_data()
    proc.convert_to_eur()
    proc.compute_log_returns()
    
    # Choix d'un actif de test parmi les colonnes du dataframe de prix
    test_isin = [c for c in proc.df_prices.columns if "^" not in c and "=X" not in c][0]
    print(f"Actif de test selectionne : {test_isin}")

    # Test de la fonction run_linear_regression pour l'actif de test sélectionné
    print("\n TEST 1 : REGRESSION LINEAIRE")
    try:
        res_reg, _ = proc.run_linear_regression(test_isin, benchmark_tickers=['^STOXX50E'])
        print(f"Rcarre Score : {res_reg['r2_score']:.4f}")
        print(" Regression OK")
    except Exception as e:
        print(f" Erreur Regression : {e}")

    # Test de la fonction run_clustering pour segmenter les actifs en 4 clusters
    print("\nTEST 2 : CLUSTERING K-MEANS ")
    try:
        df_clust = proc.run_clustering(n_clusters=4)
        print(f"Nombre d'actifs segmentes : {len(df_clust)}")
        print(" Clustering OK")
    except Exception as e:
        print(f" Erreur Clustering : {e}")

    # Test de la fonction run_classification pour l'actif de test sélectionné
    print("\nTEST 3 : CLASSIFICATION ET EVALUATION FINANCIERE")
    try:
        res_clf = proc.run_classification(test_isin, model_type='RandomForest')
        print(f"Modele teste : Random Forest")
        print(f"Precision (Accuracy) : {res_clf['accuracy']:.2%}")
        
        print("\nPERFORMANCE DE LA STRATEGIE vs BUY & HOLD (sur donnees Test) :")
        s_perf = res_clf['strategy_performance']
        b_perf = res_clf['buy_and_hold_performance']
        
        print(f"   Performance Totale : Strategie = {s_perf['Performance_Totale']:.2%} | Buy&Hold = {b_perf['Performance_Totale']:.2%}")
        print(f"   Volatilite Ann.    : Strategie = {s_perf['Volatilite_Annuelle']:.2%} | Buy&Hold = {b_perf['Volatilite_Annuelle']:.2%}")
        print(f"   Max Drawdown       : Strategie = {s_perf['Max_Drawdown']:.2%} | Buy&Hold = {b_perf['Max_Drawdown']:.2%}")
        if s_perf['Correlation_Benchmark'] is not None:
            print(f"   Correlation Indice : Strategie = {s_perf['Correlation_Benchmark']:.2f}")
            
        print(f"\nCONCLUSION DE L'IA : {res_clf['conclusion_text']}")
        print(" \nClassification et Evaluation OK")
    except Exception as e:
        print(f" Erreur Classification : {e}")

    print("\nFIN DES TESTS")

# Point d'entrée du script pour exécuter les tests lorsque le script est exécuté directement
if __name__ == "__main__":
    run_tests()