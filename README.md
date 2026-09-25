# 🚀 Financial Data Science Platform

Plateforme complète de finance quantitative développée en Python (Backend) et Streamlit (Frontend). Ce projet permet d'analyser un univers d'investissement multi-actifs, de modéliser l'exposition au marché, de segmenter les actifs par profil de risque et d'évaluer des stratégies de trading actives par Machine Learning.

## 📊 Fonctionnalités Clés

1. **Prétraitement & Feature Engineering :** 
   - Nettoyage et alignement temporel des séries temporelles (gestion des trous via *Forward/Backward Fill*).
   - Standardisation monétaire automatique vers l'Euro à partir de taux de change dynamiques (`yfinance`).
   - Calcul des log-rendements et d'indicateurs techniques avancés (Moyenne Mobile 20j, RSI 14j, Volatilité glissante, Ichimoku Tenkan).

2. **Modèle de Régression Linéaire (Alpha / Beta) :**
   - Évaluation de la sensibilité des actifs face à des indices de référence majeurs (EuroStoxx 50, S&P 500, CAC 40).
   - Analyse de la variance ($R^2$) et visualisation comparative des rendements réels vs prédits.

3. **Segmentation par Clustering (K-Means) :**
   - Regroupement des actifs selon leurs caractéristiques financières intrinsèques (Rendement annualisé, Volatilité, Max Drawdown, Beta).
   - Standardisation des données et identification des profils types (Défensifs, Risqués, etc.) pour optimiser la diversification.

4. **Classification & Stratégie IA (Machine Learning) :**
   - Entraînement de modèles supervisés (*Random Forest*, *Logistic Regression*, *SVM*) pour prédire la direction future des cours.
   - Évaluation rigoureuse sur données de test (*Time Series Split*) à l'aide de la matrice de confusion et du score F1.
   - Backtest comparatif face à la stratégie de référence passive (*Buy & Hold*).

---

## 🛠️ Installation et Lancement

1. **Cloner le dépôt :**
   ```bash
   git clone [https://github.com/votre-nom-d-utilisateur/financial-datascience-platform.git](https://github.com/votre-nom-d-utilisateur/financial-datascience-platform.git)
   cd financial-datascience-platform
2. **Installer les dépendances :**
   ```bash
   pip install -r requirements.txt
3. **Lancer le frontend :**
   ```bash
   streamlit run dashboard.py
