# Analyse de 100 Actions Américaines avec Modèles LSTM

## 🎯 Objectif

Analyser 100 actions américaines du S&P 500 en utilisant des modèles LSTM (Long Short-Term Memory) pour prédire les mouvements de prix et comparer deux approches :

1. **Modèle Global** : Un seul modèle LSTM entraîné sur toutes les actions
2. **Modèles par Secteur** : Un modèle LSTM par secteur (11 modèles au total)

L'objectif est de comparer la **performance ajustée au risque** entre ces deux approches.

## 📊 Univers d'Investissement : 100 Actions Américaines (S&P 500)

### Justification du choix :

#### 1. **Diversification Sectorielle Maximale**
- **11 secteurs différents** représentés (GICS: Global Industry Classification Standard)
- **~9-10 actions par secteur** en moyenne
- Réduction du risque spécifique par diversification
- Représentation fidèle du marché américain

#### 2. **Architecture Améliorée : Modèle par Secteur**
- Un modèle LSTM par secteur (au lieu d'un par action)
- Entraînement sur toutes les actions du secteur = plus de données
- Capture les patterns communs au secteur
- Réduction du nombre de modèles (11 au lieu de 100)
- Meilleure généralisation grâce à plus de données d'entraînement

#### 3. **Liquidité et Capitalisation**
- Toutes les actions sont des **grandes capitalisations** (large-cap)
- Volume de trading élevé = exécution facile
- Données historiques complètes et fiables

#### 4. **Représentativité du Marché**
- Actions issues du **S&P 500** (indice de référence)
- Poids significatifs dans l'économie américaine
- Couverture d'environ 80% de la capitalisation boursière US

## 🔧 Architecture du Projet

```
projet_dauphine_python/
├── main.py                    # Point d'entrée principal
├── src/
│   ├── __init__.py
│   ├── fetch_data.py          # Téléchargement des données
│   ├── data_processing.py      # Traitement et préparation des données
│   ├── strategy.py            # Modèles LSTM et stratégies
│   ├── database.py            # Gestion de la base de données
│   └── visualization.py       # Génération des graphiques
├── outs/                      # Tous les fichiers de sortie
│   ├── stock_analysis.db
│   ├── results_sector.csv
│   ├── results_global.csv
│   └── graphique_*.png
├── notebook/
│   └── analyse_stocks_lstm.ipynb  # Documentation et recherches
└── requirements.txt
```

## 🔧 Modèle Utilisé : LSTM (Long Short-Term Memory)

### Architecture :
- **LSTM(64)** → Dropout(0.2) → **LSTM(32)** → Dropout(0.2) → **Dense(32)** → Dropout(0.2) → **Dense(1)**
- Classification binaire : Hausse (1) ou Baisse (0)
- **Fenêtre temporelle** : 20 jours de données pour prédire le jour suivant
- **Méthode** : ROLLING WINDOW (fenêtre glissante) - 252 jours train, 63 jours test

### Caractéristiques techniques :
- **Données d'entrée** : 20 jours de rendements consécutifs
- **Prédiction** : Direction du mouvement (hausse/baisse) du jour suivant
- **Entraînement** : 5 époques avec validation split (10%)
- **Optimiseur** : Adam
- **Perte** : Binary cross-entropy

### Avantages du LSTM :
- **Mémoire à long terme** : Capture les dépendances temporelles complexes
- **Séquences temporelles** : Modèle adapté aux données séquentielles
- **Architecture académique** : Basé sur les travaux de Hochreiter & Schmidhuber (1997)

## 📈 Métriques Calculées

### Métriques de base :
- **Précision** : Pourcentage de prédictions correctes
- **Rendement stratégie** : Performance cumulée du modèle LSTM
- **Rendement Buy & Hold** : Performance cumulée d'achat et conservation
- **Performance relative** : Surperformance de la stratégie vs Buy & Hold

### Métriques ajustées au risque :
- **Sharpe Ratio** : Rendement ajusté à la volatilité
- **Sortino Ratio** : Rendement ajusté au risque de baisse uniquement
- **Maximum Drawdown** : Perte maximale observée
- **Volatilité annualisée** : Mesure du risque

## 🚀 Utilisation

### 1. Installation des dépendances
```bash
pip install -r requirements.txt
```

### 2. Lancer l'analyse complète
```bash
python main.py
```

L'analyse va :
1. Télécharger les données de 100 actions en parallèle
2. Entraîner le modèle global sur toutes les actions
3. Entraîner 11 modèles par secteur
4. Comparer les performances ajustées au risque
5. Générer 5 graphiques PNG dans `outs/`

### 3. Explorer les résultats
```bash
python explore_database.py
```

## 📁 Fichiers Générés (dans `outs/`)

- **stock_analysis.db** - Base de données SQLite avec tous les résultats
- **results_sector.csv** - Résultats détaillés du modèle par secteur
- **results_global.csv** - Résultats détaillés du modèle global
- **graphique_1_comparaison_performances.png** - Comparaison des performances moyennes
- **graphique_2_distribution_sharpe.png** - Distribution des Sharpe Ratios
- **graphique_3_top10_performances.png** - Top 10 actions par performance
- **graphique_4_performance_vs_risque.png** - Scatter plot Performance vs Sharpe Ratio
- **graphique_5_metriques_risque.png** - Comparaison des métriques ajustées au risque

## 🗄️ Structure de la Base de Données

### Table `stock_results` (modèles par secteur)
```sql
CREATE TABLE stock_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    name TEXT NOT NULL,
    sector TEXT NOT NULL,
    model_type TEXT NOT NULL DEFAULT 'sector',
    accuracy REAL NOT NULL,
    strategy_return REAL NOT NULL,
    buy_hold_return REAL NOT NULL,
    performance REAL NOT NULL,
    sharpe_ratio REAL,
    sortino_ratio REAL,
    max_drawdown REAL,
    volatility REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Table `global_model_results` (modèle global)
Même structure que `stock_results` avec `model_type='global'`

### Table `model_comparison` (métriques agrégées)
```sql
CREATE TABLE model_comparison (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_type TEXT NOT NULL,
    total_stocks INTEGER NOT NULL,
    avg_accuracy REAL NOT NULL,
    avg_strategy_return REAL NOT NULL,
    avg_buy_hold_return REAL NOT NULL,
    avg_performance REAL NOT NULL,
    avg_sharpe_ratio REAL,
    avg_sortino_ratio REAL,
    avg_max_drawdown REAL,
    avg_volatility REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## 📊 Notebook Jupyter

Le notebook `notebook/analyse_stocks_lstm.ipynb` contient :
- La problématique de recherche
- Les justifications des choix techniques
- Les hypothèses testées
- La documentation des résultats

## 🎯 Points Clés

### Modèle LSTM
1. **Architecture académique** : Inspirée des thèses sur la prédiction de cours boursiers
2. **Mémoire temporelle** : Capture les dépendances longues et courtes termes
3. **Régularisation** : Dropout pour éviter le surapprentissage
4. **Rolling Window** : Méthode plus réaliste que extending window

### Comparaison des Modèles
1. **Modèle Global** : Maximum de données, patterns communs
2. **Modèles par Secteur** : Spécialisation, patterns sectoriels
3. **Performance ajustée au risque** : Sharpe Ratio, Sortino Ratio, Max Drawdown

### Visualisations
1. **5 graphiques PNG** : Comparaisons visuelles des performances
2. **Haute résolution** : 300 DPI pour présentation
3. **Métriques complètes** : Performance et risque visualisés

## 📝 Notes

- Les tables de la base de données sont **vidées à chaque run** pour garantir des résultats frais
- Tous les fichiers de sortie sont sauvegardés dans le dossier `outs/`
- Le notebook Jupyter documente toutes les problématiques et recherches effectuées
