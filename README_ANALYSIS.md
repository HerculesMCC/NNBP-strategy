# Analyse de 10 Actions Américaines avec MLP

Ce projet analyse 10 actions américaines populaires en utilisant des réseaux de neurones MLP (Multi-Layer Perceptron) pour prédire les mouvements de prix et évaluer les performances d'une stratégie de trading.

## 🎯 Objectif

Développer un système d'analyse automatisé qui :
- Télécharge les données de 10 actions américaines populaires
- Entraîne des modèles MLP pour prédire les mouvements de prix
- Compare les performances de la stratégie MLP vs Buy & Hold
- Stocke tous les résultats dans une base de données structurée
- Génère des rapports et visualisations

## 📊 Actions Analysées

| Symbole | Nom | Secteur |
|---------|-----|---------|
| AAPL | Apple Inc. | Technology |
| MSFT | Microsoft Corporation | Technology |
| GOOGL | Alphabet Inc. | Technology |
| AMZN | Amazon.com Inc. | Consumer Discretionary |
| TSLA | Tesla Inc. | Consumer Discretionary |
| META | Meta Platforms Inc. | Technology |
| NVDA | NVIDIA Corporation | Technology |
| JPM | JPMorgan Chase & Co. | Financials |
| JNJ | Johnson & Johnson | Healthcare |
| V | Visa Inc. | Financials |

## 🏗️ Architecture de la Base de Données

### Table `stocks`
```sql
CREATE TABLE stocks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL,
    sector TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Table `predictions`
```sql
CREATE TABLE predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    stock_id INTEGER NOT NULL,
    date TEXT NOT NULL,
    lookback_period INTEGER NOT NULL,
    training_samples INTEGER NOT NULL,
    test_samples INTEGER NOT NULL,
    model_accuracy REAL NOT NULL,
    strategy_return REAL NOT NULL,
    buy_hold_return REAL NOT NULL,
    relative_performance REAL NOT NULL,
    strategy_volatility REAL NOT NULL,
    buy_hold_volatility REAL NOT NULL,
    positions_taken INTEGER NOT NULL,
    total_positions INTEGER NOT NULL,
    model_params TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (stock_id) REFERENCES stocks (id)
);
```

### Table `detailed_metrics`
```sql
CREATE TABLE detailed_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id INTEGER NOT NULL,
    metric_name TEXT NOT NULL,
    metric_value REAL NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (prediction_id) REFERENCES predictions (id)
);
```

## 🚀 Installation et Utilisation

### 1. Installation des dépendances
```bash
pip install -r requirements.txt
```

### 2. Exécution de l'analyse complète
```bash
python demo.py
```

### 3. Exécution étape par étape

#### Analyse des actions
```bash
python multi_stock_analysis.py
```

#### Génération du rapport
```bash
python database_analyzer.py
```

## 📈 Métriques Calculées

### Métriques de Performance
- **Précision du modèle** : Pourcentage de prédictions correctes
- **Rendement stratégie** : Rendement cumulé de la stratégie MLP
- **Rendement buy&hold** : Rendement cumulé d'une stratégie d'achat et conservation
- **Performance relative** : Différence entre stratégie et buy&hold

### Métriques de Risque
- **Volatilité stratégie** : Volatilité annualisée de la stratégie
- **Volatilité buy&hold** : Volatilité annualisée du buy&hold
- **Ratio de positions** : Pourcentage de jours où une position est prise

### Métriques Techniques
- **Période de lookback** : Nombre de jours utilisés pour la prédiction
- **Échantillons d'entraînement** : Nombre d'exemples d'entraînement
- **Échantillons de test** : Nombre d'exemples de test
- **Paramètres du modèle** : Architecture et nombre de paramètres

## 🔧 Configuration

### Paramètres modifiables dans `multi_stock_analysis.py`

```python
# Période d'analyse
start_date = "2018-01-01"
end_date = "2023-01-01"

# Paramètres du modèle
lookback = 5  # Jours de lookback pour la prédiction
epochs = 20   # Nombre d'époques d'entraînement
batch_size = 32  # Taille des lots

# Architecture du modèle
layers = [32, 16]  # Neurones par couche cachée
dropout = 0.2      # Taux de dropout
```

## 📊 Fichiers Générés

### Base de données
- `stock_predictions.db` : Base de données SQLite avec tous les résultats

### Rapports
- `stock_analysis_results.csv` : Résultats au format CSV
- `stock_performance_analysis.png` : Graphiques de performance
- `sector_analysis.png` : Analyse par secteur

## 🔍 Interrogation de la Base de Données

### Exemples de requêtes SQL

```sql
-- Top 5 des meilleures performances
SELECT s.symbol, s.name, p.relative_performance
FROM predictions p
JOIN stocks s ON p.stock_id = s.id
ORDER BY p.relative_performance DESC
LIMIT 5;

-- Performance moyenne par secteur
SELECT s.sector, AVG(p.relative_performance) as avg_performance
FROM predictions p
JOIN stocks s ON p.stock_id = s.id
GROUP BY s.sector
ORDER BY avg_performance DESC;

-- Métriques détaillées pour une action
SELECT dm.metric_name, dm.metric_value
FROM detailed_metrics dm
JOIN predictions p ON dm.prediction_id = p.id
JOIN stocks s ON p.stock_id = s.id
WHERE s.symbol = 'AAPL';
```

## 🎨 Visualisations

Le système génère automatiquement :

1. **Graphique de performance relative** : Comparaison des performances par action
2. **Graphique stratégie vs buy&hold** : Comparaison des rendements
3. **Graphique de précision** : Précision du modèle par action
4. **Graphique de volatilité** : Comparaison des volatilités
5. **Analyse par secteur** : Performance et précision par secteur

## 🔧 Personnalisation

### Ajouter de nouvelles actions
Modifiez la liste `STOCKS` dans `multi_stock_analysis.py` :

```python
STOCKS = [
    {"symbol": "NFLX", "name": "Netflix Inc.", "sector": "Consumer Discretionary"},
    # ... autres actions
]
```

### Modifier l'architecture du modèle
Changez les paramètres dans la fonction `analyze_stock()` :

```python
model = models.Sequential([
    layers.Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
    layers.Dropout(0.3),
    layers.Dense(32, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(16, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])
```

## 📝 Notes Importantes

1. **Gestion d'erreurs** : Le système utilise des données simulées si le téléchargement échoue
2. **Reproductibilité** : Tous les seeds sont fixés pour des résultats reproductibles
3. **Performance** : L'analyse complète prend environ 2-5 minutes selon la machine
4. **Base de données** : SQLite est utilisé pour la simplicité, facilement migrable vers PostgreSQL/MySQL

## 🚨 Limitations

- Les données simulées ne reflètent pas la réalité des marchés
- Le modèle MLP est basique, des architectures plus avancées pourraient améliorer les performances
- Pas de gestion des coûts de transaction
- Pas de validation croisée temporelle avancée

## 🔮 Améliorations Futures

1. **Modèles plus avancés** : LSTM, GRU, Transformers
2. **Features engineering** : Indicateurs techniques, sentiment analysis
3. **Optimisation** : Hyperparameter tuning, ensemble methods
4. **Backtesting** : Validation plus robuste avec données historiques
5. **Interface web** : Dashboard interactif pour visualiser les résultats
