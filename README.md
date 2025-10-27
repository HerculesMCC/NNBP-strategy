# Analyse d'Actions Américaines avec MLP

## 🎯 Objectif
Analyser 5 actions américaines populaires en utilisant un réseau de neurones MLP pour prédire les mouvements de prix.

## 📊 Actions Analysées
- **AAPL** - Apple Inc.
- **MSFT** - Microsoft Corporation  
- **GOOGL** - Alphabet Inc.
- **AMZN** - Amazon.com Inc.
- **TSLA** - Tesla Inc.

## 🔧 Modèle Utilisé
- **Architecture** : 2 couches (10 neurones + 1 sortie)
- **Données d'entrée** : 5 jours de rendements
- **Prédiction** : Hausse ou baisse du jour suivant
- **Entraînement** : 10 époques

## 📈 Métriques Calculées
- **Précision** : Pourcentage de prédictions correctes
- **Rendement stratégie** : Performance du modèle MLP
- **Rendement Buy & Hold** : Performance d'achat et conservation
- **Performance relative** : Différence entre les deux

## 🚀 Utilisation

### 1. Lancer l'analyse complète
```bash
python demo.py
```

### 2. Analyser seulement les actions
```bash
python stock_analysis.py
```

### 3. Voir les résultats
```bash
python stock_analyzer.py
```

## 📁 Fichiers Générés
- `stock_analysis.db` - Base de données SQLite
- `results.csv` - Résultats au format CSV

## 🗄️ Structure de la Base de Données
```sql
CREATE TABLE stock_results (
    id INTEGER PRIMARY KEY,
    symbol TEXT NOT NULL,
    name TEXT NOT NULL,
    accuracy REAL NOT NULL,
    strategy_return REAL NOT NULL,
    buy_hold_return REAL NOT NULL,
    performance REAL NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## 📝 Exemple de Résultats
```
Actions analysées: 5
Précision moyenne: 0.52
Performance moyenne: 0.08

Classement par performance:
  TSLA: 0.15
  AAPL: 0.12
  MSFT: 0.08
  GOOGL: 0.05
  AMZN: 0.02
```

## 🎯 Points Clés pour la Présentation
1. **Clarté** : Code facile à comprendre et expliquer
2. **Efficacité** : Analyse rapide (2-3 minutes)
3. **Résultats clairs** : Métriques comparables
4. **Base de données** : Stockage structuré des résultats
5. **Reproductibilité** : Seeds fixes pour résultats identiques
