# Analyse de 22 Actions Américaines avec LSTM

## 🎯 Objectif
Analyser 22 actions américaines du S&P 500 en utilisant un modèle LSTM (Long Short-Term Memory) pour prédire les mouvements de prix.

## 📊 Univers d'Investissement : 22 Actions Américaines (S&P 500)

### Justification du choix :

#### 1. **Diversification Sectorielle Équilibrée**
- **11 secteurs différents** représentés selon la classification GICS
- **2 actions par secteur** pour équilibre et comparaison intra-secteur
- Réduction du risque spécifique par diversification
- Représentation fidèle du marché américain

#### 2. **Liquidité et Capitalisation**
- Toutes les actions sont des **grandes capitalisations** (large-cap)
- Volume de trading élevé = exécution facile
- Données historiques complètes et fiables
- Sélection des leaders de chaque secteur

#### 3. **Représentativité du Marché**
- Actions issues du **S&P 500** (indice de référence)
- Poids significatifs dans l'économie américaine
- Comparaison équitable entre secteurs (même nombre d'actions)

#### 4. **Robustesse Statistique**
- **22 actions** = taille d'échantillon suffisante pour analyses statistiques
- Permet l'analyse de corrélations inter-secteurs
- Validation croisée sur plusieurs actifs
- Comparaison équitable entre secteurs

#### 5. **Accessibilité des Données**
- Toutes disponibles via yfinance
- Historique complet depuis 2020
- Pas de problèmes de données manquantes

### Actions Analysées par Secteur (2 par secteur)

**Technologie** : AAPL, MSFT  
**Finance** : JPM, V  
**Santé** : JNJ, UNH  
**Consommation Discrétionnaire** : TSLA, HD  
**Consommation Staples** : WMT, PG  
**Énergie** : XOM, CVX  
**Industriel** : BA, CAT  
**Télécommunications** : T, VZ  
**Matériaux** : LIN, APD  
**Utilitaires** : NEE, DUK  
**Immobilier** : AMT, PLD

## 🔧 Modèle Utilisé : LSTM (Long Short-Term Memory)

### Architecture inspirée des thèses académiques :
- **Couche LSTM** : 50 unités avec activation tanh
- **Dropout** : 0.2 pour la régularisation (évite le surapprentissage)
- **Couche Dense** : 25 neurones avec activation ReLU
- **Dropout** : 0.2 supplémentaire
- **Sortie** : 1 neurone avec activation sigmoid (classification binaire)

### Caractéristiques techniques :
- **Données d'entrée** : 20 jours de rendements consécutifs (fenêtre temporelle)
- **Prédiction** : Direction du mouvement (hausse/baisse) du jour suivant
- **Entraînement** : 15 époques avec validation split (10%)
- **Optimiseur** : Adam
- **Perte** : Binary cross-entropy

### Avantages du LSTM vs MLP :
- **Mémoire à long terme** : Capture les dépendances temporelles complexes
- **Séquences temporelles** : Modèle adapté aux données séquentielles
- **Architecture académique** : Basé sur les travaux de Hochreiter & Schmidhuber (1997)

## 📈 Métriques Calculées
- **Précision** : Pourcentage de prédictions correctes (format %)
- **Rendement stratégie** : Performance cumulée du modèle LSTM (format %)
- **Rendement Buy & Hold** : Performance cumulée d'achat et conservation (format %)
- **Performance relative** : Surperformance de la stratégie vs Buy & Hold (format %)

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
Précision moyenne: 52.00%
Performance moyenne: 8.00%

Classement par performance:
  MC.PA: 15.00% (Précision: 55.00%)
  TTE.PA: 12.00% (Précision: 53.00%)
  BNP.PA: 8.00% (Précision: 51.00%)
  SAN.PA: 5.00% (Précision: 50.00%)
  AI.PA: 2.00% (Précision: 49.00%)
```

**Note** : Toutes les valeurs sont maintenant affichées en pourcentage pour une meilleure lisibilité.

## 🎯 Points Clés pour la Présentation

### Modèle LSTM
1. **Architecture académique** : Inspirée des thèses sur la prédiction de cours boursiers
2. **Mémoire temporelle** : Capture les dépendances longues et courtes termes
3. **Régularisation** : Dropout pour éviter le surapprentissage
4. **Compréhensibilité** : Architecture claire et documentée

### Univers d'Investissement
1. **Diversification** : 5 secteurs différents du CAC 40
2. **Justification** : Choix argumenté (liquidité, secteurs, accessibilité)
3. **Marché européen** : Exposition géographique différente des actions US

### Affichage et Métriques
1. **Format pourcentage** : Toutes les valeurs affichées en % pour clarté
2. **Comparaisons** : Stratégie vs Buy & Hold facilement comparables
3. **Base de données** : Stockage structuré avec timestamps
4. **Reproductibilité** : Seeds fixes pour résultats identiques
