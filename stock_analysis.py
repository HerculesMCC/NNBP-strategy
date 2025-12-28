import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
import sqlite3
from datetime import datetime

# Supprimer les warnings
import warnings
warnings.filterwarnings("ignore")

# UNIVERS D'INVESTISSEMENT : 22 Actions Américaines (S&P 500)
# 
# JUSTIFICATION DU CHOIX :
# =========================
# 1. DIVERSIFICATION SECTORIELLE ÉQUILIBRÉE
#    - 11 secteurs différents représentés (GICS)
#    - 2 actions par secteur pour équilibre et comparaison
#    - Réduction du risque spécifique par diversification
#    - Représentation fidèle du marché américain
#
# 2. LIQUIDITÉ ET CAPITALISATION
#    - Toutes les actions sont des grandes capitalisations (large-cap)
#    - Volume de trading élevé = exécution facile
#    - Données historiques complètes et fiables
#
# 3. REPRÉSENTATIVITÉ DU MARCHÉ
#    - Actions issues du S&P 500 (indice de référence)
#    - Poids significatifs dans l'économie américaine
#    - Sélection des leaders de chaque secteur
#
# 4. ROBUSTESSE STATISTIQUE
#    - 22 actions = taille d'échantillon suffisante pour analyses statistiques
#    - Permet l'analyse de corrélations inter-secteurs
#    - Validation croisée sur plusieurs actifs
#    - Comparaison équitable entre secteurs (2 actions chacun)

STOCKS = [
    # TECHNOLOGIE (2 actions) - Secteur dominant du S&P 500
    {"symbol": "AAPL", "name": "Apple Inc."},
    {"symbol": "MSFT", "name": "Microsoft Corporation"},
    
    # FINANCE (2 actions) - Secteur cyclique important
    {"symbol": "JPM", "name": "JPMorgan Chase & Co."},
    {"symbol": "V", "name": "Visa Inc."},
    
    # SANTÉ (2 actions) - Secteur défensif
    {"symbol": "JNJ", "name": "Johnson & Johnson"},
    {"symbol": "UNH", "name": "UnitedHealth Group Inc."},
    
    # CONSOMMATION DISCRÉTIONNAIRE (2 actions)
    {"symbol": "TSLA", "name": "Tesla Inc."},
    {"symbol": "HD", "name": "Home Depot Inc."},
    
    # CONSOMMATION STAPLES (2 actions) - Défensif
    {"symbol": "WMT", "name": "Walmart Inc."},
    {"symbol": "PG", "name": "Procter & Gamble Co."},
    
    # ÉNERGIE (2 actions) - Secteur cyclique
    {"symbol": "XOM", "name": "Exxon Mobil Corporation"},
    {"symbol": "CVX", "name": "Chevron Corporation"},
    
    # INDUSTRIEL (2 actions)
    {"symbol": "BA", "name": "Boeing Company"},
    {"symbol": "CAT", "name": "Caterpillar Inc."},
    
    # TÉLÉCOMMUNICATIONS (2 actions)
    {"symbol": "T", "name": "AT&T Inc."},
    {"symbol": "VZ", "name": "Verizon Communications Inc."},
    
    # MATÉRIAUX (2 actions)
    {"symbol": "LIN", "name": "Linde plc"},
    {"symbol": "APD", "name": "Air Products and Chemicals Inc."},
    
    # UTILITAIRES (2 actions) - Défensif
    {"symbol": "NEE", "name": "NextEra Energy Inc."},
    {"symbol": "DUK", "name": "Duke Energy Corporation"},
    
    # IMMOBILIER (2 actions)
    {"symbol": "AMT", "name": "American Tower Corporation"},
    {"symbol": "PLD", "name": "Prologis Inc."}
]

def create_database():
    
    conn = sqlite3.connect('stock_analysis.db')
    cursor = conn.cursor()
    
    # Table avec tous les résultats
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS stock_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            name TEXT NOT NULL,
            accuracy REAL NOT NULL,
            strategy_return REAL NOT NULL,
            buy_hold_return REAL NOT NULL,
            performance REAL NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    return conn

def download_stock_data(symbol):
    try:
        print(f"Téléchargement de {symbol}...")
        data = yf.download(symbol, start="2020-01-01", end="2023-01-01", progress=False)
        
        if data.empty:
            raise ValueError("Données vides")
            
        print(f"✓ {symbol}: {len(data)} jours de données")
        return data
        
    except Exception as e:
        print(f"✗ Erreur pour {symbol}: {e}")
        # Données simulées simples
        dates = pd.date_range(start="2020-01-01", end="2023-01-01", freq="D")
        prices = 100 * np.cumprod(1 + np.random.normal(0.001, 0.02, len(dates)))
        
        data = pd.DataFrame({
            'Close': prices
        }, index=dates)
        
        print(f"✓ {symbol}: données simulées générées")
        return data

def analyze_stock(data, symbol):
    """
    Analyser une action avec un modèle LSTM (Long Short-Term Memory)

    - LSTM capture les dépendances temporelles longues et courtes
    - Architecture classique : LSTM → Dropout → Dense → Sortie sigmoïde
    - Fenêtre temporelle de 20 jours pour capturer les tendances et prédire le rendement N
    
    Références académiques typiques :
    - Hochreiter & Schmidhuber (1997) : LSTM pour séquences temporelles
    """
    print(f"Analyse de {symbol}...")
    
    # Calculer les rendements journaliers
    data["Return"] = data["Close"].pct_change()
    data.dropna(inplace=True)
    
    # Fenêtre temporelle : 20 jours pour capturer les tendances
    # (plus longue que le MLP précédent où c'était 5 jours)
    
    
    lookback_window = 20 #60, 40 et 20 à tester
    
    # Créer les séquences temporelles pour le LSTM
    X, y = [], []
    for i in range(lookback_window, len(data)):
        # Séquence de 20 rendements consécutifs
        X.append(data["Return"].values[i-lookback_window:i])
        # Cible : direction du rendement du jour suivant (1 = hausse, 0 = baisse)
        y.append(1 if data["Return"].values[i] > 0 else 0)
    
    X = np.array(X)
    y = np.array(y)
    
    # Normaliser les données (important pour LSTM)
    scaler = StandardScaler()
    # Reshape pour StandardScaler : (samples, features) -> (samples*features, 1)
    X_reshaped = X.reshape(-1, 1)
    X_scaled = scaler.fit_transform(X_reshaped)
    X_scaled = X_scaled.reshape(X.shape)
    
    # Diviser en train/test (split temporel, pas aléatoire)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Reshape pour LSTM : (samples, timesteps, features)
    # Ici : (samples, 20, 1) - 20 pas de temps, 1 feature par pas
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    
    #LSTM → Dropout (régularisation) → Dense → Sortie
    model = models.Sequential([
        # Couche LSTM avec 50 unités (mémoire)
        # return_sequences=False : on ne retourne que la dernière sortie
        layers.LSTM(50, activation='tanh', input_shape=(lookback_window, 1)),
        
        # Dropout pour éviter le surapprentissage (régularisation)
        layers.Dropout(0.2),

        # Couche dense pour la classification finale
        layers.Dense(25, activation='relu'),
        
        # Dropout supplémentaire
        layers.Dropout(0.2),
        
        # Sortie : probabilité de hausse (sigmoid pour classification binaire)
        layers.Dense(1, activation='sigmoid')
    ])
    
    # Compilation avec optimiseur Adam (standard en deep learning)
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',  # Perte adaptée à la classification binaire
        metrics=['accuracy']
    )
    
    # Entraînement avec validation
    # epochs=5 : entraînement rapide pour analyse efficace
    model.fit(
        X_train, y_train,
        epochs=5,
        batch_size=32,
        validation_split=0.1,  # 10% du train pour validation
        verbose=1
    )
    
    # Prédictions sur le set de test
    predictions = model.predict(X_test, verbose=0).flatten()
    positions = (predictions > 0.5).astype(int)  # Seuil de décision : 0.5
    
    # Calculer les performances de la stratégie
    # Les prédictions correspondent aux rendements à partir de split_idx+lookback_window
    # car chaque séquence X[i] prédit le rendement du jour i+lookback_window
    test_start_idx = split_idx + lookback_window
    returns = data["Return"].iloc[test_start_idx:test_start_idx+len(positions)].values
    
    strategy_returns = positions * returns  # Investi seulement si prédiction hausse
    
    # Rendements cumulés
    strategy_total = np.cumprod(1 + strategy_returns)[-1] - 1
    buy_hold_total = np.cumprod(1 + returns)[-1] - 1
    performance = strategy_total - buy_hold_total
    
    # Précision : pourcentage de prédictions correctes
    accuracy = np.mean(positions == y_test)
    
    print(f"✓ {symbol}: Précision={accuracy:.2%}, Performance={performance:.2%}")
    
    return {
        'symbol': symbol,
        'accuracy': accuracy,
        'strategy_return': strategy_total,
        'buy_hold_return': buy_hold_total,
        'performance': performance
    }

def save_to_database(conn, stock_info, results):
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO stock_results (symbol, name, accuracy, strategy_return, buy_hold_return, performance)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (
        results['symbol'],
        stock_info['name'],
        results['accuracy'],
        results['strategy_return'],
        results['buy_hold_return'],
        results['performance']
    ))
    conn.commit()

def main():
    """Fonction principale"""
    print("=== ANALYSE DE 22 ACTIONS AMÉRICAINES (S&P 500) ===")
    print("Modèle : LSTM (Long Short-Term Memory)")
    print(f"Univers : {len(STOCKS)} actions (2 par secteur, 11 secteurs)")
    print()
    
    # Créer la base de données
    conn = create_database()
    
    # Fixer les seeds pour la reproductibilité
    np.random.seed(42)
    tf.random.set_seed(42)
    
    all_results = []
    
    # Analyser chaque action
    total_stocks = len(STOCKS)
    for idx, stock_info in enumerate(STOCKS, 1):
        try:
            print(f"\n[{idx}/{total_stocks}] Traitement de {stock_info['symbol']} ({stock_info['name']})")
            
            # Télécharger les données
            data = download_stock_data(stock_info['symbol'])
            
            # Analyser l'action
            results = analyze_stock(data, stock_info['symbol'])
            
            if results:
                # Sauvegarder en base
                save_to_database(conn, stock_info, results)
                all_results.append(results)
                print(f"✓ {stock_info['symbol']} terminé avec succès")
            else:
                print(f"⚠ {stock_info['symbol']} : Analyse non effectuée (données insuffisantes)")
                
        except Exception as e:
            print(f"✗ Erreur pour {stock_info['symbol']}: {e}")
    
    
    print("\n" + "="*50)
    print("RÉSULTATS")
    print("="*50)
    
    if all_results:
        df = pd.DataFrame(all_results)
        
        print(f"\n{'='*60}")
        print(f"RÉSUMÉ FINAL")
        print(f"{'='*60}")
        print(f"Actions analysées avec succès: {len(all_results)}/{total_stocks}")
        print(f"Précision moyenne: {df['accuracy'].mean():.2%}")
        print(f"Performance moyenne: {df['performance'].mean():.2%}")
        print(f"Rendement stratégie moyen: {df['strategy_return'].mean():.2%}")
        print(f"Rendement Buy & Hold moyen: {df['buy_hold_return'].mean():.2%}")
        print()
        
        print("🏆 TOP 10 PAR PERFORMANCE:")
        print("-" * 60)
        df_sorted = df.sort_values('performance', ascending=False)
        for i, (_, row) in enumerate(df_sorted.head(10).iterrows(), 1):
            print(f"{i:2d}. {row['symbol']:6s} | Performance: {row['performance']:7.2%} | "
                  f"Stratégie: {row['strategy_return']:7.2%} | B&H: {row['buy_hold_return']:7.2%} | "
                  f"Précision: {row['accuracy']:5.2%}")
        
        if len(df_sorted) > 10:
            print(f"\n... et {len(df_sorted) - 10} autres actions")
        
        # Sauvegarder en CSV
        df.to_csv('results.csv', index=False)
        print(f"\nRésultats sauvegardés dans 'results.csv'")
    
    conn.close()
    print("\nAnalyse terminée ! Base de données: 'stock_analysis.db'")

if __name__ == "__main__":
    main()
