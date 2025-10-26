import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import random
import warnings
import sqlite3
from datetime import datetime
import json

# Supprimer les warnings
warnings.filterwarnings("ignore")

# Liste de 10 actions américaines populaires
STOCKS = [
    {"symbol": "AAPL", "name": "Apple Inc.", "sector": "Technology"},
    {"symbol": "MSFT", "name": "Microsoft Corporation", "sector": "Technology"},
    {"symbol": "GOOGL", "name": "Alphabet Inc.", "sector": "Technology"},
    {"symbol": "AMZN", "name": "Amazon.com Inc.", "sector": "Consumer Discretionary"},
    {"symbol": "TSLA", "name": "Tesla Inc.", "sector": "Consumer Discretionary"},
    {"symbol": "META", "name": "Meta Platforms Inc.", "sector": "Technology"},
    {"symbol": "NVDA", "name": "NVIDIA Corporation", "sector": "Technology"},
    {"symbol": "JPM", "name": "JPMorgan Chase & Co.", "sector": "Financials"},
    {"symbol": "JNJ", "name": "Johnson & Johnson", "sector": "Healthcare"},
    {"symbol": "V", "name": "Visa Inc.", "sector": "Financials"}
]

def create_database():
    """Créer la base de données SQLite avec les tables nécessaires"""
    conn = sqlite3.connect('stock_predictions.db')
    cursor = conn.cursor()
    
    # Table des actions
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS stocks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            sector TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Table des prédictions
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
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
        )
    ''')
    
    # Table des métriques détaillées
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS detailed_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prediction_id INTEGER NOT NULL,
            metric_name TEXT NOT NULL,
            metric_value REAL NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (prediction_id) REFERENCES predictions (id)
        )
    ''')
    
    conn.commit()
    return conn

def insert_stock_data(conn, stock_info):
    """Insérer ou récupérer les informations d'une action"""
    cursor = conn.cursor()
    
    # Vérifier si l'action existe déjà
    cursor.execute('SELECT id FROM stocks WHERE symbol = ?', (stock_info['symbol'],))
    result = cursor.fetchone()
    
    if result:
        return result[0]
    else:
        # Insérer la nouvelle action
        cursor.execute('''
            INSERT INTO stocks (symbol, name, sector) 
            VALUES (?, ?, ?)
        ''', (stock_info['symbol'], stock_info['name'], stock_info['sector']))
        conn.commit()
        return cursor.lastrowid

def download_stock_data(symbol, start_date="2018-01-01", end_date="2023-01-01"):
    """Télécharger les données d'une action avec gestion d'erreur"""
    try:
        print(f"Téléchargement des données pour {symbol}...")
        data = yf.download(symbol, start=start_date, end=end_date, progress=False)
        
        if data.empty:
            raise ValueError("Données vides reçues")
            
        print(f"✓ {symbol}: {len(data)} lignes téléchargées")
        return data
        
    except Exception as e:
        print(f"✗ Erreur pour {symbol}: {e}")
        print(f"Génération de données simulées pour {symbol}...")
        
        # Générer des données simulées
        np.random.seed(hash(symbol) % 2**32)  # Seed basé sur le symbole
        dates = pd.date_range(start=start_date, end=end_date, freq="D")
        n_days = len(dates)
        
        # Simulation d'un prix d'action avec marche aléatoire
        returns = np.random.normal(0.0005, 0.02, n_days)
        prices = 100 * np.cumprod(1 + returns)
        
        data = pd.DataFrame({
            'Open': prices * (1 + np.random.normal(0, 0.005, n_days)),
            'High': prices * (1 + np.abs(np.random.normal(0, 0.01, n_days))),
            'Low': prices * (1 - np.abs(np.random.normal(0, 0.01, n_days))),
            'Close': prices,
            'Volume': np.random.randint(1000000, 10000000, n_days)
        }, index=dates)
        
        print(f"✓ {symbol}: {len(data)} lignes simulées générées")
        return data

def analyze_stock(data, symbol, lookback=5):
    """Analyser une action avec le modèle MLP"""
    print(f"\n--- Analyse de {symbol} ---")
    
    # Préparation des données
    data["Return"] = data["Close"].pct_change()
    data.dropna(inplace=True)
    data["Target"] = (data["Return"] > 0).astype(int)
    
    # Création des séquences
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data["Return"].values[i-lookback:i])
        y.append(data["Target"].values[i])
    
    X = np.array(X)
    y = np.array(y)
    
    if len(X) == 0:
        print(f"✗ Pas assez de données pour {symbol}")
        return None
    
    # Standardisation
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    
    # Création du modèle
    model = models.Sequential([
        layers.Dense(32, activation="relu", input_shape=(X_train.shape[1],)),
        layers.Dropout(0.2),
        layers.Dense(16, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(1, activation="sigmoid")
    ])
    
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    
    # Entraînement
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                       epochs=20, batch_size=32, verbose=0)
    
    # Prédictions
    proba = model.predict(X_test, verbose=0).flatten()
    positions = (proba > 0.5).astype(int)
    
    # Calcul des métriques
    returns = data["Return"].iloc[-len(y_test):].values
    strategy_returns = positions * returns
    
    strategy_cumulative = np.cumprod(1 + strategy_returns)[-1] - 1
    buy_hold_cumulative = np.cumprod(1 + returns)[-1] - 1
    relative_performance = strategy_cumulative - buy_hold_cumulative
    
    strategy_vol = np.std(strategy_returns) * np.sqrt(252)
    buy_hold_vol = np.std(returns) * np.sqrt(252)
    
    accuracy = np.mean(positions == y_test)
    
    # Résultats
    results = {
        'symbol': symbol,
        'lookback_period': lookback,
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'model_accuracy': float(accuracy),
        'strategy_return': float(strategy_cumulative),
        'buy_hold_return': float(buy_hold_cumulative),
        'relative_performance': float(relative_performance),
        'strategy_volatility': float(strategy_vol),
        'buy_hold_volatility': float(buy_hold_vol),
        'positions_taken': int(np.sum(positions)),
        'total_positions': len(positions),
        'model_params': json.dumps({
            'layers': len(model.layers),
            'total_params': model.count_params(),
            'input_shape': X_train.shape[1]
        })
    }
    
    print(f"✓ {symbol}: Précision={accuracy:.3f}, Rendement stratégie={strategy_cumulative:.3f}, Buy&Hold={buy_hold_cumulative:.3f}")
    
    return results

def save_results_to_db(conn, stock_id, results):
    """Sauvegarder les résultats en base de données"""
    cursor = conn.cursor()
    
    # Insérer la prédiction
    cursor.execute('''
        INSERT INTO predictions (
            stock_id, date, lookback_period, training_samples, test_samples,
            model_accuracy, strategy_return, buy_hold_return, relative_performance,
            strategy_volatility, buy_hold_volatility, positions_taken, total_positions, model_params
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        stock_id,
        datetime.now().strftime('%Y-%m-%d'),
        results['lookback_period'],
        results['training_samples'],
        results['test_samples'],
        results['model_accuracy'],
        results['strategy_return'],
        results['buy_hold_return'],
        results['relative_performance'],
        results['strategy_volatility'],
        results['buy_hold_volatility'],
        results['positions_taken'],
        results['total_positions'],
        results['model_params']
    ))
    
    prediction_id = cursor.lastrowid
    
    # Insérer les métriques détaillées
    detailed_metrics = [
        ('accuracy', results['model_accuracy']),
        ('strategy_return', results['strategy_return']),
        ('buy_hold_return', results['buy_hold_return']),
        ('relative_performance', results['relative_performance']),
        ('strategy_volatility', results['strategy_volatility']),
        ('buy_hold_volatility', results['buy_hold_volatility']),
        ('positions_ratio', results['positions_taken'] / results['total_positions'])
    ]
    
    for metric_name, metric_value in detailed_metrics:
        cursor.execute('''
            INSERT INTO detailed_metrics (prediction_id, metric_name, metric_value)
            VALUES (?, ?, ?)
        ''', (prediction_id, metric_name, metric_value))
    
    conn.commit()
    return prediction_id

def main():
    """Fonction principale"""
    print("=== ANALYSE DE 10 ACTIONS AMÉRICAINES ===")
    print("Création de la base de données...")
    
    # Créer la base de données
    conn = create_database()
    
    # Configuration déterministe
    seed = 42
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)
    
    all_results = []
    
    for stock_info in STOCKS:
        try:
            # Télécharger les données
            data = download_stock_data(stock_info['symbol'])
            
            # Analyser l'action
            results = analyze_stock(data, stock_info['symbol'])
            
            if results:
                # Insérer en base
                stock_id = insert_stock_data(conn, stock_info)
                prediction_id = save_results_to_db(conn, stock_id, results)
                
                results['stock_id'] = stock_id
                results['prediction_id'] = prediction_id
                all_results.append(results)
                
        except Exception as e:
            print(f"✗ Erreur lors de l'analyse de {stock_info['symbol']}: {e}")
            continue
    
    # Afficher le résumé
    print("\n" + "="*60)
    print("RÉSUMÉ DE L'ANALYSE")
    print("="*60)
    
    if all_results:
        df_results = pd.DataFrame(all_results)
        
        print(f"Actions analysées: {len(all_results)}")
        print(f"Précision moyenne: {df_results['model_accuracy'].mean():.3f}")
        print(f"Rendement stratégie moyen: {df_results['strategy_return'].mean():.3f}")
        print(f"Rendement buy&hold moyen: {df_results['buy_hold_return'].mean():.3f}")
        print(f"Performance relative moyenne: {df_results['relative_performance'].mean():.3f}")
        
        # Meilleure et pire performance
        best_stock = df_results.loc[df_results['relative_performance'].idxmax()]
        worst_stock = df_results.loc[df_results['relative_performance'].idxmin()]
        
        print(f"\nMeilleure performance: {best_stock['symbol']} ({best_stock['relative_performance']:.3f})")
        print(f"Pire performance: {worst_stock['symbol']} ({worst_stock['relative_performance']:.3f})")
        
        # Sauvegarder le résumé en CSV
        df_results.to_csv('stock_analysis_results.csv', index=False)
        print(f"\nRésultats sauvegardés dans 'stock_analysis_results.csv'")
    
    conn.close()
    print("\nAnalyse terminée ! Base de données: 'stock_predictions.db'")

if __name__ == "__main__":
    main()
