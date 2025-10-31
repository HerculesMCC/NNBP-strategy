import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
import sqlite3
from datetime import datetime

#supprimer les warnings
import warnings
warnings.filterwarnings("ignore")

#liste actions US en input
STOCKS = [
    {"symbol": "AAPL", "name": "Apple Inc."},
    {"symbol": "MSFT", "name": "Microsoft Corporation"},
    {"symbol": "GOOGL", "name": "Alphabet Inc."},
    {"symbol": "AMZN", "name": "Amazon.com Inc."},
    {"symbol": "TSLA", "name": "Tesla Inc."}
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
    """Analyser une action avec un modèle MLP""" #multilayer perceptron
    print(f"Analyse de {symbol}...")
    
    # Calculer les rendements
    data["Return"] = data["Close"].pct_change()
    data.dropna(inplace=True)
    
    # Créer les données d'entraînement (5 jours pour prédire le suivant)
    X, y = [], []
    for i in range(5, len(data)):
        X.append(data["Return"].values[i-5:i])
        y.append(1 if data["Return"].values[i] > 0 else 0)
    
    X = np.array(X)
    y = np.array(y)
    
    if len(X) < 100:  # Pas assez de données
        return None
    
    # Normaliser les données
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Diviser en train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Modèle (2 couches)
    model = models.Sequential([
        layers.Dense(10, activation="relu", input_shape=(5,)),
        layers.Dense(1, activation="sigmoid")
    ])
    
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    
    # Entraînement rapide
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=2)
    
    # Prédictions
    predictions = model.predict(X_test, verbose=0).flatten()
    positions = (predictions > 0.5).astype(int)
    
    # Calculer les performances
    returns = data["Return"].iloc[-len(y_test):].values
    strategy_returns = positions * returns
    
    strategy_total = np.cumprod(1 + strategy_returns)[-1] - 1
    buy_hold_total = np.cumprod(1 + returns)[-1] - 1
    performance = strategy_total - buy_hold_total
    
    accuracy = np.mean(positions == y_test)
    
    print(f"✓ {symbol}: Précision={accuracy:.2f}, Performance={performance:.2f}")
    
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
    print("=== ANALYSE DE 5 ACTIONS AMÉRICAINES ===")
    print()
    
    # Créer la base de données
    conn = create_database()
    
    # Fixer les seeds pour la reproductibilité
    np.random.seed(42)
    tf.random.set_seed(42)
    
    all_results = []
    
    # Analyser chaque action
    for stock_info in STOCKS:
        try:
            # Télécharger les données
            data = download_stock_data(stock_info['symbol'])
            
            # Analyser l'action
            results = analyze_stock(data, stock_info['symbol'])
            
            if results:
                # Sauvegarder en base
                save_to_database(conn, stock_info, results)
                all_results.append(results)
                
        except Exception as e:
            print(f"✗ Erreur pour {stock_info['symbol']}: {e}")
    
    
    print("\n" + "="*50)
    print("RÉSULTATS")
    print("="*50)
    
    if all_results:
        df = pd.DataFrame(all_results)
        
        print(f"Actions analysées: {len(all_results)}")
        print(f"Précision moyenne: {df['accuracy'].mean():.2f}")
        print(f"Performance moyenne: {df['performance'].mean():.2f}")
        print()
        
        print("Classement par performance:")
        df_sorted = df.sort_values('performance', ascending=False)
        for _, row in df_sorted.iterrows():
            print(f"  {row['symbol']}: {row['performance']:.2f}")
        
        # Sauvegarder en CSV
        df.to_csv('results.csv', index=False)
        print(f"\nRésultats sauvegardés dans 'results.csv'")
    
    conn.close()
    print("\nAnalyse terminée ! Base de données: 'stock_analysis.db'")

if __name__ == "__main__":
    main()
