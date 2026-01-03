import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
import sqlite3
from datetime import datetime
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import matplotlib.pyplot as plt
import seaborn as sns

# Supprimer les warnings
import warnings
warnings.filterwarnings("ignore")

# Configuration des graphiques
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    try:
        plt.style.use('seaborn-darkgrid')
    except:
        plt.style.use('ggplot')
sns.set_palette("husl")

# UNIVERS D'INVESTISSEMENT : 100 Actions Américaines (S&P 500)
# 
# JUSTIFICATION DU CHOIX :
# =========================
# 1. DIVERSIFICATION SECTORIELLE MAXIMALE
#    - 11 secteurs différents représentés (GICS)
#    - ~9-10 actions par secteur en moyenne
#    - Réduction du risque spécifique par diversification
#    - Représentation fidèle du marché américain
#
# 2. ARCHITECTURE AMÉLIORÉE : MODÈLE PAR SECTEUR
#    - Un modèle LSTM par secteur (au lieu d'un par action)
#    - Entraînement sur toutes les actions du secteur = plus de données
#    - Capture les patterns communs au secteur
#    - Réduction du nombre de modèles (11 au lieu de 100)
#    - Meilleure généralisation grâce à plus de données d'entraînement
#
# 3. LIQUIDITÉ ET CAPITALISATION
#    - Toutes les actions sont des grandes capitalisations (large-cap)
#    - Volume de trading élevé = exécution facile
#    - Données historiques complètes et fiables
#
# 4. REPRÉSENTATIVITÉ DU MARCHÉ
#    - Actions issues du S&P 500 (indice de référence)
#    - Poids significatifs dans l'économie américaine
#    - Couverture d'environ 80% de la capitalisation boursière US

# Liste de 100 actions avec leur secteur
STOCKS = [
    # TECHNOLOGIE (15 actions)
    {"symbol": "AAPL", "name": "Apple Inc.", "sector": "Technologie"},
    {"symbol": "MSFT", "name": "Microsoft Corporation", "sector": "Technologie"},
    {"symbol": "GOOGL", "name": "Alphabet Inc.", "sector": "Technologie"},
    {"symbol": "AMZN", "name": "Amazon.com Inc.", "sector": "Technologie"},
    {"symbol": "META", "name": "Meta Platforms Inc.", "sector": "Technologie"},
    {"symbol": "NVDA", "name": "NVIDIA Corporation", "sector": "Technologie"},
    {"symbol": "ORCL", "name": "Oracle Corporation", "sector": "Technologie"},
    {"symbol": "CRM", "name": "Salesforce Inc.", "sector": "Technologie"},
    {"symbol": "INTC", "name": "Intel Corporation", "sector": "Technologie"},
    {"symbol": "AMD", "name": "Advanced Micro Devices Inc.", "sector": "Technologie"},
    {"symbol": "CSCO", "name": "Cisco Systems Inc.", "sector": "Technologie"},
    {"symbol": "ADBE", "name": "Adobe Inc.", "sector": "Technologie"},
    {"symbol": "AVGO", "name": "Broadcom Inc.", "sector": "Technologie"},
    {"symbol": "QCOM", "name": "Qualcomm Inc.", "sector": "Technologie"},
    {"symbol": "TXN", "name": "Texas Instruments Inc.", "sector": "Technologie"},
    
    # FINANCE (12 actions)
    {"symbol": "JPM", "name": "JPMorgan Chase & Co.", "sector": "Finance"},
    {"symbol": "BAC", "name": "Bank of America Corp.", "sector": "Finance"},
    {"symbol": "GS", "name": "Goldman Sachs Group Inc.", "sector": "Finance"},
    {"symbol": "V", "name": "Visa Inc.", "sector": "Finance"},
    {"symbol": "MA", "name": "Mastercard Inc.", "sector": "Finance"},
    {"symbol": "WFC", "name": "Wells Fargo & Company", "sector": "Finance"},
    {"symbol": "C", "name": "Citigroup Inc.", "sector": "Finance"},
    {"symbol": "AXP", "name": "American Express Company", "sector": "Finance"},
    {"symbol": "BLK", "name": "BlackRock Inc.", "sector": "Finance"},
    {"symbol": "SCHW", "name": "Charles Schwab Corporation", "sector": "Finance"},
    {"symbol": "USB", "name": "U.S. Bancorp", "sector": "Finance"},
    {"symbol": "PNC", "name": "PNC Financial Services Group", "sector": "Finance"},
    
    # SANTÉ (12 actions)
    {"symbol": "JNJ", "name": "Johnson & Johnson", "sector": "Santé"},
    {"symbol": "UNH", "name": "UnitedHealth Group Inc.", "sector": "Santé"},
    {"symbol": "PFE", "name": "Pfizer Inc.", "sector": "Santé"},
    {"symbol": "ABBV", "name": "AbbVie Inc.", "sector": "Santé"},
    {"symbol": "TMO", "name": "Thermo Fisher Scientific Inc.", "sector": "Santé"},
    {"symbol": "ABT", "name": "Abbott Laboratories", "sector": "Santé"},
    {"symbol": "LLY", "name": "Eli Lilly and Company", "sector": "Santé"},
    {"symbol": "DHR", "name": "Danaher Corporation", "sector": "Santé"},
    {"symbol": "BMY", "name": "Bristol-Myers Squibb Company", "sector": "Santé"},
    {"symbol": "AMGN", "name": "Amgen Inc.", "sector": "Santé"},
    {"symbol": "GILD", "name": "Gilead Sciences Inc.", "sector": "Santé"},
    {"symbol": "CVS", "name": "CVS Health Corporation", "sector": "Santé"},
    
    # CONSOMMATION DISCRÉTIONNAIRE (10 actions)
    {"symbol": "TSLA", "name": "Tesla Inc.", "sector": "Consommation Discrétionnaire"},
    {"symbol": "NKE", "name": "Nike Inc.", "sector": "Consommation Discrétionnaire"},
    {"symbol": "HD", "name": "Home Depot Inc.", "sector": "Consommation Discrétionnaire"},
    {"symbol": "MCD", "name": "McDonald's Corporation", "sector": "Consommation Discrétionnaire"},
    {"symbol": "SBUX", "name": "Starbucks Corporation", "sector": "Consommation Discrétionnaire"},
    {"symbol": "NFLX", "name": "Netflix Inc.", "sector": "Consommation Discrétionnaire"},
    {"symbol": "LOW", "name": "Lowe's Companies Inc.", "sector": "Consommation Discrétionnaire"},
    {"symbol": "TJX", "name": "TJX Companies Inc.", "sector": "Consommation Discrétionnaire"},
    {"symbol": "BKNG", "name": "Booking Holdings Inc.", "sector": "Consommation Discrétionnaire"},
    {"symbol": "GM", "name": "General Motors Company", "sector": "Consommation Discrétionnaire"},
    
    # CONSOMMATION STAPLES (8 actions)
    {"symbol": "WMT", "name": "Walmart Inc.", "sector": "Consommation Staples"},
    {"symbol": "PG", "name": "Procter & Gamble Co.", "sector": "Consommation Staples"},
    {"symbol": "KO", "name": "The Coca-Cola Company", "sector": "Consommation Staples"},
    {"symbol": "PEP", "name": "PepsiCo Inc.", "sector": "Consommation Staples"},
    {"symbol": "COST", "name": "Costco Wholesale Corporation", "sector": "Consommation Staples"},
    {"symbol": "CL", "name": "Colgate-Palmolive Company", "sector": "Consommation Staples"},
    {"symbol": "MDLZ", "name": "Mondelez International Inc.", "sector": "Consommation Staples"},
    {"symbol": "STZ", "name": "Constellation Brands Inc.", "sector": "Consommation Staples"},
    
    # ÉNERGIE (8 actions)
    {"symbol": "XOM", "name": "Exxon Mobil Corporation", "sector": "Énergie"},
    {"symbol": "CVX", "name": "Chevron Corporation", "sector": "Énergie"},
    {"symbol": "SLB", "name": "Schlumberger Limited", "sector": "Énergie"},
    {"symbol": "COP", "name": "ConocoPhillips", "sector": "Énergie"},
    {"symbol": "EOG", "name": "EOG Resources Inc.", "sector": "Énergie"},
    {"symbol": "MPC", "name": "Marathon Petroleum Corporation", "sector": "Énergie"},
    {"symbol": "PSX", "name": "Phillips 66", "sector": "Énergie"},
    {"symbol": "VLO", "name": "Valero Energy Corporation", "sector": "Énergie"},
    
    # INDUSTRIEL (10 actions)
    {"symbol": "BA", "name": "Boeing Company", "sector": "Industriel"},
    {"symbol": "CAT", "name": "Caterpillar Inc.", "sector": "Industriel"},
    {"symbol": "GE", "name": "General Electric Company", "sector": "Industriel"},
    {"symbol": "HON", "name": "Honeywell International Inc.", "sector": "Industriel"},
    {"symbol": "UPS", "name": "United Parcel Service Inc.", "sector": "Industriel"},
    {"symbol": "RTX", "name": "Raytheon Technologies Corporation", "sector": "Industriel"},
    {"symbol": "LMT", "name": "Lockheed Martin Corporation", "sector": "Industriel"},
    {"symbol": "DE", "name": "Deere & Company", "sector": "Industriel"},
    {"symbol": "EMR", "name": "Emerson Electric Co.", "sector": "Industriel"},
    {"symbol": "ETN", "name": "Eaton Corporation plc", "sector": "Industriel"},
    
    # TÉLÉCOMMUNICATIONS (5 actions)
    {"symbol": "T", "name": "AT&T Inc.", "sector": "Télécommunications"},
    {"symbol": "VZ", "name": "Verizon Communications Inc.", "sector": "Télécommunications"},
    {"symbol": "CMCSA", "name": "Comcast Corporation", "sector": "Télécommunications"},
    {"symbol": "DIS", "name": "Walt Disney Company", "sector": "Télécommunications"},
    {"symbol": "CHTR", "name": "Charter Communications Inc.", "sector": "Télécommunications"},
    
    # MATÉRIAUX (8 actions)
    {"symbol": "LIN", "name": "Linde plc", "sector": "Matériaux"},
    {"symbol": "APD", "name": "Air Products and Chemicals Inc.", "sector": "Matériaux"},
    {"symbol": "ECL", "name": "Ecolab Inc.", "sector": "Matériaux"},
    {"symbol": "SHW", "name": "Sherwin-Williams Company", "sector": "Matériaux"},
    {"symbol": "PPG", "name": "PPG Industries Inc.", "sector": "Matériaux"},
    {"symbol": "FCX", "name": "Freeport-McMoRan Inc.", "sector": "Matériaux"},
    {"symbol": "NEM", "name": "Newmont Corporation", "sector": "Matériaux"},
    {"symbol": "DD", "name": "DuPont de Nemours Inc.", "sector": "Matériaux"},
    
    # UTILITAIRES (6 actions)
    {"symbol": "NEE", "name": "NextEra Energy Inc.", "sector": "Utilitaires"},
    {"symbol": "DUK", "name": "Duke Energy Corporation", "sector": "Utilitaires"},
    {"symbol": "SO", "name": "Southern Company", "sector": "Utilitaires"},
    {"symbol": "AEP", "name": "American Electric Power Company", "sector": "Utilitaires"},
    {"symbol": "SRE", "name": "Sempra Energy", "sector": "Utilitaires"},
    {"symbol": "EXC", "name": "Exelon Corporation", "sector": "Utilitaires"},
    
    # IMMOBILIER (6 actions)
    {"symbol": "AMT", "name": "American Tower Corporation", "sector": "Immobilier"},
    {"symbol": "PLD", "name": "Prologis Inc.", "sector": "Immobilier"},
    {"symbol": "EQIX", "name": "Equinix Inc.", "sector": "Immobilier"},
    {"symbol": "PSA", "name": "Public Storage", "sector": "Immobilier"},
    {"symbol": "WELL", "name": "Welltower Inc.", "sector": "Immobilier"},
    {"symbol": "SPG", "name": "Simon Property Group Inc.", "sector": "Immobilier"}
]

def create_database():
    """Créer la base de données et vider les tables à chaque run"""
    conn = sqlite3.connect('stock_analysis.db')
    cursor = conn.cursor()
    
    # Table avec tous les résultats (modèles par secteur)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS stock_results (
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
        )
    ''')
    
    # Table pour les résultats du modèle global
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS global_model_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            name TEXT NOT NULL,
            sector TEXT NOT NULL,
            model_type TEXT NOT NULL DEFAULT 'global',
            accuracy REAL NOT NULL,
            strategy_return REAL NOT NULL,
            buy_hold_return REAL NOT NULL,
            performance REAL NOT NULL,
            sharpe_ratio REAL,
            sortino_ratio REAL,
            max_drawdown REAL,
            volatility REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Table pour les métriques agrégées de comparaison
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS model_comparison (
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
        )
    ''')
    
    # Migration: ajouter les nouvelles colonnes si elles n'existent pas
    try:
        cursor.execute("PRAGMA table_info(stock_results)")
        columns = [col[1] for col in cursor.fetchall()]
        
        new_columns = {
            'sector': 'TEXT',
            'model_type': 'TEXT DEFAULT "sector"',
            'sharpe_ratio': 'REAL',
            'sortino_ratio': 'REAL',
            'max_drawdown': 'REAL',
            'volatility': 'REAL'
        }
        
        for col_name, col_type in new_columns.items():
            if col_name not in columns:
                cursor.execute(f'ALTER TABLE stock_results ADD COLUMN {col_name} {col_type}')
                print(f"  ✓ Colonne '{col_name}' ajoutée à stock_results")
    except sqlite3.OperationalError as e:
        pass
    
    # VIDER LES TABLES à chaque run
    cursor.execute('DELETE FROM stock_results')
    cursor.execute('DELETE FROM global_model_results')
    cursor.execute('DELETE FROM model_comparison')
    print("  ✓ Tables vidées pour un nouveau run")
    
    conn.commit()
    return conn

def download_stock_data(symbol, verbose=True):
    """Télécharger les données d'une action"""
    try:
        if verbose:
            print(f"  Téléchargement de {symbol}...")
        data = yf.download(symbol, start="2020-01-01", end="2023-01-01", progress=False)
        
        if data.empty:
            raise ValueError("Données vides")
            
        if verbose:
            print(f"  ✓ {symbol}: {len(data)} jours de données")
        return data
        
    except Exception as e:
        if verbose:
            print(f"  ✗ Erreur pour {symbol}: {e}")
        return None

def prepare_sector_data(stocks_in_sector, rolling_window_days=252, test_window_days=63, data_cache=None):
    """
    Préparer les données d'entraînement pour un secteur avec ROLLING WINDOW
    Combine toutes les séquences de toutes les actions du secteur
    
    ROLLING WINDOW vs EXTENDING WINDOW:
    - EXTENDING: utilise toutes les données jusqu'à un point (fenêtre qui s'étend)
    - ROLLING: utilise une fenêtre de taille fixe qui glisse dans le temps (plus réaliste)
    
    Args:
        rolling_window_days: Taille de la fenêtre d'entraînement (252 = ~1 an de trading)
        test_window_days: Taille de la fenêtre de test (63 = ~3 mois de trading)
    """
    lookback_window = 20
    all_X_train, all_y_train = [], []
    all_X_test, all_y_test = [], []
    all_stock_data = []  # Stocker les données brutes pour le scaler
    
    for stock_info in stocks_in_sector:
        try:
            symbol = stock_info['symbol']
            # Utiliser le cache si disponible, sinon télécharger
            if data_cache is not None and symbol in data_cache:
                data = data_cache[symbol]
            else:
                data = download_stock_data(symbol, verbose=False)
                if data_cache is not None and data is not None:
                    data_cache[symbol] = data
            
            if data is None or data.empty:
                continue
                
            # Calculer les rendements
            data["Return"] = data["Close"].pct_change()
            data.dropna(inplace=True)
            
            if len(data) < lookback_window + rolling_window_days + test_window_days:
                continue
            
            # ROLLING WINDOW: créer plusieurs fenêtres glissantes
            # On commence après le lookback_window et on crée des fenêtres qui glissent
            total_days = len(data)
            start_idx = lookback_window
            
            # Créer des fenêtres glissantes jusqu'à ce qu'on n'ait plus assez de données
            while start_idx + rolling_window_days + test_window_days <= total_days:
                # Fenêtre d'entraînement: de start_idx à start_idx + rolling_window_days
                train_end = start_idx + rolling_window_days
                # Fenêtre de test: de train_end à train_end + test_window_days
                test_end = train_end + test_window_days
                
                # Créer les séquences pour la fenêtre d'entraînement
                for i in range(start_idx, train_end):
                    all_X_train.append(data["Return"].values[i-lookback_window:i])
                    all_y_train.append(1 if data["Return"].values[i] > 0 else 0)
                
                # Créer les séquences pour la fenêtre de test
                for i in range(train_end, test_end):
                    all_X_test.append(data["Return"].values[i-lookback_window:i])
                    all_y_test.append(1 if data["Return"].values[i] > 0 else 0)
                
                # Stocker les données brutes pour le scaler (on utilise la fenêtre d'entraînement)
                train_data = data["Return"].values[start_idx:train_end]
                all_stock_data.append(train_data)
                
                # Faire glisser la fenêtre (on avance de test_window_days)
                start_idx += test_window_days
                
        except Exception as e:
            print(f"  ⚠ Erreur pour {stock_info['symbol']}: {e}")
            continue
    
    if len(all_X_train) < 200:  # Pas assez de données pour entraîner
        return None
    
    all_X_train = np.array(all_X_train)
    all_y_train = np.array(all_y_train)
    all_X_test = np.array(all_X_test)
    all_y_test = np.array(all_y_test)
    
    # Normaliser sur TOUTES les données d'entraînement (rolling windows combinées)
    scaler = StandardScaler()
    # Combiner toutes les données d'entraînement pour le fit
    if all_stock_data:
        all_train_data = np.concatenate(all_stock_data)
    else:
        # Fallback: utiliser les séquences d'entraînement directement
        all_train_data = all_X_train.reshape(-1)
    scaler.fit(all_train_data.reshape(-1, 1))
    
    # Normaliser les séquences d'entraînement et de test
    X_train_reshaped = all_X_train.reshape(-1, 1)
    X_train_scaled = scaler.transform(X_train_reshaped)
    X_train_scaled = X_train_scaled.reshape(all_X_train.shape)
    
    X_test_reshaped = all_X_test.reshape(-1, 1)
    X_test_scaled = scaler.transform(X_test_reshaped)
    X_test_scaled = X_test_scaled.reshape(all_X_test.shape)
    
    # Reshape pour LSTM
    X_train = X_train_scaled.reshape((X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
    X_test = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))
    
    return (X_train, X_test, all_y_train, all_y_test, scaler)

def train_sector_model(sector_name, stocks_in_sector, data_cache=None):
    """
    Entraîner un modèle LSTM pour un secteur entier
    Utilise toutes les données des actions du secteur
    """
    print(f"\n🏭 Entraînement du modèle pour le secteur: {sector_name}")
    print(f"   Actions dans le secteur: {len(stocks_in_sector)}")
    
    # Préparer les données du secteur (utilise le cache)
    result = prepare_sector_data(stocks_in_sector, data_cache=data_cache)
    if result is None:
        print(f"  ✗ Pas assez de données pour entraîner le modèle {sector_name}")
        return None, None
    
    try:
        X_train, X_test, y_train, y_test, scaler = result
    except:
        print(f"  ✗ Erreur lors de la préparation des données pour {sector_name}")
        return None, None
    
    print(f"  ✓ {len(X_train)} séquences d'entraînement (rolling windows), {len(X_test)} séquences de test (rolling windows)")
    print(f"  📊 Méthode: ROLLING WINDOW (fenêtre glissante)")
    
    # Architecture LSTM améliorée
    model = models.Sequential([
        layers.LSTM(64, activation='tanh', input_shape=(20, 1), return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(32, activation='tanh'),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Entraînement
    model.fit(
        X_train, y_train,
        epochs=5,
        batch_size=32,
        validation_split=0.1,
        verbose=1
    )
    
    # Évaluer sur le test set
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"  ✓ Modèle entraîné - Précision test: {test_accuracy:.2%}")
    
    return model, scaler

def calculate_risk_metrics(returns):
    """
    Calculer les métriques ajustées au risque
    """
    if len(returns) == 0 or np.std(returns) == 0:
        return {
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'max_drawdown': 0.0,
            'volatility': 0.0
        }
    
    # Annualiser les métriques (252 jours de trading par an)
    trading_days = len(returns)
    annualization_factor = np.sqrt(252 / trading_days) if trading_days > 0 else 1
    
    # Volatilité annualisée
    volatility = np.std(returns) * annualization_factor
    
    # Rendement moyen annualisé
    mean_return = np.mean(returns) * 252 if trading_days > 0 else 0
    
    # Sharpe Ratio (rendement excédentaire / volatilité)
    # On assume un taux sans risque de 0% pour simplifier
    sharpe_ratio = (mean_return / volatility) if volatility > 0 else 0.0
    
    # Sortino Ratio (utilise seulement la volatilité négative)
    negative_returns = returns[returns < 0]
    downside_std = np.std(negative_returns) * annualization_factor if len(negative_returns) > 0 else volatility
    sortino_ratio = (mean_return / downside_std) if downside_std > 0 else 0.0
    
    # Maximum Drawdown
    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0.0
    
    return {
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'max_drawdown': max_drawdown,
        'volatility': volatility
    }

def predict_with_model(model, scaler, data, symbol, rolling_window_days=252, test_window_days=63):
    """
    Utiliser un modèle (secteur ou global) pour prédire une action spécifique avec ROLLING WINDOW
    """
    lookback_window = 20
    
    # Calculer les rendements
    data["Return"] = data["Close"].pct_change()
    data.dropna(inplace=True)
    
    if len(data) < lookback_window + rolling_window_days + test_window_days:
        return None
    
    # ROLLING WINDOW: utiliser la dernière fenêtre de test disponible
    total_days = len(data)
    
    # Trouver la dernière fenêtre de test possible
    test_start_idx = total_days - test_window_days
    train_start_idx = test_start_idx - rolling_window_days
    
    if train_start_idx < lookback_window:
        return None
    
    # Créer les séquences pour la fenêtre de test
    X_test, y_test = [], []
    for i in range(test_start_idx, total_days):
        X_test.append(data["Return"].values[i-lookback_window:i])
        y_test.append(1 if data["Return"].values[i] > 0 else 0)
    
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    # Normaliser avec le scaler
    X_test_reshaped = X_test.reshape(-1, 1)
    X_test_scaled = scaler.transform(X_test_reshaped)
    X_test_scaled = X_test_scaled.reshape(X_test.shape)
    
    # Reshape pour LSTM
    X_test = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))
    
    # Prédictions
    predictions = model.predict(X_test, verbose=0).flatten()
    positions = (predictions > 0.5).astype(int)
    
    # Calculer les performances sur la fenêtre de test
    returns = data["Return"].iloc[test_start_idx:total_days].values
    
    if len(returns) != len(positions):
        min_len = min(len(returns), len(positions))
        returns = returns[:min_len]
        positions = positions[:min_len]
        y_test = y_test[:min_len]
    
    strategy_returns = positions * returns
    
    strategy_total = np.cumprod(1 + strategy_returns)[-1] - 1
    buy_hold_total = np.cumprod(1 + returns)[-1] - 1
    performance = strategy_total - buy_hold_total
    accuracy = np.mean(positions == y_test)
    
    # Calculer les métriques ajustées au risque
    risk_metrics = calculate_risk_metrics(strategy_returns)
    
    return {
        'symbol': symbol,
        'accuracy': accuracy,
        'strategy_return': strategy_total,
        'buy_hold_return': buy_hold_total,
        'performance': performance,
        **risk_metrics
    }

def predict_with_sector_model(model, scaler, data, symbol, rolling_window_days=252, test_window_days=63):
    """
    Utiliser le modèle du secteur pour prédire une action spécifique avec ROLLING WINDOW
    (Wrapper pour compatibilité)
    """
    return predict_with_model(model, scaler, data, symbol, rolling_window_days, test_window_days)

# Verrou pour les écritures en base de données (SQLite n'est pas thread-safe)
db_lock = threading.Lock()

def save_to_database(conn, stock_info, results, model_type='sector'):
    """Sauvegarder les résultats (thread-safe)"""
    with db_lock:
        cursor = conn.cursor()
        
        # Déterminer la table à utiliser
        table_name = 'global_model_results' if model_type == 'global' else 'stock_results'
        
        cursor.execute(f'''
            INSERT INTO {table_name} (
                symbol, name, sector, model_type, accuracy, strategy_return, 
                buy_hold_return, performance, sharpe_ratio, sortino_ratio, 
                max_drawdown, volatility
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            results['symbol'],
            stock_info['name'],
            stock_info['sector'],
            model_type,
            results['accuracy'],
            results['strategy_return'],
            results['buy_hold_return'],
            results['performance'],
            results.get('sharpe_ratio', 0.0),
            results.get('sortino_ratio', 0.0),
            results.get('max_drawdown', 0.0),
            results.get('volatility', 0.0)
        ))
        conn.commit()

def process_stock_prediction(stock_info, model, scaler, data_cache, model_type='sector'):
    """Traiter une prédiction pour une action (utilisé pour parallélisation)"""
    try:
        symbol = stock_info['symbol']
        
        # Récupérer les données depuis le cache
        if symbol not in data_cache:
            data = download_stock_data(symbol)
            if data is None or data.empty:
                return None, f"⚠ {symbol}: Données vides"
            data_cache[symbol] = data
        else:
            data = data_cache[symbol]
        
        # Prédire avec le modèle
        results = predict_with_model(model, scaler, data, symbol)
        
        if results:
            return results, f"✓ {symbol}: Précision={results['accuracy']:.2%}, Performance={results['performance']:.2%}, Sharpe={results.get('sharpe_ratio', 0):.2f}"
        else:
            return None, f"⚠ {symbol}: Données insuffisantes"
            
    except Exception as e:
        return None, f"✗ Erreur pour {symbol}: {e}"

def train_global_model(all_stocks, data_cache=None):
    """
    Entraîner un modèle LSTM global sur TOUTES les actions
    """
    print(f"\n🌍 Entraînement du MODÈLE GLOBAL")
    print(f"   Actions dans l'univers: {len(all_stocks)}")
    
    # Préparer les données de tous les stocks
    result = prepare_sector_data(all_stocks, data_cache=data_cache)
    if result is None:
        print(f"  ✗ Pas assez de données pour entraîner le modèle global")
        return None, None
    
    try:
        X_train, X_test, y_train, y_test, scaler = result
    except:
        print(f"  ✗ Erreur lors de la préparation des données pour le modèle global")
        return None, None
    
    print(f"  ✓ {len(X_train)} séquences d'entraînement (rolling windows), {len(X_test)} séquences de test (rolling windows)")
    print(f"  📊 Méthode: ROLLING WINDOW (fenêtre glissante)")
    
    # Architecture LSTM améliorée (même que pour les secteurs)
    model = models.Sequential([
        layers.LSTM(64, activation='tanh', input_shape=(20, 1), return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(32, activation='tanh'),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Entraînement
    model.fit(
        X_train, y_train,
        epochs=5,
        batch_size=32,
        validation_split=0.1,
        verbose=1
    )
    
    # Évaluer sur le test set
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"  ✓ Modèle global entraîné - Précision test: {test_accuracy:.2%}")
    
    return model, scaler

def main():
    """Fonction principale"""
    print("=== ANALYSE DE 100 ACTIONS AMÉRICAINES (S&P 500) ===")
    print("Modèle : LSTM par secteur (Long Short-Term Memory)")
    print(f"Univers : {len(STOCKS)} actions réparties sur 11 secteurs")
    print("Architecture : 1 modèle LSTM par secteur (entraîné sur toutes les actions du secteur)")
    print("Méthode : ROLLING WINDOW (fenêtre glissante) - 252 jours train, 63 jours test")
    print()
    
    # Créer la base de données
    conn = create_database()
    
    # Fixer les seeds pour la reproductibilité
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Grouper les actions par secteur
    stocks_by_sector = defaultdict(list)
    for stock in STOCKS:
        stocks_by_sector[stock['sector']].append(stock)
    
    print(f"📊 Répartition par secteur:")
    for sector, stocks in stocks_by_sector.items():
        print(f"   {sector}: {len(stocks)} actions")
    print()
    
    # PARALLÉLISATION : Télécharger toutes les données en parallèle
    print("\n📥 TÉLÉCHARGEMENT PARALLÈLE DES DONNÉES...")
    print("-" * 70)
    data_cache = {}
    
    def download_and_cache(stock_info):
        """Télécharger et mettre en cache les données"""
        symbol = stock_info['symbol']
        try:
            data = download_stock_data(symbol, verbose=False)
            if data is not None and not data.empty:
                return symbol, data
            return symbol, None
        except Exception as e:
            return symbol, None
    
    # Télécharger toutes les données en parallèle
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(download_and_cache, stock): stock for stock in STOCKS}
        completed = 0
        for future in as_completed(futures):
            completed += 1
            symbol, data = future.result()
            if data is not None:
                data_cache[symbol] = data
            if completed % 10 == 0:
                print(f"  Progression: {completed}/{len(STOCKS)} téléchargements...")
    
    print(f"  ✓ {len(data_cache)}/{len(STOCKS)} actions téléchargées avec succès\n")
    
    all_results = []
    global_results = []
    
    # ============================================
    # ÉTAPE 1: ENTRAÎNER LE MODÈLE GLOBAL
    # ============================================
    print("\n" + "="*70)
    print("MODÈLE GLOBAL - ENTRAÎNEMENT SUR TOUS LES STOCKS")
    print("="*70)
    
    global_model, global_scaler = train_global_model(STOCKS, data_cache=data_cache)
    
    if global_model is not None:
        print(f"\n  🔄 Prédictions parallèles avec le modèle global pour {len(STOCKS)} actions...")
        
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {
                executor.submit(process_stock_prediction, stock_info, global_model, global_scaler, data_cache, 'global'): stock_info 
                for stock_info in STOCKS
            }
            
            for future in as_completed(futures):
                stock_info = futures[future]
                try:
                    results, message = future.result()
                    print(f"  {message}")
                    
                    if results:
                        # Sauvegarder en base (thread-safe)
                        save_to_database(conn, stock_info, results, model_type='global')
                        global_results.append(results)
                except Exception as e:
                    print(f"  ✗ Erreur pour {stock_info['symbol']}: {e}")
        
        print(f"  ✓ {len(global_results)}/{len(STOCKS)} actions analysées avec le modèle global")
    
    # ============================================
    # ÉTAPE 2: MODÈLES PAR SECTEUR
    # ============================================
    # Pour chaque secteur : entraîner un modèle et prédire toutes les actions
    for sector_name, stocks_in_sector in stocks_by_sector.items():
        print(f"\n{'='*70}")
        print(f"SECTEUR: {sector_name}")
        print(f"{'='*70}")
        
        # Entraîner le modèle du secteur (utilise les données du cache)
        model, scaler = train_sector_model(sector_name, stocks_in_sector, data_cache=data_cache)
        
        if model is None:
            print(f"  ⚠ Impossible d'entraîner le modèle pour {sector_name}")
            continue
        
        # PARALLÉLISATION : Prédire toutes les actions du secteur en parallèle
        print(f"\n  🔄 Prédictions parallèles pour {len(stocks_in_sector)} actions...")
        
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {
                executor.submit(process_stock_prediction, stock_info, model, scaler, data_cache, 'sector'): stock_info 
                for stock_info in stocks_in_sector
            }
            
            sector_results = []
            for future in as_completed(futures):
                stock_info = futures[future]
                try:
                    results, message = future.result()
                    print(f"  {message}")
                    
                    if results:
                        # Sauvegarder en base (thread-safe)
                        save_to_database(conn, stock_info, results, model_type='sector')
                        sector_results.append(results)
                        all_results.append(results)
                except Exception as e:
                    print(f"  ✗ Erreur pour {stock_info['symbol']}: {e}")
            
            print(f"  ✓ {len(sector_results)}/{len(stocks_in_sector)} actions analysées pour {sector_name}")
    
    # ============================================
    # ÉTAPE 3: COMPARAISON DES MODÈLES
    # ============================================
    print("\n" + "="*70)
    print("COMPARAISON DES MODÈLES - PERFORMANCE AJUSTÉE AU RISQUE")
    print("="*70)
    
    # Comparer les résultats
    df_sector = None
    df_global = None
    sector_metrics = None
    global_metrics = None
    
    if all_results and global_results:
        df_sector = pd.DataFrame(all_results)
        df_global = pd.DataFrame(global_results)
        
        # Calculer les métriques moyennes pour chaque modèle
        def calculate_aggregate_metrics(df, model_name):
            sharpe = df['sharpe_ratio'].mean() if 'sharpe_ratio' in df.columns else 0.0
            sortino = df['sortino_ratio'].mean() if 'sortino_ratio' in df.columns else 0.0
            max_dd = df['max_drawdown'].mean() if 'max_drawdown' in df.columns else 0.0
            vol = df['volatility'].mean() if 'volatility' in df.columns else 0.0
            
            return {
                'model_type': model_name,
                'total_stocks': len(df),
                'avg_accuracy': df['accuracy'].mean(),
                'avg_strategy_return': df['strategy_return'].mean(),
                'avg_buy_hold_return': df['buy_hold_return'].mean(),
                'avg_performance': df['performance'].mean(),
                'avg_sharpe_ratio': sharpe,
                'avg_sortino_ratio': sortino,
                'avg_max_drawdown': max_dd,
                'avg_volatility': vol
            }
        
        sector_metrics = calculate_aggregate_metrics(df_sector, 'sector')
        global_metrics = calculate_aggregate_metrics(df_global, 'global')
        
        # Sauvegarder les métriques de comparaison
        with db_lock:
            cursor = conn.cursor()
            for metrics in [sector_metrics, global_metrics]:
                cursor.execute('''
                    INSERT INTO model_comparison (
                        model_type, total_stocks, avg_accuracy, avg_strategy_return,
                        avg_buy_hold_return, avg_performance, avg_sharpe_ratio,
                        avg_sortino_ratio, avg_max_drawdown, avg_volatility
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    metrics['model_type'],
                    metrics['total_stocks'],
                    metrics['avg_accuracy'],
                    metrics['avg_strategy_return'],
                    metrics['avg_buy_hold_return'],
                    metrics['avg_performance'],
                    metrics['avg_sharpe_ratio'],
                    metrics['avg_sortino_ratio'],
                    metrics['avg_max_drawdown'],
                    metrics['avg_volatility']
                ))
            conn.commit()
        
        # Afficher la comparaison
        if sector_metrics and global_metrics:
            print("\n📊 MÉTRIQUES MOYENNES:")
            print("-" * 70)
            print(f"{'Métrique':<25} | {'Modèle Global':<20} | {'Modèle par Secteur':<20}")
            print("-" * 70)
            print(f"{'Nombre d\'actions':<25} | {global_metrics['total_stocks']:<20} | {sector_metrics['total_stocks']:<20}")
            print(f"{'Précision moyenne':<25} | {global_metrics['avg_accuracy']:<20.2%} | {sector_metrics['avg_accuracy']:<20.2%}")
            print(f"{'Rendement stratégie':<25} | {global_metrics['avg_strategy_return']:<20.2%} | {sector_metrics['avg_strategy_return']:<20.2%}")
            print(f"{'Rendement Buy & Hold':<25} | {global_metrics['avg_buy_hold_return']:<20.2%} | {sector_metrics['avg_buy_hold_return']:<20.2%}")
            print(f"{'Performance relative':<25} | {global_metrics['avg_performance']:<20.2%} | {sector_metrics['avg_performance']:<20.2%}")
            print()
            print("📈 MÉTRIQUES AJUSTÉES AU RISQUE:")
            print("-" * 70)
            print(f"{'Sharpe Ratio (moyen)':<25} | {global_metrics['avg_sharpe_ratio']:<20.2f} | {sector_metrics['avg_sharpe_ratio']:<20.2f}")
            print(f"{'Sortino Ratio (moyen)':<25} | {global_metrics['avg_sortino_ratio']:<20.2f} | {sector_metrics['avg_sortino_ratio']:<20.2f}")
            print(f"{'Max Drawdown (moyen)':<25} | {global_metrics['avg_max_drawdown']:<20.2%} | {sector_metrics['avg_max_drawdown']:<20.2%}")
            print(f"{'Volatilité (moyenne)':<25} | {global_metrics['avg_volatility']:<20.2%} | {sector_metrics['avg_volatility']:<20.2%}")
            print()
            
            # Déterminer le meilleur modèle
            print("🏆 CONCLUSION:")
            print("-" * 70)
            if global_metrics['avg_sharpe_ratio'] > sector_metrics['avg_sharpe_ratio']:
                print(f"  ✓ Modèle GLOBAL meilleur selon le Sharpe Ratio ({global_metrics['avg_sharpe_ratio']:.2f} vs {sector_metrics['avg_sharpe_ratio']:.2f})")
            else:
                print(f"  ✓ Modèle PAR SECTEUR meilleur selon le Sharpe Ratio ({sector_metrics['avg_sharpe_ratio']:.2f} vs {global_metrics['avg_sharpe_ratio']:.2f})")
            
            if global_metrics['avg_performance'] > sector_metrics['avg_performance']:
                print(f"  ✓ Modèle GLOBAL meilleur selon la Performance ({global_metrics['avg_performance']:.2%} vs {sector_metrics['avg_performance']:.2%})")
            else:
                print(f"  ✓ Modèle PAR SECTEUR meilleur selon la Performance ({sector_metrics['avg_performance']:.2%} vs {global_metrics['avg_performance']:.2%})")
    
    # Résumé final
    print("\n" + "="*70)
    print("RÉSULTATS FINAUX - MODÈLES PAR SECTEUR")
    print("="*70)
    
    if all_results:
        df = pd.DataFrame(all_results)
        
        print(f"\nActions analysées avec succès: {len(all_results)}/{len(STOCKS)}")
        print(f"Précision moyenne: {df['accuracy'].mean():.2%}")
        print(f"Performance moyenne: {df['performance'].mean():.2%}")
        print(f"Rendement stratégie moyen: {df['strategy_return'].mean():.2%}")
        print(f"Rendement Buy & Hold moyen: {df['buy_hold_return'].mean():.2%}")
        print()
        
        print("🏆 TOP 15 PAR PERFORMANCE (Modèles par secteur):")
        print("-" * 70)
        df_sorted = df.sort_values('performance', ascending=False)
        for i, (_, row) in enumerate(df_sorted.head(15).iterrows(), 1):
            sharpe = row.get('sharpe_ratio', 0)
            print(f"{i:2d}. {row['symbol']:6s} | Performance: {row['performance']:7.2%} | "
                  f"Stratégie: {row['strategy_return']:7.2%} | B&H: {row['buy_hold_return']:7.2%} | "
                  f"Précision: {row['accuracy']:5.2%} | Sharpe: {sharpe:5.2f}")
        
        if len(df_sorted) > 15:
            print(f"\n... et {len(df_sorted) - 15} autres actions")
        
        # Sauvegarder en CSV
        df.to_csv('results_sector.csv', index=False)
        print(f"\nRésultats modèles par secteur sauvegardés dans 'results_sector.csv'")
    
    if global_results:
        df_global = pd.DataFrame(global_results)
        df_global.to_csv('results_global.csv', index=False)
        print(f"Résultats modèle global sauvegardés dans 'results_global.csv'")
    
    # ============================================
    # ÉTAPE 4: GÉNÉRATION DES GRAPHIQUES
    # ============================================
    if all_results and global_results:
        print("\n" + "="*70)
        print("GÉNÉRATION DES GRAPHIQUES")
        print("="*70)
        generate_visualizations(df_sector, df_global, sector_metrics, global_metrics)
    
    conn.close()
    print("\nAnalyse terminée ! Base de données: 'stock_analysis.db'")

def generate_visualizations(df_sector, df_global, sector_metrics, global_metrics):
    """
    Générer 5 graphiques PNG pertinents pour l'analyse
    """
    print("\n📊 Génération des graphiques...")
    
    # 1. COMPARAISON DES PERFORMANCES MOYENNES (Modèle Global vs Par Secteur)
    fig, ax = plt.subplots(figsize=(12, 6))
    models = ['Modèle Global', 'Modèle par Secteur']
    metrics = {
        'Performance (%)': [global_metrics['avg_performance']*100, sector_metrics['avg_performance']*100],
        'Rendement Stratégie (%)': [global_metrics['avg_strategy_return']*100, sector_metrics['avg_strategy_return']*100],
        'Précision (%)': [global_metrics['avg_accuracy']*100, sector_metrics['avg_accuracy']*100]
    }
    
    x = np.arange(len(models))
    width = 0.25
    multiplier = 0
    
    for metric, values in metrics.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, values, width, label=metric)
        ax.bar_label(rects, fmt='%.2f%%', padding=3)
        multiplier += 1
    
    ax.set_ylabel('Valeur (%)', fontsize=12)
    ax.set_title('Comparaison des Performances Moyennes\nModèle Global vs Modèle par Secteur', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(models)
    ax.legend(loc='upper left', ncol=1)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('graphique_1_comparaison_performances.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Graphique 1: Comparaison des performances moyennes")
    
    # 2. DISTRIBUTION DES SHARPE RATIOS (Boxplot comparatif)
    fig, ax = plt.subplots(figsize=(10, 6))
    sharpe_global = df_global['sharpe_ratio'].fillna(0) if 'sharpe_ratio' in df_global.columns else pd.Series([0]*len(df_global))
    sharpe_sector = df_sector['sharpe_ratio'].fillna(0) if 'sharpe_ratio' in df_sector.columns else pd.Series([0]*len(df_sector))
    sharpe_data = [sharpe_global, sharpe_sector]
    box = ax.boxplot(sharpe_data, labels=['Modèle Global', 'Modèle par Secteur'], 
                     patch_artist=True, showmeans=True)
    
    # Colorier les boxplots
    colors = ['#3498db', '#e74c3c']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Sharpe Ratio', fontsize=12)
    ax.set_title('Distribution des Sharpe Ratios\nComparaison des Modèles', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('graphique_2_distribution_sharpe.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Graphique 2: Distribution des Sharpe Ratios")
    
    # 3. TOP 10 ACTIONS PAR PERFORMANCE (Modèle par Secteur)
    fig, ax = plt.subplots(figsize=(12, 7))
    top_10 = df_sector.nlargest(10, 'performance')
    colors_bar = ['#2ecc71' if x > 0 else '#e74c3c' for x in top_10['performance']]
    bars = ax.barh(range(len(top_10)), top_10['performance']*100, color=colors_bar, alpha=0.8)
    ax.set_yticks(range(len(top_10)))
    if 'sector' in top_10.columns:
        labels = [f"{row['symbol']} ({row['sector']})" for _, row in top_10.iterrows()]
    else:
        labels = [f"{row['symbol']}" for _, row in top_10.iterrows()]
    ax.set_yticklabels(labels)
    ax.set_xlabel('Performance Relative (%)', fontsize=12)
    ax.set_title('Top 10 Actions par Performance\nModèle par Secteur', fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
    ax.grid(axis='x', alpha=0.3)
    
    # Ajouter les valeurs sur les barres
    for i, (idx, row) in enumerate(top_10.iterrows()):
        ax.text(row['performance']*100 + (0.5 if row['performance'] > 0 else -1.5), 
                i, f"{row['performance']*100:.2f}%", 
                va='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('graphique_3_top10_performances.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Graphique 3: Top 10 actions par performance")
    
    # 4. SCATTER PLOT: Performance vs Sharpe Ratio (Risque/Rendement)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Modèle Global
    sharpe_global = df_global['sharpe_ratio'].fillna(0) if 'sharpe_ratio' in df_global.columns else pd.Series([0]*len(df_global))
    perf_global = df_global['performance']*100
    scatter1 = ax1.scatter(sharpe_global, perf_global, alpha=0.6, s=60, c=perf_global, 
                          cmap='RdYlGn', edgecolors='black', linewidth=0.5)
    ax1.set_xlabel('Sharpe Ratio', fontsize=11)
    ax1.set_ylabel('Performance Relative (%)', fontsize=11)
    ax1.set_title('Modèle Global\nPerformance vs Risque Ajusté', fontsize=12, fontweight='bold')
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax1.axvline(x=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax1.grid(alpha=0.3)
    plt.colorbar(scatter1, ax=ax1, label='Performance (%)')
    
    # Modèle par Secteur
    sharpe_sector = df_sector['sharpe_ratio'].fillna(0) if 'sharpe_ratio' in df_sector.columns else pd.Series([0]*len(df_sector))
    perf_sector = df_sector['performance']*100
    scatter2 = ax2.scatter(sharpe_sector, perf_sector, alpha=0.6, s=60, c=perf_sector, 
                          cmap='RdYlGn', edgecolors='black', linewidth=0.5)
    ax2.set_xlabel('Sharpe Ratio', fontsize=11)
    ax2.set_ylabel('Performance Relative (%)', fontsize=11)
    ax2.set_title('Modèle par Secteur\nPerformance vs Risque Ajusté', fontsize=12, fontweight='bold')
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax2.axvline(x=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax2.grid(alpha=0.3)
    plt.colorbar(scatter2, ax=ax2, label='Performance (%)')
    
    plt.tight_layout()
    plt.savefig('graphique_4_performance_vs_risque.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Graphique 4: Performance vs Risque ajusté (Sharpe Ratio)")
    
    # 5. COMPARAISON DES MÉTRIQUES AJUSTÉES AU RISQUE
    fig, ax = plt.subplots(figsize=(12, 7))
    risk_metrics = {
        'Sharpe Ratio': [global_metrics['avg_sharpe_ratio'], sector_metrics['avg_sharpe_ratio']],
        'Sortino Ratio': [global_metrics['avg_sortino_ratio'], sector_metrics['avg_sortino_ratio']],
        'Max Drawdown (%)': [abs(global_metrics['avg_max_drawdown'])*100, abs(sector_metrics['avg_max_drawdown'])*100],
        'Volatilité (%)': [global_metrics['avg_volatility']*100, sector_metrics['avg_volatility']*100]
    }
    
    x = np.arange(len(risk_metrics))
    width = 0.35
    models_labels = ['Modèle Global', 'Modèle par Secteur']
    
    for i, (metric, values) in enumerate(risk_metrics.items()):
        offset = width * i
        x_pos = np.array([0, 1]) + offset - width * (len(risk_metrics) - 1) / 2
        bars = ax.bar(x_pos, values, width, label=metric, alpha=0.8)
        # Ajouter les valeurs sur les barres
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.2f}' if 'Ratio' in metric else f'{val:.2f}%',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_ylabel('Valeur', fontsize=12)
    ax.set_title('Comparaison des Métriques Ajustées au Risque\nModèle Global vs Modèle par Secteur', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(models_labels)
    ax.legend(loc='upper left', ncol=2)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('graphique_5_metriques_risque.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Graphique 5: Métriques ajustées au risque")
    
    print(f"\n✅ 5 graphiques générés avec succès !")
    print("   - graphique_1_comparaison_performances.png")
    print("   - graphique_2_distribution_sharpe.png")
    print("   - graphique_3_top10_performances.png")
    print("   - graphique_4_performance_vs_risque.png")
    print("   - graphique_5_metriques_risque.png")

if __name__ == "__main__":
    main()
