"""
Module pour les stratégies de trading avec modèles LSTM
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from src.data_processing import prepare_sector_data, calculate_risk_metrics


# Fixer les seeds pour la reproductibilité
tf.random.set_seed(42)
np.random.seed(42)

# Verrou pour les écritures en base de données (SQLite n'est pas thread-safe)
db_lock = threading.Lock()


def create_lstm_model(input_shape=(20, 1)):
    """
    Créer un modèle LSTM avec l'architecture standard
    
    Args:
        input_shape: Forme des données d'entrée (timesteps, features)
    
    Returns:
        Modèle Keras compilé
    """
    model = models.Sequential([
        layers.LSTM(64, activation='tanh', input_shape=input_shape, return_sequences=True),
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
    
    return model


def train_sector_model(sector_name, stocks_in_sector, data_cache=None, epochs=5):
    """
    Entraîner un modèle LSTM pour un secteur entier
    Utilise toutes les données des actions du secteur
    
    Args:
        sector_name: Nom du secteur
        stocks_in_sector: Liste des dictionnaires avec les informations des actions
        data_cache: Dictionnaire avec les données déjà téléchargées
        epochs: Nombre d'époques d'entraînement
    
    Returns:
        Tuple (model, scaler) ou (None, None) en cas d'erreur
    """
    print(f"\n🏭 Entraînement du modèle pour le secteur: {sector_name}")
    print(f"   Actions dans le secteur: {len(stocks_in_sector)}")
    
    # Préparer les données du secteur
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
    
    # Créer et entraîner le modèle
    model = create_lstm_model()
    
    model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=32,
        validation_split=0.1,
        verbose=1
    )
    
    # Évaluer sur le test set
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"  ✓ Modèle entraîné - Précision test: {test_accuracy:.2%}")
    
    return model, scaler


def train_global_model(all_stocks, data_cache=None, epochs=5):
    """
    Entraîner un modèle LSTM global sur TOUTES les actions
    
    Args:
        all_stocks: Liste de tous les dictionnaires avec les informations des actions
        data_cache: Dictionnaire avec les données déjà téléchargées
        epochs: Nombre d'époques d'entraînement
    
    Returns:
        Tuple (model, scaler) ou (None, None) en cas d'erreur
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
    
    # Créer et entraîner le modèle
    model = create_lstm_model()
    
    model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=32,
        validation_split=0.1,
        verbose=1
    )
    
    # Évaluer sur le test set
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"  ✓ Modèle global entraîné - Précision test: {test_accuracy:.2%}")
    
    return model, scaler


def predict_with_model(model, scaler, data, symbol, rolling_window_days=252, test_window_days=63):
    """
    Utiliser un modèle (secteur ou global) pour prédire une action spécifique avec ROLLING WINDOW
    
    Args:
        model: Modèle LSTM entraîné
        scaler: StandardScaler utilisé pour normaliser les données
        data: DataFrame avec les données OHLCV
        symbol: Symbole de l'action
        rolling_window_days: Taille de la fenêtre d'entraînement
        test_window_days: Taille de la fenêtre de test
    
    Returns:
        Dictionnaire avec les résultats ou None en cas d'erreur
    """
    lookback_window = 20
    
    # Calculer les rendements
    data = data.copy()
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


def process_stock_prediction(stock_info, model, scaler, data_cache, model_type='sector'):
    """
    Traiter une prédiction pour une action (utilisé pour parallélisation)
    
    Args:
        stock_info: Dictionnaire avec les informations de l'action
        model: Modèle LSTM entraîné
        scaler: StandardScaler
        data_cache: Dictionnaire avec les données téléchargées
        model_type: Type de modèle ('sector' ou 'global')
    
    Returns:
        Tuple (results, message) ou (None, message) en cas d'erreur
    """
    try:
        symbol = stock_info['symbol']
        
        # Récupérer les données depuis le cache
        if symbol not in data_cache:
            from src.fetch_data import download_stock_data
            data = download_stock_data(symbol, start_date="2020-01-01", end_date="2023-01-01", verbose=False)
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


def save_to_database(conn, stock_info, results, model_type='sector'):
    """
    Sauvegarder les résultats dans la base de données (thread-safe)
    
    Args:
        conn: Connexion SQLite
        stock_info: Dictionnaire avec les informations de l'action
        results: Dictionnaire avec les résultats de la prédiction
        model_type: Type de modèle ('sector' ou 'global')
    """
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

