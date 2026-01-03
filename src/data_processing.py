"""
Module pour le traitement et la préparation des données
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from collections import defaultdict


def prepare_sector_data(stocks_in_sector, rolling_window_days=252, test_window_days=63, data_cache=None):
    """
    Préparer les données d'entraînement pour un secteur avec ROLLING WINDOW
    Combine toutes les séquences de toutes les actions du secteur
    
    ROLLING WINDOW vs EXTENDING WINDOW:
    - EXTENDING: utilise toutes les données jusqu'à un point (fenêtre qui s'étend)
    - ROLLING: utilise une fenêtre de taille fixe qui glisse dans le temps (plus réaliste)
    
    Args:
        stocks_in_sector: Liste des dictionnaires avec les informations des actions du secteur
        rolling_window_days: Taille de la fenêtre d'entraînement (252 = ~1 an de trading)
        test_window_days: Taille de la fenêtre de test (63 = ~3 mois de trading)
        data_cache: Dictionnaire avec les données déjà téléchargées
    
    Returns:
        Tuple (X_train, X_test, y_train, y_test, scaler) ou None si pas assez de données
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
                data = data_cache[symbol].copy()
            else:
                from src.fetch_data import download_stock_data
                data = download_stock_data(symbol, start_date="2020-01-01", end_date="2023-01-01", verbose=False)
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
                
                # Stocker les données brutes pour le scaler
                train_data = data["Return"].values[start_idx:train_end]
                all_stock_data.append(train_data)
                
                # Faire glisser la fenêtre
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


def calculate_risk_metrics(returns):
    """
    Calculer les métriques ajustées au risque
    
    Args:
        returns: Array numpy avec les rendements quotidiens
    
    Returns:
        Dictionnaire avec sharpe_ratio, sortino_ratio, max_drawdown, volatility
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


def group_stocks_by_sector(stocks):
    """
    Grouper les actions par secteur
    
    Args:
        stocks: Liste des dictionnaires avec les informations des actions
    
    Returns:
        Dictionnaire {sector: [stock_info, ...]}
    """
    stocks_by_sector = defaultdict(list)
    for stock in stocks:
        stocks_by_sector[stock['sector']].append(stock)
    return stocks_by_sector

