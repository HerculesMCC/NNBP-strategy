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

# Supprimer les warnings de yfinance
warnings.filterwarnings("ignore")

# Téléchargement des données avec gestion d'erreur
try:
    print("Téléchargement des données Yahoo Finance...")
    data = yf.download("AAPL", start="2018-01-01", end="2023-01-01", progress=False)
    
    # Vérifier si les données sont vides
    if data.empty:
        raise ValueError("Données vides reçues de Yahoo Finance")
        
    print(f"Données téléchargées avec succès: {len(data)} lignes")
    
except Exception as e:
    print(f"Erreur lors du téléchargement: {e}")
    print("Génération de données simulées...")
    
    # Générer des données simulées comme fallback
    np.random.seed(42)
    dates = pd.date_range(start="2018-01-01", end="2023-01-01", freq="D")
    n_days = len(dates)
    
    # Simulation d'un prix d'action avec marche aléatoire
    returns = np.random.normal(0.0005, 0.02, n_days)  # rendement moyen 0.05% par jour, volatilité 2%
    prices = 100 * np.cumprod(1 + returns)  # prix initial à 100
    
    data = pd.DataFrame({
        'Open': prices * (1 + np.random.normal(0, 0.005, n_days)),
        'High': prices * (1 + np.abs(np.random.normal(0, 0.01, n_days))),
        'Low': prices * (1 - np.abs(np.random.normal(0, 0.01, n_days))),
        'Close': prices,
        'Volume': np.random.randint(1000000, 10000000, n_days)
    }, index=dates)
    
    print(f"Données simulées générées: {len(data)} lignes")

data["Return"] = data["Close"].pct_change()
data.dropna(inplace=True)

data["Target"] = (data["Return"] > 0).astype(int)

#deterministic
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)
random.seed(seed)

lookback = 5
X, y = [], []
for i in range(lookback, len(data)):
    X.append(data["Return"].values[i-lookback:i])
    y.append(data["Target"].values[i])

X = np.array(X)
y = np.array(y)

# Vérifier que nous avons des données
if len(X) == 0:
    print("Erreur: Aucune donnée disponible pour l'entraînement")
    exit(1)

print(f"Données préparées: {len(X)} échantillons, {X.shape[1]} features")

# Standardisation
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split train / test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

print(f"Données d'entraînement: {len(X_train)} échantillons")
print(f"Données de test: {len(X_test)} échantillons")

# Création du modèle
model = models.Sequential([
    layers.Dense(32, activation="relu", input_shape=(X_train.shape[1],)),
    layers.Dropout(0.2),  # Ajout de dropout pour éviter l'overfitting
    layers.Dense(16, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(1, activation="sigmoid")  # sortie = proba
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

print("Modèle créé avec succès")
print(f"Architecture du modèle: {model.count_params()} paramètres")

# Entraînement
print("Début de l'entraînement...")
history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                    epochs=20, batch_size=32, verbose=1)

# Prédictions
proba = model.predict(X_test, verbose=0).flatten()
positions = (proba > 0.5).astype(int)  # 1 si on prend position

# Calcul des rendements
returns = data["Return"].iloc[-len(y_test):].values
strategy_returns = positions * returns

# Métriques de performance
strategy_cumulative = np.cumprod(1 + strategy_returns)[-1] - 1
buy_hold_cumulative = np.cumprod(1 + returns)[-1] - 1

print("\n" + "="*50)
print("RÉSULTATS DE LA STRATÉGIE")
print("="*50)
print(f"Rendement cumulé de la stratégie: {strategy_cumulative:.4f} ({strategy_cumulative*100:.2f}%)")
print(f"Rendement cumulé buy & hold: {buy_hold_cumulative:.4f} ({buy_hold_cumulative*100:.2f}%)")
print(f"Performance relative: {strategy_cumulative - buy_hold_cumulative:.4f} ({(strategy_cumulative - buy_hold_cumulative)*100:.2f}%)")

# Métriques supplémentaires
accuracy = np.mean(positions == y_test)
print(f"Précision de la stratégie: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Nombre de positions prises: {np.sum(positions)}/{len(positions)} ({np.sum(positions)/len(positions)*100:.1f}%)")

# Volatilité
strategy_vol = np.std(strategy_returns) * np.sqrt(252)  # annualisée
buy_hold_vol = np.std(returns) * np.sqrt(252)
print(f"Volatilité stratégie (annualisée): {strategy_vol:.4f} ({strategy_vol*100:.2f}%)")
print(f"Volatilité buy & hold (annualisée): {buy_hold_vol:.4f} ({buy_hold_vol*100:.2f}%)")