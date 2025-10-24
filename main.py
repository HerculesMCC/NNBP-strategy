import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import random


data = yf.download("AAPL", start="2018-01-01", end="2023-01-01")

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

# Standardisation
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split train / test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)


model = models.Sequential([
    layers.Dense(32, activation="relu", input_shape=(X_train.shape[1],)),
    layers.Dense(16, activation="relu"),
    layers.Dense(1, activation="sigmoid")  # sortie = proba
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])


history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                    epochs=20, batch_size=32, verbose=1)

proba = model.predict(X_test).flatten()

positions = (proba > 0.5).astype(int)  # 1 si on prend position

returns = data["Return"].iloc[-len(y_test):].values
strategy_returns = positions * returns

print("Rendement cumulé stratégie :", np.cumprod(1+strategy_returns)[-1] - 1)