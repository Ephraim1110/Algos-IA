"""
Prédiction du 'nombre' pour 2025 avec un MLPRegressor (scikit-learn)
"""

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
import os
from datetime import datetime

# === Données ===
data = {
    "annee": [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
    "nombre": [18458, 18734, 18655, 18347, 15699, 21570, 24159, 23751, 25618]
}

df = pd.DataFrame(data)

# === Variables ===
X = df[['annee']].values
y = df[['nombre']].values

# === Normalisation ===
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# === Modèle MLP ===
# paramètres définis comme variables pour pouvoir les modifier facilement
hidden_layer_sizes = (16,8)
activation = 'identity'
solver = 'lbfgs'
max_iter = 5000
learning_rate = 'adaptive'
random_state = 42

model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes,
                     activation=activation,
                     solver=solver,
                     max_iter=max_iter,
                     learning_rate=learning_rate,
                     random_state=random_state)

# === Entraînement ===
model.fit(X_scaled, y_scaled.ravel())

# === Prédiction 2025 ===
annee_2025_scaled = scaler_X.transform([[2025]])
pred_scaled = model.predict(annee_2025_scaled)
pred_2025 = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))

print(f"✅ Prédiction du 'nombre' pour 2025 : {pred_2025[0][0]:.0f}")

# === Logging des paramètres et du résultat dans un CSV ===
params = {
    'hidden_layer_sizes': hidden_layer_sizes,
    'activation': activation,
    'solver': solver,
    'max_iter': max_iter,
    'learning_rate': learning_rate,
    'random_state': random_state
}

row = {
    'timestamp': datetime.now().isoformat(),
    'hidden_layer_sizes': params['hidden_layer_sizes'],
    'activation': params['activation'],
    'solver': params['solver'],
    'max_iter': params['max_iter'],
    'learning_rate': params['learning_rate'],
    'random_state': params['random_state'],
    'pred_2025': float(pred_2025[0][0]),
    'loss': getattr(model, 'loss_', None),
    'n_iter': getattr(model, 'n_iter_', None)
}

results_file = 'results.csv'
df_row = pd.DataFrame([row])
write_header = not os.path.exists(results_file)
df_row.to_csv(results_file, mode='a', index=False, header=write_header)
print(f"✅ Résultats sauvegardés dans {results_file}")
