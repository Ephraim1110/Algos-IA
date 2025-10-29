"""
🎯 Prédiction du 'nombre' pour 2025 (données 2016–2024)
avec XGBoost + Algorithme Génétique
(Objectif : maximiser accuracy ±5% → tendre vers 1)
"""

import numpy as np
import pandas as pd
import random
import os
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor

# ===========================
# 🔹 Jeu de données (2016–2024)
# ===========================
data = {
    "annee": [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
    "nombre": [18458, 18734, 18655, 18347, 15699, 21570, 24159, 23751, 25618]
}
df = pd.DataFrame(data)

# ===========================
# 🔹 Création des features
# ===========================
df['annee_norm'] = (df['annee'] - df['annee'].min()) / (df['annee'].max() - df['annee'].min())
df['annee_carre'] = df['annee_norm'] ** 2
X = df[['annee_norm', 'annee_carre']].values
y = df['nombre'].values

# Split temporel
train_size = int(len(df) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# ===========================
# 🔹 Normalisation
# ===========================
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

# ===========================
# 🔹 Fonction d’évaluation
# ===========================
def accuracy_within_5pct_real(y_true, y_pred):
    rel_error = np.abs((y_true - y_pred) / y_true)
    return np.mean(rel_error <= 0.05)  # 1 = parfait

# ===========================
# 🔹 Fonction de fitness
# ===========================
def fitness_function(params):
    model = XGBRegressor(
        n_estimators=params['n_estimators'],
        max_depth=params['max_depth'],
        learning_rate=params['learning_rate'],
        subsample=params['subsample'],
        colsample_bytree=params['colsample_bytree'],
        gamma=params['gamma'],
        reg_lambda=params['reg_lambda'],
        objective='reg:squarederror',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train_scaled)
    preds_scaled = model.predict(X_test_scaled)
    preds_real = scaler_y.inverse_transform(preds_scaled.reshape(-1, 1))
    y_test_real = scaler_y.inverse_transform(y_test_scaled.reshape(-1, 1))
    
    acc = accuracy_within_5pct_real(y_test_real, preds_real)
    rmse = np.sqrt(np.mean((y_test_real - preds_real) ** 2))
    return acc - 0.00001 * rmse  # on veut acc proche de 1

# ===========================
# 🔹 Espace de recherche GA
# ===========================
param_space = {
    'n_estimators': [50, 100, 150, 200],
    'max_depth': [2, 3, 4, 5],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 0.1, 0.2, 0.3],
    'reg_lambda': [0.5, 1.0, 1.5, 2.0]
}

# ===========================
# 🔹 Paramètres GA
# ===========================
POP_SIZE = 8
N_GENERATIONS = 8
MUTATION_RATE = 0.3
ELITISM = 2

def random_params():
    return {key: random.choice(values) for key, values in param_space.items()}

population = [random_params() for _ in range(POP_SIZE)]
best_scores = []

# ===========================
# 🔹 Boucle d’évolution
# ===========================
for gen in range(N_GENERATIONS):
    print(f"\n🌱 Génération {gen+1}/{N_GENERATIONS}")
    fitness_scores = [fitness_function(p) for p in population]
    ranked = sorted(zip(population, fitness_scores), key=lambda x: x[1], reverse=True)
    
    best_acc = ranked[0][1]
    best_scores.append(best_acc)
    print(f"🧠 Meilleure fitness : {best_acc:.3f} avec {ranked[0][0]}")

    # Sélection + Croisement + Mutation
    new_population = [p for p, _ in ranked[:ELITISM]]
    while len(new_population) < POP_SIZE:
        parent1, parent2 = random.sample(ranked[:5], 2)
        child = {key: random.choice([parent1[0][key], parent2[0][key]]) for key in param_space.keys()}
        new_population.append(child)
    for indiv in new_population[ELITISM:]:
        if random.random() < MUTATION_RATE:
            key = random.choice(list(param_space.keys()))
            indiv[key] = random.choice(param_space[key])
    population = new_population

# ===========================
# 🔹 Meilleur modèle final
# ===========================
best_params, best_score = ranked[0]
print("\n✅ Meilleurs hyperparamètres trouvés :")
print(best_params)
print(f"🎯 Fitness finale (vers 1) : {best_score:.3f}")

# ===========================
# 🔹 Prédiction pour 2025
# ===========================
best_model = XGBRegressor(**best_params, objective='reg:squarederror', random_state=42)
best_model.fit(X_train_scaled, y_train_scaled)

X_2025 = np.array([[2025]])
X_2025_norm = (X_2025 - df['annee'].min()) / (df['annee'].max() - df['annee'].min())
X_2025_features = np.hstack([X_2025_norm, X_2025_norm**2])
X_2025_scaled = scaler_X.transform(X_2025_features)

pred_scaled = best_model.predict(X_2025_scaled)
pred_2025 = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))
print(f"\n🔮 Prédiction du nombre pour 2025 : {pred_2025[0][0]:.0f}")

# ===========================
# 🔹 Logging CSV
# ===========================
row = {
    'timestamp': datetime.now().isoformat(),
    'pred_2025': float(pred_2025[0][0]),
    'fitness_finale': float(best_score),
    **best_params
}
results_file = 'results_xgboost_genetic_2016_2024.csv'
df_row = pd.DataFrame([row])
write_header = not os.path.exists(results_file)
df_row.to_csv(results_file, mode='a', index=False, header=write_header)
print(f"✅ Résultats sauvegardés dans {results_file}")
