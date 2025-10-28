"""
🎯 Prédiction du 'nombre' pour 2025 avec RandomForestRegressor + Algorithme Génétique
(Objectif : maximiser accuracy ±5% → tendre vers 1)
"""

import numpy as np
import pandas as pd
import os
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import random
import matplotlib.pyplot as plt

# === Données ===
data = {
    "annee": [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
    "nombre": [18458, 18734, 18655, 18347, 15699, 21570, 24159, 23751, 25618]
}
df = pd.DataFrame(data)

# === Split train/test ===
X = df[['annee']].values
y = df['nombre'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# === Normalisation ===
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1,1)).ravel()
y_test_scaled = scaler_y.transform(y_test.reshape(-1,1)).ravel()

# === Custom scoring : accuracy ±5% ===
def accuracy_within_5pct_real(y_true, y_pred):
    y_true = np.array(y_true).reshape(-1,1)
    y_pred = np.array(y_pred).reshape(-1,1)
    rel_error = np.abs((y_true - y_pred) / y_true)
    return np.mean(rel_error <= 0.05)  # 1 = parfait, 0 = nul

# === Fitness function (objectif = max accuracy) ===
def fitness_function(params):
    model = RandomForestRegressor(
        n_estimators=params['n_estimators'],
        max_depth=params['max_depth'],
        min_samples_split=params['min_samples_split'],
        min_samples_leaf=params['min_samples_leaf'],
        random_state=42
    )
    model.fit(X_train_scaled, y_train_scaled)
    preds_scaled = model.predict(X_test_scaled)
    preds_real = scaler_y.inverse_transform(preds_scaled.reshape(-1,1))
    y_test_real = scaler_y.inverse_transform(y_test_scaled.reshape(-1,1))
    acc = accuracy_within_5pct_real(y_test_real, preds_real)
    return acc  # on maximise

# === Espace de recherche ===
param_space = {
    'n_estimators': [10, 50, 100, 200, 300],
    'max_depth': [2, 3, 4, 5, 6, None],
    'min_samples_split': [2, 3, 4, 5],
    'min_samples_leaf': [1, 2, 3, 4]
}

# === Paramètres GA ===
POP_SIZE = 10
N_GENERATIONS = 10
MUTATION_RATE = 0.3
ELITISM = 2

# === Fonctions auxiliaires ===
def random_params():
    return {
        'n_estimators': random.choice(param_space['n_estimators']),
        'max_depth': random.choice(param_space['max_depth']),
        'min_samples_split': random.choice(param_space['min_samples_split']),
        'min_samples_leaf': random.choice(param_space['min_samples_leaf'])
    }

# === Génération initiale ===
population = [random_params() for _ in range(POP_SIZE)]
best_scores = []

# === Boucle GA ===
for gen in range(N_GENERATIONS):
    print(f"\n🌱 Génération {gen+1}/{N_GENERATIONS}")
    
    fitness_scores = [fitness_function(p) for p in population]
    ranked = sorted(zip(population, fitness_scores), key=lambda x: x[1], reverse=True)
    
    best_acc = ranked[0][1]
    best_scores.append(best_acc)
    print(f"🧠 Meilleure accuracy ±5% : {best_acc:.3f}  avec {ranked[0][0]}")
    
    # Sélection (élitisme)
    new_population = [p for p, _ in ranked[:ELITISM]]
    
    # Croisement
    while len(new_population) < POP_SIZE:
        parent1, parent2 = random.sample(ranked[:5], 2)
        child = {}
        for key in param_space.keys():
            child[key] = random.choice([parent1[0][key], parent2[0][key]])
        new_population.append(child)
    
    # Mutation
    for indiv in new_population[ELITISM:]:
        if random.random() < MUTATION_RATE:
            key = random.choice(list(param_space.keys()))
            indiv[key] = random.choice(param_space[key])
    
    population = new_population

# === Meilleur modèle final ===
best_params, best_score = ranked[0]
print("\n✅ Meilleurs hyperparamètres trouvés :")
print(best_params)
print(f"🎯 Accuracy ±5% finale (vers 1) : {best_score:.3f}")

# === Prédiction 2025 ===
best_model = RandomForestRegressor(**best_params, random_state=42)
best_model.fit(X_train_scaled, y_train_scaled)

X_2025_scaled = scaler_X.transform([[2025]])
pred_scaled = best_model.predict(X_2025_scaled)
pred_2025 = scaler_y.inverse_transform(pred_scaled.reshape(-1,1))
print(f"\n📊 Prédiction du 'nombre' pour 2025 : {pred_2025[0][0]:.0f}")

# === Logging CSV ===
row = {
    'timestamp': datetime.now().isoformat(),
    'pred_2025': float(pred_2025[0][0]),
    'accuracy_within_5pct_test': float(best_score),
    'n_estimators': best_params['n_estimators'],
    'max_depth': str(best_params['max_depth']),
    'min_samples_split': best_params['min_samples_split'],
    'min_samples_leaf': best_params['min_samples_leaf']
}

results_file = 'results_rf_genetic.csv'
df_row = pd.DataFrame([row])
write_header = not os.path.exists(results_file)
df_row.to_csv(results_file, mode='a', index=False, header=write_header)
print(f"✅ Résultats sauvegardés dans {results_file}")

# === Visualisation évolution des générations ===
plt.figure(figsize=(8,4))
plt.plot(range(1, N_GENERATIONS+1), best_scores, marker='o')
plt.title("Évolution de l'accuracy ±5% au fil des générations")
plt.xlabel("Génération")
plt.ylabel("Accuracy (0 → 1)")
plt.grid(True)
plt.show()
