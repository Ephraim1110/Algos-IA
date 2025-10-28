"""
Prédiction du 'nombre' pour 2025 avec un MLPRegressor + Algorithme Génétique (optimisation de l'accuracy)
"""

import numpy as np
import pandas as pd
import os
from datetime import datetime
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import random

# === Données ===
data = {
    "annee": [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
    "nombre": [18458, 18734, 18655, 18347, 15699, 21570, 24159, 23751, 25618]
}
df = pd.DataFrame(data)

# === Split train/test ===
X = df[['annee']].values
y = df[['nombre']].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# === Normalisation ===
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train)

# ============================================================
#      ⚙️ Algorithme génétique pour optimiser l'accuracy
# ============================================================

def evaluate_mlp(params):
    """Évalue un MLPRegressor et renvoie -accuracy (à maximiser)."""
    hidden_layer_sizes, activation, solver, learning_rate = params

    model = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        solver=solver,
        learning_rate=learning_rate,
        max_iter=5000,
        random_state=42
    )

    model.fit(X_train_scaled, y_train_scaled.ravel())
    y_pred_scaled = model.predict(X_test_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))
    y_true = y_test.reshape(-1, 1)

    # Accuracy (tolérance de 5%)
    tolerance_pct = 0.05
    rel_error = np.abs((y_true - y_pred) / y_true)
    accuracy_within_5pct = np.mean(rel_error <= tolerance_pct)

    return -accuracy_within_5pct, model


# --- Espaces de recherche ---
hidden_layer_choices = [
    (8,), (16,), (32,), (16, 8), (32, 16), (64, 32), (32, 16, 8)
]
activation_choices = ['identity', 'relu', 'tanh']
solver_choices = ['lbfgs', 'adam']
learning_rate_choices = ['constant', 'adaptive']

# --- Paramètres GA --- 
POP_SIZE = 10
N_GENERATIONS = 8
MUTATION_RATE = 0.3
ELITISM = 2

def random_individual():
    return (
        random.choice(hidden_layer_choices),
        random.choice(activation_choices),
        random.choice(solver_choices),
        random.choice(learning_rate_choices)
    )

def mutate(individual):
    ind = list(individual)
    i = random.randint(0, len(ind) - 1)
    if i == 0:
        ind[i] = random.choice(hidden_layer_choices)
    elif i == 1:
        ind[i] = random.choice(activation_choices)
    elif i == 2:
        ind[i] = random.choice(solver_choices)
    else:
        ind[i] = random.choice(learning_rate_choices)
    return tuple(ind)

def crossover(parent1, parent2):
    return tuple(random.choice(genes) for genes in zip(parent1, parent2))

# === Boucle GA ===
population = [random_individual() for _ in range(POP_SIZE)]
best_score = np.inf  # ici = -accuracy (donc plus petit = meilleur)
best_params = None
best_model = None

for gen in range(N_GENERATIONS):
    print(f"\n=== Génération {gen+1}/{N_GENERATIONS} ===")
    scores = []
    for ind in population:
        score, model = evaluate_mlp(ind)
        scores.append((score, ind, model))
        print(f"Individu {ind} → Accuracy = {-score:.3f}")

    # Trie par accuracy décroissante (car score = -accuracy)
    scores.sort(key=lambda x: x[0])
    if scores[0][0] < best_score:
        best_score, best_params, best_model = scores[0]

    print(f"→ Meilleure accuracy de la génération : {-scores[0][0]:.3f}")

    # Sélection élitiste
    next_gen = [ind for (_, ind, _) in scores[:ELITISM]]

    # Reproduction
    while len(next_gen) < POP_SIZE:
        parent1, parent2 = random.sample(scores[:5], 2)
        child = crossover(parent1[1], parent2[1])
        if random.random() < MUTATION_RATE:
            child = mutate(child)
        next_gen.append(child)

    population = next_gen

print("\n✅ Meilleurs hyperparamètres trouvés :")
print(best_params)
print(f"Accuracy : {-best_score:.3f}")

# ============================================================
#         📈 Prédiction finale pour 2025 avec le meilleur modèle
# ============================================================

annee_2025_scaled = scaler_X.transform([[2025]])
pred_scaled = best_model.predict(annee_2025_scaled)
pred_2025 = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))
print(f"\n📊 Prédiction du 'nombre' pour 2025 : {pred_2025[0][0]:.0f}")

# ============================================================
#         🗂️ Logging dans results_genetic.csv
# ============================================================

row = {
    'timestamp': datetime.now().isoformat(),
    'best_hidden_layer_sizes': best_params[0],
    'activation': best_params[1],
    'solver': best_params[2],
    'learning_rate': best_params[3],
    'pred_2025': float(pred_2025[0][0]),
    'accuracy_within_5pct': float(-best_score)
}

results_file = 'results_genetic.csv'
df_row = pd.DataFrame([row])
write_header = not os.path.exists(results_file)
df_row.to_csv(results_file, mode='a', index=False, header=write_header)
print(f"✅ Résultats sauvegardés dans {results_file}")
