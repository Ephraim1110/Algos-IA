# Algo MLP (prévision 2025)

Ce dépôt contient un script pédagogique `algo-mlp.py` qui entraîne un petit réseau de neurones (MLP) en pur Python
pour prédire la colonne `nombre` à partir de `code_region` et `annee`.

Caractéristiques
- Pas de dépendances externes (implémentation en pur Python)
- Lecture CSV si `data.csv` est présent (colonnes attendues : `Code_region`, `annee`, `nombre`)
- Sinon, le script utilise un petit jeu de données de secours (extrait de la capture fournie)

Usage
1. Placer un CSV nommé `data.csv` à la racine (ou indiquer un autre chemin via `--csv`). Exemple de colonnes :

   Code_region,annee,nombre
   32,2020,15699
   32,2021,21570

2. Lancer le script :

```bash
python3 algo-mlp.py --csv data.csv --predict_year 2025
```

3. Pour prédire pour une région précise :

```bash
python3 algo-mlp.py --csv data.csv --predict_year 2025 --region 32
```

Notes
- Le MLP est volontairement simple (1 couche cachée). C'est un point de départ pour expérimenter.
- Pour de meilleurs résultats en production, utiliser `scikit-learn` ou `PyTorch` et préparer davantage les features (moyennes mobiles, features temporelles, regularization, validation croisée, etc.).

Prochaines améliorations possibles
- Supporter davantage de features (taux_pour_mille, population, etc.)
- Sauvegarder/charger le modèle
- Ajouter evaluation (train/test split) et hyperparam tuning
# CalclulGradient-Perceptron

Petit dépôt d'exemple contenant des scripts de préprocessing et d'exécution SQL.

## Utilisation du script SQL

Un utilitaire `execute_sql.py` permet d'exécuter le fichier `data.sql` contre la base SQLite `data.bd`.

Exemples:

- Dry-run (par défaut, rollback à la fin):

	python3 execute_sql.py

- Appliquer les changements (commit):

	python3 execute_sql.py --apply

- Spécifier une autre base ou fichier SQL:

	python3 execute_sql.py --db mydb.bd --file setup.sql

Le script affiche un aperçu du SQL exécuté et retourne des codes d'erreur en cas de problème.
# CalclulGradient-Perceptron