Projet de Machine Learning: Data-scientist-salary


Ce projet de machine learning se concentre sur plusieurs étapes essentielles, notamment le choix de la variable cible, le rééquilibrage de sa distribution, la mise en place d'une pipeline pour le traitement des données, la sélection du meilleur modèle, et son enregistrement pour le déploiement. [Cf. le Script](Script.ipynb).


---

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Plan**  *

- [1. Import des bibliotheques](#1-import-des-bibliotheques)
- [2. Chargement et préparation des données](#2-chargement-et-préparation-des-données)
- [3. Séparation des données en ensembles d'entraînement et de test](#3-séparation-des-données-en-ensembles-dentraînement-et-de-test)
- [4. Définition du prétraitement des données](#4-définition-du-prétraitement-des-données)
- [5. Fonction d'évaluation des modèles](#5-fonction-dévaluation-des-modèles)
- [6. Définition des modèles et des grilles d'hyperparamètres](#6-définition-des-modèles-et-des-grilles-dhyperparamètres)
- [7. Entraînement et évaluation des modèles](#7-entraînement-et-évaluation-des-modèles)
- [8. Sélection et affichage du meilleur modèle](#8-sélection-et-affichage-du-meilleur-modèle)
- [9. Sauvegarde et rechargement du meilleur modèle](#9-sauvegarde-et-rechargement-du-meilleur-modèle)


<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## 1. Import des bibliotheques
```
# Bibliothèques standards
import numpy as np
import pandas as pd

# Séparation des données (train/test)
from sklearn.model_selection import train_test_split

# Prétraitement des données
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Pipelines
from sklearn.pipeline import Pipeline, make_pipeline

# Modèles de Machine Learning
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import PolynomialFeatures

# Évaluation et validation des modèles
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error
```


## 2. Chargement et préparation des données
```
# Chargement du jeu de données Iris
df=pd.read_csv("/Users/etienne/Documents/Data-scientist-salary/Data-Scientist-Salary-app/Données/salaries.csv")
# Préparation des données
y = np.log1p(df['salary_in_usd'])
#Selection des variables
X = df.drop(["salary_in_usd", 'salary'], axis=1)
```


## 3. Séparation des données en ensembles d'entraînement et de test

```
# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

## 4. Définition du prétraitement des données

```
# Définir le prétraitement des variables
numeric_features=X.select_dtypes(exclude="object").columns.tolist()
numeric_transformer = StandardScaler()

categorical_features=X.select_dtypes(include="object").columns.tolist()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])
```

## 5. Fonction d'évaluation des modèles

```
# Fonction pour évaluer les models sur les données test et enregistrer les résultats
def evaluate_model(model_name, pipeline, param_grid):
    grid_search = GridSearchCV(pipeline, param_grid, scoring='neg_mean_squared_error', cv=3, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_estimator = grid_search.best_estimator_
    best_params=grid_search.best_params_

    # Prédire sur les données de test
    y_pred = best_estimator.predict(X_test)
    test_score = mean_squared_error(y_test, y_pred)

    results.append({
        'model': model_name,
        'best_estimator': best_estimator,
        'best_score': test_score,
        'best_params': best_params
    })

    print(f"Model: {model_name}, Test Score (MSE): {test_score}, Best Params: {grid_search.best_params_}")
```

## 6. Définition des modèles et des grilles d'hyperparamètres
```
# Modèles et leurs grilles de paramètres ajustées
models = [
    {
        'name': 'k-NN Regression',
        'pipeline': Pipeline(steps=[('preprocessor', preprocessor), ('regressor', KNeighborsRegressor())]),
        'param_grid': {
            'regressor__n_neighbors': [3, 5, 7]
        }
    },
    {
        'name': 'Polynomial Regression',
        'pipeline': Pipeline(steps=[('preprocessor', preprocessor), ('poly', PolynomialFeatures()), ('regressor', LinearRegression())]),
        'param_grid': {
            'poly__degree': [2, 3],
            'poly__interaction_only': [False]  # Utiliser uniquement False pour réduire la complexité
        }
    },
    {
        'name': 'ElasticNet',
        'pipeline': Pipeline(steps=[('preprocessor', preprocessor), ('regressor', ElasticNet(max_iter=10000))]),
        'param_grid': {
            'regressor__alpha': [0.1, 1.0, 10.0],
            'regressor__l1_ratio': [0.1, 0.5, 0.9]
        }
    },
    {
        'name': 'BaggingRegressor',
        'pipeline': Pipeline(steps=[('preprocessor', preprocessor), ('regressor', BaggingRegressor())]),
        'param_grid': {
            'regressor__n_estimators': [10, 50, 100],
            'regressor__max_samples': [0.5, 1.0],
            'regressor__max_features': [0.5, 1.0]
        }
    },
    {
        'name': 'RandomForestRegressor',
        'pipeline': Pipeline(steps=[('preprocessor', preprocessor), ('regressor', RandomForestRegressor())]),
        'param_grid': {
            'regressor__n_estimators': [50, 100],
            'regressor__max_depth': [10, 20],
            'regressor__criterion': ['squared_error', 'absolute_error']
        }
    },
    {
        'name': 'GradientBoostingRegressor',
        'pipeline': Pipeline(steps=[('preprocessor', preprocessor), ('regressor', GradientBoostingRegressor())]),
        'param_grid': {
            'regressor__n_estimators': [50, 100],
            'regressor__learning_rate': [0.01, 0.1],
            'regressor__max_depth': [3, 5],
            'regressor__loss': ['squared_error', 'absolute_error', 'huber']
        }
    },
    {
        'name': 'AdaBoostRegressor',
        'pipeline': Pipeline(steps=[('preprocessor', preprocessor), ('regressor', AdaBoostRegressor())]),
        'param_grid': {
            'regressor__n_estimators': [50, 100],
            'regressor__learning_rate': [0.01, 0.1],
            'regressor__loss': ['linear', 'square', 'exponential']
        }
    },
    {
        'name': 'SVR',
        'pipeline': Pipeline(steps=[('preprocessor', preprocessor), ('regressor', SVR())]),
        'param_grid': {
            'regressor__kernel': ['linear', 'rbf'],
            'regressor__C': [0.1, 1, 10],
            'regressor__epsilon': [0.01, 0.1, 0.5]
        }
    }
]
```

## 7. Entraînement et évaluation des modèles
```
# Liste pour stocker les résultats des modèles
results = []
# Évaluation des modèles
for model in models:
    evaluate_model(model['name'], model['pipeline'], model['param_grid'])

```

## 8. Sélection et affichage du meilleur modèle

```
# Afficher les résultats des meilleurs modèles
best_model = min(results, key=lambda x: x['best_score'])
print("\nBest Model:")
print(f"Model: {best_model['model']}, Test Score (MSE): {best_model['best_score']}, Best Params: {best_model['best_params']}")
```

## 9. Sauvegarde et rechargement du meilleur modèle

```
# Enregistrer le meilleur modèle
joblib.dump(best_model['best_estimator'], 'meilleur_modele.pkl')
# Charger le modèle enregistré
meilleur_modele = joblib.load('meilleur_modele.pkl')

# Prédire avec le modèle chargé
predictions = meilleur_modele.predict(X_test)
```