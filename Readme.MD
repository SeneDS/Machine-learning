Nous avons vu comment implémenter des modèles de régression et de classification, les évaluer, les comparer, et les intégrer dans des pipelines de feature engineering. Maintenant, explorons une méthode avancée appelée stacking.

Le stacking (ou empilement) est une technique d’ensemble learning où plusieurs modèles sont combinés pour améliorer les performances globales. Les prédictions de plusieurs modèles de base (également appelés "modèles de niveau 0") sont utilisées comme caractéristiques d’entrée pour un modèle de "niveau 1", appelé métamodèle.

*CF*[L'appli streamlit](https://machine-learning-edftkeykwjbfnr7dsab6cj.streamlit.app)

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
- [10. Le StackingRegressor ?](#10-le-stackingregressor)


<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## 1. Import des bibliotheques
```
# Bibliothèques standards
import numpy as np
import pandas as pd

# Importation des jeux de données
from sklearn.datasets import fetch_california_housing, load_iris

# Séparation des données (train/test)
from sklearn.model_selection import train_test_split

# Prétraitement des données
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Pipelines
from sklearn.pipeline import Pipeline, make_pipeline

# Modèles de Machine Learning
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import StackingClassifier


# Évaluation et validation des modèles
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score
```


## 2. Chargement et préparation des données
```
# Chargement du jeu de données Iris
iris = load_iris(as_frame=True)
df_iris=iris.frame
X=df_iris.drop("target", axis=1)
y=df_iris["target"]
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
    grid_search = GridSearchCV(pipeline, param_grid, scoring='accuracy', cv=3, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_estimator = grid_search.best_estimator_
    best_params=grid_search.best_params_

    # Prédire sur les données de test
    y_pred = best_estimator.predict(X_test)
    test_score = accuracy_score(y_test, y_pred)

    results.append({
        'model': model_name,
        'best_estimator': best_estimator,
        'best_score': test_score,
        'best_params': best_params
    })

    print(f"Model: {model_name}, Test Score (Accuracy): {test_score}, Best Params: {grid_search.best_params_}")
```

## 6. Définition des modèles et des grilles d'hyperparamètres
```
# Modèles et leurs grilles de paramètres ajustées à tester
models = [
    {
        'name':  'Logistic Regression',
        'pipeline': Pipeline(steps=[('preprocessor', preprocessor), ('classifier', LogisticRegression(max_iter=5000))]),
        'param_grid': {
            'classifier__C' : [0.1, 1, 10]
        }
    },
    {
        'name': 'KNN',
        'pipeline': Pipeline(steps=[('preprocessor', preprocessor), ('classifier', KNeighborsClassifier())]),
        'param_grid': {
            'classifier__n_neighbors': [3, 5, 7]
        }
    },
    {
        'name': 'SVM',
        'pipeline': Pipeline(steps=[('preprocessor', preprocessor), ('classifier', SVC(probability=True))]),
        'param_grid': {
            'classifier__C': [0.1, 1, 10], 'classifier__kernel':['linear', 'rbf']
        }
    },
    {
        'name': 'Random Forest ',
        'pipeline': Pipeline(steps=[('preprocessor', preprocessor), ('classifier', RandomForestClassifier(random_state=42))]),
        'param_grid': { 
            'classifier__n_estimators': [10, 50, 100], 
            'classifier__max_depth': [5, 10, 20],
            'classifier__max_samples': [0.5, 1.0],
            'classifier__max_features': [0.5, 1.0]
        }
    },
    {
        'name': 'Stacking Classifier',
        'pipeline': Pipeline(steps=[
            ('preprocessor', preprocessor), 
            ('classifier', StackingClassifier(
                estimators=[
                    ('logreg', LogisticRegression(max_iter=5000)),
                    ('knn', KNeighborsClassifier()),
                    ('svm', SVC(probability=True)),
                    ('rf', RandomForestClassifier(random_state=42))
                ],
                final_estimator=LogisticRegression(max_iter=5000),
                cv=5,
                n_jobs=-1
            ))
        ]),
        'param_grid': {}  # Pas de paramètres à ajuster ici (par défaut)
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
## 10. Le StackingRegressor ?

```
from sklearn.ensemble import StackingRegressor

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
            'poly__interaction_only': [False]
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
        'name': 'Stacking Regressor',
        'pipeline': Pipeline(steps=[
            ('preprocessor', preprocessor), 
            ('regressor', StackingRegressor(
                estimators=[
                    ('elasticnet', ElasticNet(max_iter=10000)),
                    ('randomforest', RandomForestRegressor(n_estimators=100)),
                    ('gradientboost', GradientBoostingRegressor(n_estimators=100, max_depth=3))
                ],
                final_estimator=GradientBoostingRegressor(n_estimators=50),
                n_jobs=-1
            ))
        ]),
        'param_grid': {}  # Pas d'hyperparamètres à ajuster ici
    }
]

```
Oubien

```
final_estimators = [
    ('ElasticNet', ElasticNet(alpha=0.1, l1_ratio=0.5)),
    ('RandomForest', RandomForestRegressor(n_estimators=50, max_depth=5)),
    ('GradientBoosting', GradientBoostingRegressor(n_estimators=50))
]

for name, estimator in final_estimators:
    stacking = StackingRegressor(
        estimators=[
            ('elasticnet', ElasticNet(max_iter=10000)),
            ('randomforest', RandomForestRegressor(n_estimators=100)),
            ('gradientboost', GradientBoostingRegressor(n_estimators=100))
        ],
        final_estimator=estimator
    )
    scores = cross_val_score(stacking, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    print(f"{name}: Mean MSE = {-scores.mean()}")
```

