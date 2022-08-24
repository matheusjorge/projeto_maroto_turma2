from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
import numpy as np

def model_generation():
    logistic_regression = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LogisticRegression())
        ]
    )
    
    svc = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", SVC())
        ]
    )
    
    decision_tree = Pipeline(
        [
            ("model", DecisionTreeClassifier())
        ]
    )
    
    random_forest = Pipeline(
        [
            ("model", RandomForestClassifier())
        ]
    )
    
    return [
        logistic_regression,
        svc,
        decision_tree,
        random_forest
    ]

def best_model(models, X_train, y_train):
    results = []
    for model in models:
        acc = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy").mean()
        results.append(acc)
        
    results = np.array(results)
    
    return models[np.argmax(results)]

def modeling(X_train, y_train):
    models = model_generation()
    best = best_model(models, X_train, y_train)
    
    best.fit(X_train, y_train)
    
    return best