from sklearn.metrics import accuracy_score

def evaluate(model, X_test, y_test):
    pred = model.predict(X_test)
    return accuracy_score(y_test, pred)