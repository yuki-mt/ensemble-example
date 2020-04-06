from mlens.ensemble import SuperLearner
from sklearn.metrics import accuracy_score
import numpy as np
from estimator import MyClassifier
from sklearn.preprocessing import StandardScaler

seed = 615

def get_dataset():
    X_train = np.array([
        [100, 200],
        [5, 7],
        [60, 80],
    ])
    y_train = np.array([True, True, True])
    return X_train, np.copy(X_train), y_train, np.copy(y_train)

def get_stacked_model(X, y, is_processing=True):
    ensemble = SuperLearner(scorer=accuracy_score, random_state=seed)
    preprocessers = [StandardScaler()] if is_processing else []
    ensemble.add([MyClassifier(5.0)], preprocessing=preprocessers)
    ensemble.add_meta(MyClassifier(0.5))
    ensemble.fit(X, y)
    return ensemble

def predict(model, X, y):
    preds = model.predict(X)
    print('======================')
    print(preds)


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = get_dataset()
    # prediction gets [1, 1, 1] becauase "mean of all rows > 5.0"
    model = get_stacked_model(X_train, y_train, is_processing=False)
    predict(model, X_test, y_test)

    # prediction gets [0, 0, 0] becauase "mean of all standardized rows < 5.0"
    model = get_stacked_model(X_train, y_train)
    predict(model, X_test, y_test)
