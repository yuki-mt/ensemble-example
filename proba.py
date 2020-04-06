from sklearn import datasets
from sklearn.model_selection import train_test_split
from mlens.ensemble import SuperLearner
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

seed = 532

def accuracy(y, p):
    if isinstance(p[0], np.ndarray):
        p = p.argmax(axis=1)
    return accuracy_score(y, p)

def get_dataset():
    iris = datasets.load_iris()
    return train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=seed, stratify=iris.target
    )

def get_stacked_model(X, y):
    ensemble = SuperLearner(scorer=accuracy, random_state=seed)
    # call predict_proba instead of predict
    ensemble.add([SVC(probability=True), RandomForestClassifier(random_state=seed)], proba=True)
    ensemble.add_meta(LogisticRegression())
    ensemble.fit(X, y)
    print('accuracy score in training')
    print('-m: mean. -s: std')
    print(pd.DataFrame(ensemble.data))
    return ensemble

def predict(model, X, y):
    preds = model.predict(X)
    print('======================')
    print('accuracy score in test')
    print(accuracy_score(preds, y))


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = get_dataset()
    model = get_stacked_model(X_train, y_train)
    predict(model, X_test, y_test)
