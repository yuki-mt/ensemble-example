from sklearn import datasets
from sklearn.model_selection import train_test_split
from mlens.ensemble import SuperLearner
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score
import pandas as pd

seed = 532
def f1(y, p):
    return f1_score(y, p, average='micro')

def get_dataset():
    iris = datasets.load_iris()
    return train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=seed, stratify=iris.target
    )

def get_stacked_model(X, y):
    ensemble = SuperLearner(scorer=f1, random_state=seed)
    ensemble.add([RandomForestClassifier(random_state=seed), SVC()])
    ensemble.add_meta(LogisticRegression())
    ensemble.fit(X, y)
    print('f1-score in training')
    print('-m: mean. -s: std')
    print(pd.DataFrame(ensemble.data))
    return ensemble

def predict(model, X, y):
    preds = model.predict(X)
    print('======================')
    print('f1-score in test')
    print(f1(preds, y))


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = get_dataset()
    model = get_stacked_model(X_train, y_train)
    predict(model, X_test, y_test)
