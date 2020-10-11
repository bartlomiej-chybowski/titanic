from typing import Tuple, List

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

import statsmodels.api as sm
from joblib import dump, load

from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score


class Predictor:
    def __init__(self, file: str) -> None:
        self.file = file
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.column_transformer = ColumnTransformer(
            transformers=[('encoder', OneHotEncoder(), [1, 6])],
            remainder='passthrough')
        self.scaler = StandardScaler()
        self.model = None
        self.removed = []

    def _preprocessing(self) -> Tuple:
        # load dataset
        dataset = pd.read_csv(self.file, encoding='utf-8', na_values='?')
        dataset = dataset.rename(columns=lambda item: item.strip().lower())
        x = dataset.loc[:, ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]].values
        y = dataset.loc[:, ["survived"]].values

        # missing numerical data
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        imputer.fit(x[:, [0, 2, 3, 4, 5]])
        x[:, [0, 2, 3, 4, 5]] = imputer.transform(x[:, [0, 2, 3, 4, 5]])

        # missing categorical data
        imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        imputer.fit(x[:, [1, 6]])
        x[:, [1, 6]] = imputer.transform(x[:, [1, 6]])

        self.column_transformer.fit(x)
        x = np.array(self.column_transformer.transform(x))
        return x, y

    def _split_test_train(self, x: pd.Series, y: pd.Series) -> None:
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.2,
                                                                                random_state=1)

        self.x_train.astype('float64')
        self.x_test.astype('float64')
        self.y_train.astype('float64')
        self.y_test.astype('float64')

    def _scale(self) -> None:
        self.x_train[:, 5:] = self.scaler.fit_transform(self.x_train[:, 5:])
        self.x_test[:, 5:] = self.scaler.transform(self.x_test[:, 5:])

    def _backward_elimination(self, x: np.array, sl: int = 0.05) -> Tuple:
        num_vars = len(x[0])
        for i in range(0, num_vars):
            regressor_ols = sm.OLS(self.y_train, x).fit()
            max_var = max(regressor_ols.pvalues).astype(float)
            if max_var > sl:
                for j in range(0, num_vars - i):
                    if regressor_ols.pvalues[j].astype(float) == max_var:
                        x = np.delete(x, j, 1)
                        self.removed.append(j)
        regressor_ols.summary()
        return x, self.removed

    def train(self):
        x, y = self._preprocessing()
        self._split_test_train(x, y)
        self._scale()
        x_train, columns = self._backward_elimination(self.x_train.astype('float64'))
        for column in columns:
            self.x_test = np.delete(self.x_test, column, 1)

        models = [self._decision_tree(x_train), self._svm(x_train), self._kernel_svm(x_train),
                  self._k_nn(x_train), self._logistic_regression(x_train),
                  self._naive_bayes(x_train), self._random_forest(x_train)]

        model = max(models, key=lambda item: item[1])
        dump({
            'model': model[0],
            'accuracy': model[1],
            'scaler': self.scaler,
            'removed': self.removed,
            'transformer': self.column_transformer
        }, 'Titanic/model.joblib')

    def predict(self, row: List) -> str:
        """
        Predict.

        Parameters
        ----------
        row: List[pclass: int, sex: str, age: float, sibsp: float,
                  parch: float, fare: float, embarked: str]

        Returns
        -------
        str
        """
        if self.model is None:
            model_dict = load('Titanic/model.joblib')
            self.model = model_dict['model']
            self.scaler = model_dict['scaler']
            self.removed = model_dict['removed']
            self.column_transformer = model_dict['transformer']

        # one hot encoder
        x = np.array(self.column_transformer.transform([row]))
        print(x.shape)
        # scaler
        x[:, 5:] = self.scaler.transform(x[:, 5:])
        print(x.shape)

        for column in self.removed:
            x = np.delete(x, column, 1)
        print(x.shape)

        return self._outcome(self.model.predict(x))

    @staticmethod
    def _outcome(prediction: int) -> str:
        if prediction == 0:
            return "not survived"
        elif prediction == 1:
            return "survived"
        else:
            return "error"

    def _decision_tree(self, x_train: np.array) -> Tuple:
        classifier = DecisionTreeClassifier(criterion='entropy')
        classifier.fit(x_train, self.y_train)
        y_pred = classifier.predict(self.x_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        print('Decision tree {}'.format(accuracy))
        return classifier, accuracy

    def _svm(self, x_train: np.array) -> Tuple:
        classifier = SVC(kernel='linear')
        classifier.fit(x_train, self.y_train)
        y_pred = classifier.predict(self.x_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        print('SVM {}'.format(accuracy))
        return classifier, accuracy

    def _kernel_svm(self, x_train: np.array) -> Tuple:
        classifier = SVC(kernel='rbf')
        classifier.fit(x_train, self.y_train)
        y_pred = classifier.predict(self.x_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        print('Kernel SVM {}'.format(accuracy))
        return classifier, accuracy

    def _k_nn(self, x_train: np.array) -> Tuple:
        classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
        classifier.fit(x_train, self.y_train)
        y_pred = classifier.predict(self.x_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        print('K-NN {}'.format(accuracy))
        return classifier, accuracy

    def _logistic_regression(self, x_train: np.array) -> Tuple:
        classifier = LogisticRegression(C=1)
        classifier.fit(x_train, self.y_train)
        y_pred = classifier.predict(self.x_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        print('Logistic Regression {}'.format(accuracy))
        return classifier, accuracy

    def _naive_bayes(self, x_train: np.array) -> Tuple:
        classifier = GaussianNB()
        classifier.fit(x_train, self.y_train)
        y_pred = classifier.predict(self.x_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        print('Naive Bayes {}'.format(accuracy))
        return classifier, accuracy

    def _random_forest(self, x_train: np.array) -> Tuple:
        classifier = RandomForestClassifier(n_estimators=10, criterion='entropy')
        classifier.fit(x_train, self.y_train)
        y_pred = classifier.predict(self.x_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        print('Random Forest {}'.format(accuracy))
        return classifier, accuracy
