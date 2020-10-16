import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from sklearn.metrics import confusion_matrix


class GaussianNB:
    def __init__(self):
        self._is_trained = False
        self._classes = []
        self._posteriors = []
        self._priors = []

    def train(self, data, labels):
        self._classes = np.array(labels.unique())
        for _ in self._classes:
            data_c = data[labels == c]
            self._priors.append(len(data_c.index) / len(data.index))
            self._posteriors.append({
                "mu": data_c.mean(),
                "cov": data_c.cov(),
            })

        self._is_trained = True

    def classify(self, data):
        if not self._is_trained:
            raise AssertionError("Model not trained")

        probabilities = []
        for i in range(len(self._classes)):
            posterior = multivariate_normal.pdf(
                data,
                mean=self._posteriors[i]["mu"],
                cov=self._posteriors[i]["cov"]
            )
            probabilities.append(self._priors[i] * posterior)
        probabilities = np.array(probabilities)
        labels_idx = np.argmax(probabilities, axis=0)
        labels = np.array([self._classes[i] for i in labels_idx])

        return labels

    def get_classes(self):
        return self._classes

    def get_confusion_matrix(self, y_true, y_pred, labels=None):
        if not self._is_trained:
            raise AssertionError("Model not trained")

        cm = confusion_matrix(y_true, y_pred, labels=None)
        df_cm = pd.DataFrame(cm, index=[i for i in self._classes],
                             columns=[i for i in self._classes])

        return df_cm

    def get_pdfs(self, grid):
        X, Y = grid
        pos = np.empty(X.shape + (2,))
        pos[:, :, 0] = X
        pos[:, :, 1] = Y
        return np.array([multivariate_normal.pdf(
            pos,
            mean=self._posteriors[i]["mu"],
            cov=self._posteriors[i]["cov"]
        ) for i in range(len(self._classes))])
