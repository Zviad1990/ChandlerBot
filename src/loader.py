import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import BallTree
from sklearn.base import BaseEstimator
from sklearn.pipeline import make_pipeline


def prepare_data():
    lines = pd.DataFrame()
    for data in os.listdir("data"):
        if data != 'links.csv':
            temp = pd.read_csv("data/"+data)
            temp.drop("Unnamed: 0", axis=1, inplace=True)
            lines = lines.append(temp)
        else:
            pass
    lines['context'] = lines['script'].shift(1)
    chandler = lines[lines['friend'] == 'Chandler']
    chandler = chandler[~chandler['context'].isnull()]
    vector = TfidfVectorizer()
    matrix = vector.fit_transform(chandler.context)
    svd = TruncatedSVD(n_components=900)
    smal_matr = svd.fit_transform(matrix)
    return smal_matr, chandler, vector, svd


def softmax(x):
    proba = np.exp(-x)
    return proba / sum(proba)


class NeighbourCample(BaseEstimator):

    def __init__(self, k=5, temperature=1.0):
        self.k = k
        self.temperature = temperature

    def fit(self, X, y):
        self.tree_ = BallTree(X)
        self.y_ = np.array(y)

    def predict(self, X, random_state=None):
        distances, indices = self.tree_.query(X, return_distance=True, k=self.k)
        result = []
        for distance, index in zip(distances, indices):
            result.append(np.random.choice(index, p=softmax(distance * self.temperature)))
        return self.y_[result]