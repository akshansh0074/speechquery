from .writer import WorksheetWriter
import pandas as pd
import numpy as np
from random import seed
from random import randrange
from csv import reader
from math import sqrt
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model, metrics
from sklearn.model_selection import train_test_split


class Model:
    def __init__(self):
        super().__init__()
        self.recognizer = speechRecognizer()
        self.writer = WorksheetWriter()

    def load_csv(filename):
        dataset = list()
        with open(filename, 'r') as file:
            csv_reader = reader(file)
            for row in csv_reader:
                if not row:
                    continue
                dataset.append(row)
        return dataset

    def train_test_split(dataset, split):
        train = list()
        train_size = split * len(dataset)
        dataset_copy = list(dataset)
        while len(train) < train_size:
            index = randrange(len(dataset_copy))
            train.append(dataset_copy.pop(index))
        return train, dataset_copy

    def prediction(self,X,y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,
                                                            random_state=1)
        reg = linear_model.LinearRegression()
        reg.fit(X_train, y_train)
        print('Coefficients: \n', reg.coef_)
        print('Variance score: {}'.format(reg.score(X_test, y_test)))
        plt.style.use('fivethirtyeight')
        plt.scatter(reg.predict(X_train), reg.predict(X_train) - y_train,
                    color="green", s=10, label='Train data')
        plt.scatter(reg.predict(X_test), reg.predict(X_test) - y_test,
                    color="blue", s=10, label='Test data')
        plt.hlines(y=0, xmin=0, xmax=50, linewidth=2)
        plt.legend(loc='upper right')
        plt.title("Residual errors")
        plt.show()




