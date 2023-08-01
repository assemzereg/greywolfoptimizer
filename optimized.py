import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def tanh(x):
    return np.tanh(x)


class GreyWolfOptimizer:
    def __init__(self, fitness_func, num_features, population_size=10, alpha=0.5, beta=1, delta=2):
        self.fitness_func = fitness_func
        self.num_features = num_features
        self.population_size = population_size
        self.alpha = alpha
        self.beta = beta
        self.delta = delta

        self.positions = None
        self.scores = None
        self.best_score = None
        self.best_position = None

    def initialize(self):
        self.positions = np.random.rand(
            self.population_size, self.num_features)
        self.scores = np.zeros(self.population_size)

        for i in range(self.population_size):
            self.scores[i] = self.fitness_func(self.positions[i])

        self.best_score = np.max(self.scores)
        self.best_position = self.positions[np.argmax(self.scores)]

    def update(self, epoch):
        a = 2 - epoch * ((2) / self.max_epochs)  # linearly decreasing alpha
        for i in range(self.population_size):
            for j in range(self.num_features):
                r1 = np.random.rand()
                r2 = np.random.rand()
                A1 = 2 * self.alpha * r1 - self.alpha
                C1 = 2 * r2
                D_alpha = abs(
                    C1 * self.best_position[j] - self.positions[i][j])
                X1 = self.best_position[j] - A1 * D_alpha

                r1 = np.random.rand()
                r2 = np.random.rand()
                A2 = 2 * self.alpha * r1 - self.alpha
                C2 = 2 * r2
                D_beta = abs(C2 * self.best_position[j] - self.positions[i][j])
                X2 = self.best_position[j] - A2 * D_beta

                r1 = np.random.rand()
                r2 = np.random.rand()
                A3 = 2 * self.alpha * r1 - self.alpha
                C3 = 2 * r2
                D_delta = abs(
                    C3 * self.best_position[j] - self.positions[i][j])
                X3 = self.best_position[j] - A3 * D_delta

                self.positions[i][j] = (X1 + X2 + X3) / 3

                # ensure position values are between 0 and 1
                self.positions[i][j] = np.clip(self.positions[i][j], 0, 1)

            self.scores[i] = self.fitness_func(self.positions[i])

            if self.scores[i] > self.best_score:
                self.best_score = self.scores[i]
                self.best_position = self.positions[i]

    def optimize(self, max_epochs):
        self.max_epochs = max_epochs

        self.initialize()

        for epoch in range(max_epochs):
            self.update(epoch)

        return self.best_position


def get_accuracy(X_train, X_test, y_train, y_test):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)


def feature_selection_gwo(data, target_col, fitness_func=get_accuracy, num_features=10, population_size=20, alpha=0.5, beta=1, delta=2, max_epochs=50):

    # split data into features and target
    X = data.drop(columns=[target_col])
    y = data[target_col]
    X = (X - X.min()) / (X.max() - X.min())

    # define fitness function for GWO

    def gwo_fitness_func(position):
        selected_features = X.columns[position >= 0.5]
        X_selected = X[selected_features]
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y, test_size=0.3, random_state=42)
        return fitness_func(X_train, X_test, y_train, y_test)

    # run GWO feature selection
    gwo = GreyWolfOptimizer(gwo_fitness_func, num_features,
                            population_size, alpha, beta, delta)
    selected_features = X.columns[gwo.optimize(max_epochs) >= 0.5]

    # add selected features and target back to data
    data_selected = data[selected_features]
    data_selected[target_col] = y

    return data_selected
