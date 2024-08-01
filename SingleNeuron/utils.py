import numpy as np


class SingleNeuron:
    def __init__(self, epoch = 100, learning_rate = 0.01):
        self.learning_rate = learning_rate
        self.epoch = epoch

    def fit(self, X, y):
        self.w = np.zeros(X.shape[1])
        self.b = 0
        for i in range(self.epoch):
            b_temp = self.b
            for i in range(X.shape[0]):
                z = np.dot(self.w, X[i]) + self.b
                y_hat = 1 if z >= 0 else 0
                self.w += (y[i] - y_hat) * X[i]
                self.b += (y[i] - y_hat)

            # print('Times {} - pred - {} w - {} b - {}'.format(i+1, y_hat, w, b))
            if self.b == b_temp:
                break