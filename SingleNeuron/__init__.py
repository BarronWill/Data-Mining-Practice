import numpy as np
from utils import *

# Dữ liệu mẫu
X = np.array([[0, 2], [1, 0], [0, -2], [2, 0]])
y = np.array([1, 1, 0, 0])

SN= SingleNeuron()
SN.fit(X, y)
print("Trọng số:", SN.w)
print("Bias:", SN.b)