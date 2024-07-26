import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from util import *
from sklearn.naive_bayes import GaussianNB


test_data = pd.read_csv('income_test.csv')
train_data = pd.read_csv('income_train.csv')
print(train_data.head(5))

# Check null value
check_null(train_data)

y_train = train_data[train_data.columns[-1]]
x_train = train_data.drop([train_data.columns[-1]], axis=1)

x_test = test_data.drop([test_data.columns[-1]], axis=1)
y_test = test_data[test_data.columns[-1]]

# Normalize numeric data
x_train_norm = normalize(x_train)
x_test_norm = normalize(x_test)

# Train model
GNB = GaussianNB()
GNB.fit(x_train_norm, y_train)
report(("Gaussian Naive Bayes", GNB), x_train_norm, y_train)