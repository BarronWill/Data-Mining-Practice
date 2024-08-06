import numpy as np
import pandas as pd
from util import *
# Larger dataset with more categories
data = {
    "Cloud": ["It", "Nhieu", "It", "Nhieu", "Nhieu", "It", "Nhieu", "Nhieu"],
    "Pressure": ["Cao", "Cao", "Thap", "Thap", "TB", "Cao", "Cao", "Thap"],
    "Wind": ["B", "B", "B", "B", "B", "Nam", "Nam", "Nam"],
    "Rain": ["Ko", "Co", "Ko", "Co", "Co", "Ko", "Co", "Ko"]
}
df = pd.DataFrame(data)
print(df.head())
X = df.drop([df.columns[-1]], axis=1)
y = df[df.columns[-1]]
lenx = X.shape[0]
X_train = X.iloc[:lenx-1]
Y_train = y.iloc[:lenx-1]

X_test = X.iloc[lenx-1]
Y_test = y.iloc[lenx-1]

print(X_test.values)
nb = NaiveBayes()
nb.fit(X_train, Y_train)
y_pred = nb.predict(X_test.values)
print(y_pred)