import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt

df = pd.read_csv('PlayTennis.csv')
print(df.head())

X = df.drop([df.columns[-1]], axis=1)
Y = df[df.columns[-1]]

# Label Encoder
encoder = LabelEncoder()
for i in X.columns:
    X[i] = encoder.fit_transform(X[i])
Y = pd.DataFrame(encoder.fit_transform(Y), columns=[df.columns[-1]])

# Prepare samples
lenx = X.shape[0]
X_train = X.iloc[:lenx-1]
Y_train = Y.iloc[:lenx-1]
X_test = X.iloc[lenx-1].values
Y_test = Y.iloc[lenx-1].values

# Train model
clf = DecisionTreeClassifier(criterion='entropy')
clf.fit(X_train, Y_train)

# Prediction
y_pred = clf.predict(X_test.reshape(1,-1))
print(y_pred)

# Draw tree
text_representation = tree.export_text(clf)
print(text_representation)

plt.figure(figsize=(12, 8))  # Adjust the size as needed
tree.plot_tree(clf, filled=True, feature_names=df.columns)
plt.show()
