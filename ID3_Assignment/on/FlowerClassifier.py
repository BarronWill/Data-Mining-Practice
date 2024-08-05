import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn import metrics
import seaborn as sns


# Prepare dataset
X_train = pd.read_csv('xtrainV3.csv')
X_test = pd.read_csv('xtestV3.csv').values
Y_train = pd.read_csv('ytrainV3.csv')
Y_test = pd.read_csv('ytestV3.csv').values

df = pd.concat([X_train, Y_train], axis=1)
clf = DecisionTreeClassifier().fit(X_train, Y_train)
y_pred = clf.predict(X_test)

print(metrics.accuracy_score(y_pred, Y_test))
print(metrics.confusion_matrix(Y_test, y_pred))
# plt.figure(figsize=(18,7))
# sns.pairplot(df, hue=df.columns[-1])
# plt.show()

# tree.plot_tree(clf, filled=True, class_names=X_train.columns )
# plt.show()