import numpy as np
import pandas as pd

class NaiveBayes:
  def __init__(self):
    # self.df = None
    self.X = None
    self.y = None
    self.prob = {}
    self.classes = None
    self.px = 0
    self.pxci = None
    self.result = {}


  def find_laplace_prob(self, df, c, labels):
    # print("{} - {} ".format(df[df == c].shape[0], df.shape[0]))
    return (df[df == c].shape[0] + 1) / (df.shape[0] + labels)


  def fit(self , X, y):
    self.X = X
    self.y = y
    self.classes = y.unique()
    # self.classes = df.iloc[:,-1].unique()
    for c in self.classes:
      self.prob[c] = self.find_laplace_prob(y, c, len(self.classes))


  def predict(self, x):
    self.pxci = {c : 1 for c in self.classes}
    columns = self.X.columns
    for c in self.classes:
      for index, feature in enumerate(x):
        num_labels = self.X[columns[index]].nunique()
        filtered_df = self.X[self.y == c][columns[index]]
        self.pxci[c] = self.pxci[c] * self.find_laplace_prob(filtered_df, feature, num_labels)
        # print("{} - {} - {} - {}".format(features, c, num_labels, self.find_laplace_prob(filtered_df, features, num_labels)))
      self.px += self.pxci[c] * self.prob[c]
    for c in self.classes:
      self.result[c] = (self.pxci[c] * self.prob[c]) / self.px
    return max(self.result, key=self.result.get)