import numpy as np
import pandas as pd

class NaiveBayes:
  def __init__(self):
    pass

  def find_laplace_prob(self, df, c, labels):
    # print("{} - {} ".format(df[df == c].shape[0], df.shape[0]))
    return (df[df == c].shape[0] + 1) / (df.shape[0] + labels)


  def fit(self , df):
    self.df = df
    self.classes = df.iloc[:,-1].unique()
    self.prob = {}
    for c in self.classes:
      self.prob[c] = self.find_laplace_prob(df.iloc[:,-1], c, len(self.classes))


  def predict(self, x):
    self.px = 0
    self.pxci = {c : 1 for c in self.classes}
    self.result = {}
    df = self.df
    columns = df.columns
    for c in self.classes:
      index = 0
      for features in x:
        num_labels = df[columns[index]].nunique()
        filtered_df = df[df[columns[-1]] == c][columns[index]]
        self.pxci[c] = self.pxci[c] * self.find_laplace_prob(filtered_df, features, num_labels)
        # print("{} - {} - {} - {}".format(features, c, num_labels, self.find_laplace_prob(filtered_df, features, num_labels)))
        index = index + 1
      self.px += self.pxci[c] * self.prob[c]

    for c in self.classes:
      self.result[c] = (self.pxci[c] * self.prob[c]) / self.px