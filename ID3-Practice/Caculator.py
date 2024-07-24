import numpy as np
import pandas as pd
import math


def find_entropy(df_filter_data):
    entropy = 0
    for i in range(df_filter_data.nunique()):
        p = df_filter_data[df_filter_data == df_filter_data.unique()[i]].shape[0] / df_filter_data.shape[0]
        entropy += -(p * np.log2(p)) if p > 0 else 0
    return entropy


def entropy_and_infogain(df, feature):
    # Thực hiện tính info gain
    info = 0
    for i in range(df[feature].nunique()):
        # Lấy các hàng có nhóm giá trị thuộc cột feature GIỐNG NHAU
        df_filter_data = df[df[feature] == df[feature].unique()[i]][df.columns[-1]]
        weight = df_filter_data.shape[0] / df.shape[0]
        entropy = find_entropy(df_filter_data)
        info += weight * entropy
        print("Entropy of {} - {}: {}".format(feature, df[feature].unique()[i], find_entropy(df_filter_data)))
    print("Sum of {}'s entropy: {}".format(feature, info))
    print("Information gain of {}: {}".format(feature, find_entropy(df[df.columns[-1]]) - info))