import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def handle_continuos_data(X):
    obj_df = X.select_dtypes(include=['object'])
    if obj_df.columns == None:
        print('There are not continuous data')
        return X
    encoder = LabelEncoder()
    for i in obj_df.columns:
        obj_df[i] = encoder.fit_transform(obj_df[i])