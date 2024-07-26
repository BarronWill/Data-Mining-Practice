import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn import metrics
from sklearn.model_selection import cross_val_score


def check_null(df):
    null_columns = df.columns[df.isnull().any()].tolist()
    if null_columns:
        print("Columns with null values:", null_columns)
    else:
        print("No columns have null values.")


def normalize(df):
    # Normalize numeric data
    scale = StandardScaler()
    df_num = df.select_dtypes(include=['int64', 'float64'])
    num_data = scale.fit_transform(df_num)
    df_numdata = pd.DataFrame(num_data, columns=df_num.columns)

    # Normalize categorical data
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop='first')
    df_cate = df.select_dtypes(include=['object'])
    cat_data = encoder.fit_transform(df_cate)
    df_catedata = pd.DataFrame(cat_data, columns=encoder.get_feature_names_out(df_cate.columns))

    return pd.concat([df_numdata, df_catedata], axis=1)

def report(model, x, y):
    i, v = model
    scores = cross_val_score(v, x, y, cv=10)
    accuracy = metrics.accuracy_score(y, v.predict(x))
    confusion_matrix = metrics.confusion_matrix(y, v.predict(x))
    classification = metrics.classification_report(y, v.predict(x))
    print()
    print('============================== {} Model Evaluation =============================='.format(i))
    print()
    print("Cross Validation Mean Score:" "\n", scores.mean())
    print()
    print("Model Accuracy:" "\n", accuracy)
    print()
    print("Confusion matrix:" "\n", confusion_matrix)
    print()
    print("Classification report:" "\n", classification)
    print()


