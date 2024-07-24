import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import Caculator
import math


df = pd.read_csv('PlayTennis.csv')
print(df.head(5))

col = df.columns[:-1]
for i in col:
    Caculator.entropy_and_infogain(df, i)
    print('-----------------------------')
