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


nb = NaiveBayes()
nb.fit(df)
y_pred = nb.predict(['It', 'Thap', 'Nam'])
print(y_pred)