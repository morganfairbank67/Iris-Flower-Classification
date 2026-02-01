import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import seaborn as sns

from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler

data = datasets.load_iris()
print(data.keys())

df = pd.DataFrame(data.data, columns=data.feature_names)

scaler = MinMaxScaler()

scaler.fit(df)
scaled = scaler.fit_transform(df)
scaled_df = pd.DataFrame(scaled, columns=df.columns)

scaled_df['target'] = data.target
target_dict = dict(zip([0,1,2], data.target_names))
scaled_df['target_name'] = scaled_df['target'].map(target_dict)

print(target_dict)

print(scaled_df)