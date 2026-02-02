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

print(scaled_df)

# plt.figure(figsize=(7,7))
# plt.pie(scaled_df.target.value_counts(), labels=data.target_names, textprops={'fontsize': 14}, colors=['r','b','g'], autopct= lambda x : '{p:.2f}% ({v:.1f})'.format(p=x, v = x*len(scaled_df)/100))
# plt.show()

# means = scaled_df.groupby(['target']).mean(numeric_only=True)
# print(means)

# features = means.columns
# x_axis = np.array([1,2,3,4])
# y_axis = means.values

# plt.figure(figsize=(10,5))
# plt.bar(x_axis-.25, y_axis[0], width=0.25, color='r')
# plt.bar(x_axis, y_axis[1], width=0.25, color='b')
# plt.bar(x_axis+.25, y_axis[2], width=0.25, color='g')
# plt.xticks(x_axis, features)
# plt.legend(data.target_names)

# plt.show()

# sns.pairplot(scaled_df[scaled_df.columns[[0,1,2,3,5]]], hue='target_name', palette='Set1', kind='kde')
# plt.show()

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

X = scaled_df.values[:,:4].astype(np.float32())
Y = scaled_df.values[:,4].astype(int)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, random_state=1)

print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

clf = DecisionTreeClassifier(random_state=1)
print(clf.fit(X_train, Y_train).score(X_test, Y_test))

params = {'criterion':['gini', 'entropy'], 'splitter':['best', 'random'], 'max_depth':np.linspace(5,90,18).astype(np.int32), 'min_samples_split':np.arange(2,10), 'max_features':['auto', 'sqrt', 'log2']}

rand_search = RandomizedSearchCV(DecisionTreeClassifier(random_state=1), params, scoring='accuracy', random_state=1, cv=5)
rand_search.fit(X_train, Y_train)
rand_params = rand_search.best_params_
print(rand_params, '/n')
print('Train Acc: ', rand_search.best_score_)
preds = rand_search.predict(X_test)
print('Test ACC: ', accuracy_score(preds, Y_test))