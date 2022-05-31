from sklearn.datasets import load_iris
from perceptron import Perceptron
import numpy as np
import pandas as pd
import pickle



iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
df = df.loc[df.target != 2, ['sepal length (cm)', 'petal length (cm)', 'target']]
pcp = Perceptron(n_iter=10)
pcp.fit(df.iloc[:, :2].values, df.target.values)
with open("RTA_model_pick.pkl", 'wb') as f:
    pickle.dump(pcp, f)

