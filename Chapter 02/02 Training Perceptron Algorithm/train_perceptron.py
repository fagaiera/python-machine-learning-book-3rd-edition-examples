from perceptron import Perceptron
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""Read the data frame"""
df = pd.read_csv('iris.data', header=None)

# select setosa and versicolor
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)

# extract sepal length and petal length
X = df.iloc[0:100, [0, 2]].values

ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(X, y)

fig = plt.figure()
fig.canvas.set_window_title('Training Perceptron Algorithm')

plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of updates')

plt.show()