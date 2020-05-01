import pyprind
import os
import tarfile
import pandas as pd
import numpy as np

with tarfile.open('aclImdb_v1.tar.gz', 'r:gz') as tar:
        tar.extractall()

basepath = 'aclImdb'

labels = {'pos': 1, 'neg': 0}
pbar = pyprind.ProgBar(50000)
df = pd.DataFrame()
for s in ('test', 'train'):
    for l in ('pos', 'neg'):
        path = os.path.join(basepath, s, l)
        for file in sorted(os.listdir(path)):
            with open(os.path.join(path, file), 
                      'r', encoding='utf-8') as infile:
                txt = infile.read()
            df = df.append([[txt, labels[l]]], 
                           ignore_index=True)
            pbar.update()
df.columns = ['review', 'sentiment']

# Shuffling the DataFrame:
np.random.seed(0)
df = df.reindex(np.random.permutation(df.index))

# Saving the assembled data as CSV file:
df.to_csv('movie_data.csv', index=False, encoding='utf-8')

df = pd.read_csv('movie_data.csv', encoding='utf-8')
print(df.head(3))
print(df.shape)