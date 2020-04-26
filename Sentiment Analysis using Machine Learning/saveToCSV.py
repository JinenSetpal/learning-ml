import pyprind
import pandas as pd
import os
import numpy as np

basepath = '/home/jinen/Downloads/aclImdb'

labels = {'pos': 1, 'neg': 0}
pbar = pyprind.ProgBar(50000)
df = pd.DataFrame()
for s in ('test', 'train'):
    for l in ('pos', 'neg'):
        for file in sorted(os.listdir(os.path.join(basepath, s, l))):
            with open(os.path.join(basepath, s, l, file), 'r', encoding='utf-8') as infile:
                txt = infile.read()
                df = df.append([[txt, labels[l]]], ignore_index=True)
                pbar.update()
df.columns = ['review', 'sentiment']

np.random.seed(0)
df = df.reindex(np.random.permutation(df.index))
df.to_csv('movie_data.csv', index=False, encoding='utf-8')
print(df.head(3))
