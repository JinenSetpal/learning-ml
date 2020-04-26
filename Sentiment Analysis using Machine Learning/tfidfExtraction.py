import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer

count = CountVectorizer()
docs = np.array(['The sun is shining',
                 'The weather is sweet',
                 'The sun is shining and the weather is sweet',
                 'and one and one is two'])

tfdif = TfidfTransformer(use_idf=True,
                         norm='l2',
                         smooth_idf=True)
np.set_printoptions(precision=2)
print(tfdif.fit_transform(count.fit_transform(docs)).toarray())
