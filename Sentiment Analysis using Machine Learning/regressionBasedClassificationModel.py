from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import pandas as pd


def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]


def tokenizer(test):
    return test.split()


porter = PorterStemmer()

df = pd.read_csv('movie_data.csv', header=None)
df.columns = ['review', 'sentiment']

X_train = df.loc[:25000, 'review'].values
X_test = df.loc[25000:, 'review'].values
y_train = df.loc[:25000, 'sentiment'].values
y_test = df.loc[25000:, 'sentiment'].values

tfidf = TfidfVectorizer(strip_accents=None,
                        lowercase=False,
                        preprocessor=None)

param_grid = [{'vect__ngram_range': [(1, 1)],
               'vect__stop_words': [stopwords.words('english'), None],
               'vect__tokenizer': [tokenizer, tokenizer_porter],
               'clf__penalty': ['l1', 'l2'],
               'clf__C': [1.0, 10.0, 100.0]},
              {'vect__ngram_range': [(1, 1)],
               'vect__stop_words': [stopwords.words('english'), None],
               'vect__tokenizer': [tokenizer, tokenizer_porter],
               'vect__use_idf':[False],
               'vect__norm':[None],
               'clf__penalty': ['l1', 'l2'],
               'clf__C': [1.0, 10.0, 100.0]},
              ]

lr_tfidf = Pipeline([('vect', tfidf,),
                     ('clf', LogisticRegression(random_state=0,
                                                solver='liblinear'))])

gs_lr_tfdif = GridSearchCV(lr_tfidf, param_grid,
                           scoring='accuracy',
                           cv=5, verbose=2,
                           n_jobs=2)

gs_lr_tfdif.fit(X_train, y_train)
print('Best parameter set %s' % gs_lr_tfdif.best_params_)
print('CV accuracy %.3f' % gs_lr_tfdif.best_score_)

clf = gs_lr_tfdif.best_estimator_
print('Test accuracy: %.3f' % clf.score(X_test, y_test))
