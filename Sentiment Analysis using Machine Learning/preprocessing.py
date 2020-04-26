import re

import pandas as pd


def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',
                           text)
    text = (re.sub('[\W]+', ' ', text.lower()) +
            ' '.join(emoticons).replace('-', ''))
    return text


df = pd.read_csv('movie_data.csv', header=None)
df.columns = ['review', 'sentiment']

print(preprocessor(df.loc[0, 'review'][-50:]))
