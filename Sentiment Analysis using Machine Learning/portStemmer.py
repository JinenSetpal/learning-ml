from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords


def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]


porter = PorterStemmer()
stop = stopwords.words('english')
print([w for w in tokenizer_porter('runners like running and thus they run')[-10:] if w not in stop])
