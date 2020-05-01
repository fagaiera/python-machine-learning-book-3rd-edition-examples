from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import pandas as pd
import nltk
import re

# Cleaning text data
df = pd.read_csv('movie_data.csv', encoding='utf-8')

print(df.loc[0, 'review'][-50:])

def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',
                           text)
    text = (re.sub('[\W]+', ' ', text.lower()) +
            ' '.join(emoticons).replace('-', ''))
    return text

preprocessor(df.loc[0, 'review'][-50:])
preprocessor("</a>This :) is :( a test :-)!")


df['review'] = df['review'].apply(preprocessor)

# ## Processing documents into tokens

porter = PorterStemmer()

def tokenizer(text):
    return text.split()


def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]


print(tokenizer('runners like running and thus they run'))

print(tokenizer_porter('runners like running and thus they run'))

nltk.download('stopwords')

stop = stopwords.words('english')
[w for w in tokenizer_porter('a runner likes running and runs a lot')[-10:]
if w not in stop]