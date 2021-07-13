import pandas as pd
from timeit import default_timer as timer
import numpy as np
from sklearn.decomposition import PCA
from matplotlib import pyplot
from keras.preprocessing.text import text_to_word_sequence
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import Word2Vec
from gensim.models import KeyedVectors

df_rat = pd.read_csv("ratings.csv")
df_mov = pd.read_csv("movies.csv")

# define the document
text = 'Toy Story (1995)'


# tokenize the document
# result = text_to_word_sequence(text)
# print(result)

def process_titles():
    titles = df_mov['title']

    new_titles_list = []
    for title in titles:
        res = text_to_word_sequence(title)
        new_titles_list.append(res)
        # print(res)

    print(new_titles_list)


# sw = set(stopwords.words('english'))
# print(sw)
sw = {'to', 'the', 'a'}


# ps = PorterStemmer()


def clean_title(title):
    result = text_to_word_sequence(title)

    # result = result.split()
    # result = [w for w in result if w not in sw]
    return result


# print(df_mov['title'].apply(clean_title).head(10))
# df_mov['cleaned_title'] = df_mov['title'].apply(clean_title)
title_list = df_mov['title'].apply(clean_title).tolist()


# print(title_list)
def train_word2vec():
    model = Word2Vec(title_list, min_count=1, window=5, size=100, sg=1)
    # summarize vocabulary
    words = list(model.wv.vocab)
    print(len(words))
    # access vector for one word
    # word_embedding = model.wv.__getitem__('story')

    # Store just the words + their trained embeddings.
    word_vectors = model.wv
    word_vectors.save("word2vec.word_vectors")


#train_word2vec()
# Load back with memory-mapping = read-only, shared across processes.
wv = KeyedVectors.load("word2vec.word_vectors", mmap='r')
#vector = wv['toy']  # Get numpy vector of a word
#print(vector)
res = wv.most_similar('toy')
print(res)
# print(word_embedding)
sim = wv.similarity('toy', 'story')
print(sim)

