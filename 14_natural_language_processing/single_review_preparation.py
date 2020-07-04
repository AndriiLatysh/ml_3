import pandas as pd
import numpy as np
import re
import string
import nltk
import nltk.corpus as nltk_corpus


imdb_reviews = pd.read_csv("data/IMDB Dataset.csv")

X = imdb_reviews["review"].iloc[3]
print(X, "\n")

X = re.sub("<.*?>", " ", X)
X = X.lower()
X = X.translate(str.maketrans("", "", string.punctuation))

lemmatizer = nltk.stem.WordNetLemmatizer()
# stemmer = nltk.stem.LancasterStemmer()
# stemmer = nltk.stem.PorterStemmer()

stop_words = nltk_corpus.stopwords.words("english")
# print(stop_words)

X = nltk.word_tokenize(X)

# X = [word for word in X if word not in stop_words]

X = [lemmatizer.lemmatize(word) for word in X]
# X = [stemmer.stem(word) for word in X]

X = " ".join(X)

print(X, "\n")
