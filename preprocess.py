import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from IPython.display import display
import nltk.stem
import pandas as pd
import contractions
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import re
import numpy as np

np.random.seed(448)

fname = "aita_clean.csv"

def get_basic_stemmed_and_lemmatized_df(ngram_start=1, ngram_end=1):
    df = pd.read_csv(fname)
    df["body"] = df["title"].astype(str) + df["body"].astype(str)
    stem_and_lemmatize(df)

    return df

def get_data(ngram_start=1, ngram_end=1):
    df = pd.read_csv(fname)
    df["body"] = df["title"].astype(str) + df["body"].astype(str)
    stem_and_lemmatize(df)  

    train_data, test_data = train_test_split(df, test_size=0.33, random_state=448)

    X, vectorizer = tf_idf_bag_of_words(df, ngram_start, ngram_end)  
    return df

def preprocess_string(string, stopWords):
  return [contractions.fix(word) for word in word_tokenize(string) if word not in stopWords]

def get_stopwords():
    pronouns = []
    with open("pronouns.txt", 'r') as f:
        pronouns = [x.strip() for x in f.readlines()]

    stopWords = set(filter(lambda x : x not in pronouns, stopwords.words('english')))
    return stopWords

def stem_and_lemmatize(df):
    lemmatizer = WordNetLemmatizer()
    stopWords = get_stopwords()
    df["processed_body_split"] = df["body"].map(lambda x : [lemmatizer.lemmatize(word) for word in preprocess_string(x.lower(), stopWords)])
    df["processed_body"] = df["processed_body_split"].map(lambda x : ' '.join(x))
    df["num_words"] = df["processed_body_split"].map(lambda x : len(x))

def tf_idf_bag_of_words(input_corpus, ngram_start=1, ngram_end=1):
    vectorizer = TfidfVectorizer(
      lowercase = True, 
      analyzer = "word", 
      stop_words = get_stopwords(), 
      ngram_range = (ngram_start, ngram_end), 
      norm = "l2",
      min_df=2,
      max_df=0.8
    )
    X = vectorizer.fit_transform(input_corpus)
    return X, vectorizer

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = get_data()
