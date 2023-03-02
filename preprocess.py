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

def expand_contractions(string):
    expanded_words = []   
    for word in string.split():
      expanded_words.append(contractions.fix(word))  

    expanded_text = ' '.join(expanded_words)
    return expanded_text

def preprocess_string(string):
  # removes everything in brackets '[...]' = ''
  # string = re.sub("[\[].*?[\]]", "", string)
  return word_tokenize(expand_contractions(string))

def get_stopwords():
    pronouns = []
    with open("pronouns.txt", 'r') as f:
        pronouns = [x.strip() for x in f.readlines()]

    stopwords = set(filter(lambda x : x not in pronouns, stopwords.words('english'))).add('AITA')
    return stopwords

def stem_and_lemmatize(df):
    lemmatizer = WordNetLemmatizer()
    df["lemmatized_body"] = df["body"].map(lambda x : [lemmatizer.lemmatize(word) for word in preprocess_string(x)])

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
