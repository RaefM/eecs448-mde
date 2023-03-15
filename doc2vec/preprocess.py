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
import string
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper
from nltk.corpus import wordnet

np.random.seed(448)

fname = "aita_clean.csv"

def get_wordnet_pos(treebank_tag):
        """
        Source: Deepak on stack overflow 
        https://stackoverflow.com/questions/15586721/wordnet-lemmatization-and-pos-tagging-in-python
        return WORDNET POS compliance to WORDNET lemmatization (a,n,r,v) 
        """
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            # As default pos in lemmatization is Noun
            return wordnet.NOUN

def get_basic_stemmed_and_lemmatized_df():
    df = pd.read_csv(fname)
    df["body"] = df["title"].astype(str) + df["body"].astype(str)
    stem_and_lemmatize(df)
    
    df["score"] = np.log(df["score"])
    df["num_comments"] = np.log(df["num_comments"])
    df["num_words"] = np.log(df["num_words"])
    
    mapper = DataFrameMapper([(["score", "num_comments", "num_words"], StandardScaler())])
#     scaled_features = mapper.fit_transform(df.copy())
#     scaled_features_df = pd.DataFrame(scaled_features, index=df.index, columns=df.columns)
    df[["score", "num_comments", "num_words"]] = mapper.fit_transform(df.copy())
    
    return df

def get_data(ngram_start=1, ngram_end=1):
    df = pd.read_csv(fname)
    df["body"] = df["title"].astype(str) + df["body"].astype(str)
    stem_and_lemmatize(df)  

    train_data, test_data = train_test_split(df, test_size=0.33, random_state=448)

    X, vectorizer = tf_idf_bag_of_words(df, ngram_start, ngram_end)  
    return df

def preprocess_string(inp_string, stopWords):
    def exclude_word(word):
        return word in stopWords or any(char.isdigit() for char in word)
    def remove_punc(word):
        return word.translate(str.maketrans('', '', string.punctuation))
    # expand any contractions
    expanded_string = ' '.join([contractions.fix(word) for word in inp_string.split()])
    # for words without numbers that aren't stopwords, remove punctuation and return
    return [remove_punc(word) for word in word_tokenize(expanded_string) if not exclude_word(word)]

def get_stopwords():
    pronouns = []
    with open("pronouns.txt", 'r') as f:
        pronouns = [x.strip() for x in f.readlines()]

    stopWords = set(filter(lambda x : x not in pronouns, stopwords.words('english')))
    return stopWords

def stem_and_lemmatize(df):
    lemmatizer = WordNetLemmatizer()
    stopWords = get_stopwords()
    def lemmatize(x):
        tokens = preprocess_string(x.lower(), stopWords)
        tokens_with_pos = nltk.pos_tag(tokens)
        return [lemmatizer.lemmatize(word, get_wordnet_pos(pos)) for (word, pos) in tokens_with_pos]
    
#     df["processed_body_split"] = df["body"].map(lambda x : [lemmatizer.lemmatize(word) for word in preprocess_string(x.lower(), stopWords)])
    df["processed_body_split"] = df["body"].map(lemmatize)
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
