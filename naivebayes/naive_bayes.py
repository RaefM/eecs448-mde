# EECS 487 Intro to NLP
# Assignment 1

import pandas as pd
import scipy.sparse
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np


def load_posts(filename):
    df = None

    ###################################################################
    # TODO: load data into pandas dataframe
    ###################################################################
    df = pd.read_csv(filename, usecols=[ "post", "is_asshole"])
    df.rename(columns={"post": "text", "is_asshole": "label"}, inplace=True)
    ###################################################################

    return df


def get_basic_stats(df):
    avg_len = 0
    std_len = 0
    num_articles = {0: 0, 1: 0}

    ###################################################################
    # TODO: calculate mean and std of the number of tokens in the data
    ###################################################################
    num_tokens = []
    for i, row in df.iterrows():
        num_tokens.append(len(word_tokenize(row['text'])))
        num_articles[row['label']] += 1
        
    avg_len = np.mean(num_tokens)
    std_len = np.std(num_tokens)
    ###################################################################
    
    (f"Average number of tokens per headline: {avg_len}")
    print(f"Standard deviation: {std_len}")
    print(f"Number of legitimate/clickbait headlines: {num_articles}")


class NaiveBayes:
    """Naive Bayes classifier."""

    def __init__(self):
        self.ngram_count = []
        self.total_count = []
        self.category_prob = []
    
    def fit(self, data):

        ###################################################################
        # TODO: store ngram counts for each category in self.ngram_count
        ###################################################################
        # convert to lowercase
        df = data.copy()
        num_labels = len(df['label'].unique())
        df['text'] = df['text'].apply(lambda x: x.lower()) 
        
        # use count vectorizer with max_df = 0.8, min_df = 3, using uni and bigrams
        self.vectorizer = CountVectorizer(max_df = 0.8, min_df = 3, ngram_range = (1, 2))
                
        # fit the vectorizer to the entire training corpus
        self.vectorizer.fit(df['text'])
        
        # vocab size
        V = len(self.vectorizer.get_feature_names_out())
        
        # initialize counts and probs
        for i in range(num_labels):
            self.ngram_count.append(np.zeros(V))
            self.total_count.append(0)
            self.category_prob.append(0)
        
        # transform all reviews into count arrays
        df = pd.concat([df, pd.DataFrame.sparse.from_spmatrix(self.vectorizer.transform(df['text']))], axis=1)
        
        def update_prob(row):
             # update counts of each token for this row's class with the counts it introduced
            self.ngram_count[row['label']] = (
                np.asarray([sum(tup) for tup in zip(self.ngram_count[row['label']], row['counts'])])
             )
            # update the total number of tokens per category by adding all word counts
            self.total_count[row['label']] += sum(row['counts'])
            # update the number of headlines per category
            self.category_prob[row['label']] += 1
        
        # sum of the transformed values for each row, adding to self.total_count[0] if 0 and 1 otherwise
        df.apply(update_prob)
                    
        # convert the number of headlines per category to the prob by dividing by the number of 
        num_headlines = sum(self.category_prob)
        for i in range(len(self.category_prob)):
            self.category_prob[i] /= num_headlines
        ###################################################################
    
    def calculate_prob(self, docs, c_i):
        prob = None

        ###################################################################
        # TODO: calculate probability of category c_i given each headline in docs
        ###################################################################
        probs = []
        V = len(self.vectorizer.get_feature_names_out())
        transformed_docs = self.vectorizer.transform(docs)
        
        for doc in transformed_docs.toarray():
            sum_of_log_probs = np.log(self.category_prob[c_i])
            
            for j, freq in enumerate(doc):
                if freq > 0:
                    # prob is (count(this word, c_i) + 1) / (count(c_i) + V)
                    prob = (self.ngram_count[c_i][j] + 1) / (self.total_count[c_i] + V)
                    # log(p(x1|ci)* p(x2|ci) * p(x2|ci)) = log(p(x1|ci)) + log(p(x2|ci)) + log(p(x2|xi))
                    sum_of_log_probs += np.log(prob) * freq
            
            probs.append(sum_of_log_probs)
            
        prob = probs
        ###################################################################

        return prob

    def predict(self, docs):
        prediction = [None] * len(docs)

        ###################################################################
        # TODO: predict categories for the headlines
        ###################################################################
        probs = []
        for c_i in range(len(self.category_prob)):
            probs.append(self.calculate_prob(docs, c_i))
            
        prediction = np.argmax(np.asarray(probs), axis=0)
        ###################################################################

        return prediction


def evaluate(predictions, labels):
    accuracy, mac_f1, mic_f1 = None, None, None

    ###################################################################
    # TODO: calculate accuracy, macro f1, micro f1
    # Note: you can assume labels contain all values from 0 to C - 1, where
    # C is the number of categories
    ###################################################################
    num_labels = len(set(labels))
    num_headlines = len(predictions)
    conf_mat = np.zeros(shape=(num_labels, num_labels))
    
    def get_tp(cm, c_i):
        return cm[c_i][c_i]
    
    def get_fp(cm, c_i):
        return sum(cm.T[c_i]) - cm[c_i][c_i]
        
    def get_fn(cm, c_i):
        return sum(cm[c_i]) - cm[c_i][c_i]
    
    def get_f1(cm, c_i):
        tp = get_tp(cm, c_i)
        fp = get_fp(cm, c_i)
        fn = get_fn(cm, c_i)
        
        P = tp / (tp + fp)
        R = tp / (tp + fn)
        
        return 2*P*R / (P + R)
    
    for (j, p) in zip(labels, predictions):
        conf_mat[j][p] += 1
        
    sum_of_all_f1 = 0
    sum_of_all_tp = 0
    sum_of_all_fp = 0
    sum_of_all_fn = 0
    
    for i in range(num_labels):
        sum_of_all_f1 += get_f1(conf_mat, i)
        sum_of_all_tp += get_tp(conf_mat, i)
        sum_of_all_fp += get_fp(conf_mat, i)
        sum_of_all_fn += get_fn(conf_mat, i)
        
    # f1 = 2*TP / (2TP + FP + FN)
    mic_f1 = 2*sum_of_all_tp / (2*sum_of_all_tp + sum_of_all_fp + sum_of_all_fn)
    mac_f1 = sum_of_all_f1 / num_labels
        
    accuracy = 0
    for i in range(num_labels):
        accuracy += conf_mat[i][i]
        
    accuracy /= num_headlines
    ###################################################################

    return accuracy, mac_f1, mic_f1
