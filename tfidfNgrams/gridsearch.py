from preprocess import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
import joblib
import numpy as np
import pandas as pd

def train_test_split_data(df):
    X = df.drop(['is_asshole', 'id'], axis=1)
    y = df.is_asshole.values
    return train_test_split(X, y, test_size=0.2, random_state=448)

def train_on_df(df):
    X_train, X_test, y_train, y_test = train_test_split_data(df)
    params = {'C': [0.01, 0.1, 1, 10, 100], 'l1_ratio': [0, 0.2, 0.4, 0.6, 0.8, 1]}
    logistic_classifier = GridSearchCV(
        LogisticRegression(penalty = 'elasticnet', class_weight = 'balanced', solver = 'saga'), 
        params, 
        scoring='balanced_accuracy',
        n_jobs = 4, 
        cv = 5, 
        verbose = 2
    )
    logistic_classifier.fit(X_train, y_train)
    y_pred = logistic_classifier.predict(X_test)
    print("Had balanced accuracy " + str(balanced_accuracy_score(y_test, y_pred)))
    return logistic_classifier

if __name__ == "__main__":
    np.random.seed(448)

    print("Reading base...")
    base_df = pd.read_csv('aita_preprocessed.csv', sep='\t')
    X = base_df.processed_body.values

    # feel free to comment this out if using Albert's csvs
    # if so, use uni_df = pd.read_csv(<unigram csv>, sep='\t') (assuming tab sep)
    # Start block
    print("Generating Ngrams...")
    unigram_vectorizer = TfidfVectorizer(ngram_range=(1,1), max_df=0.75, min_df=2)
    bigram_vectorizer = TfidfVectorizer(ngram_range=(1,2), max_df=0.75, min_df=2)
    trigram_vectorizer = TfidfVectorizer(ngram_range=(1,3), max_df=0.75, min_df=2)
    X_uni = unigram_vectorizer.fit_transform(X)
    X_bi = bigram_vectorizer.fit_transform(X)
    X_tri = trigram_vectorizer.fit_transform(X)

    uni_df = pd.DataFrame.sparse.from_spmatrix(X_uni)
    bi_df = pd.DataFrame.sparse.from_spmatrix(X_bi)
    tri_df = pd.DataFrame.sparse.from_spmatrix(X_tri)
    # End block

    print("Reading doc2vec...")
    doc2vec_df = pd.read_csv('aita_doc2vec.csv', sep='\t')

    # TODO: If we want to add all features, we can also manually add them here without joins
    # just like how id and is_asshole are currently added
    print("Extending Dataframes...")
    uni_df['id'], uni_df['is_asshole'] = base_df['id'], base_df['is_asshole']
    bi_df['id'], bi_df['is_asshole'] = base_df['id'], base_df['is_asshole']
    tri_df['id'], tri_df['is_asshole'] = base_df['id'], base_df['is_asshole']
    doc2vec_df['id'], doc2vec_df['is_asshole'] = base_df['id'], base_df['is_asshole']
    doc2vec_df = doc2vec_df.drop("Unnamed: 0", axis=1)

    print("Beginning grid search...")
    for curr_df, df_name in [(doc2vec_df, 'D2V'), (uni_df, "UNI"), (bi_df, "BI"), (tri_df, "TRI")]:
        print('Assessing ' + df_name)
        curr_clf = train_on_df(curr_df)
        print('Had optimal parameters' + str(curr_clf.best_params_))
        print('\n')
        joblib.dump(curr_clf.best_estimator_, df_name + '.pkl')

    pass
