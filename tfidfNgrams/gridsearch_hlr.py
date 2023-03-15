from preprocess import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
import joblib
import numpy as np
import pandas as pd
from math import isclose

def train_test_split_data(df):
    X = df.drop(['is_asshole', 'id'], axis=1)
    y = df.is_asshole.values
    return train_test_split(X, y, test_size=0.2, random_state=448, stratify=y)

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

def augment_with_all_data(base_df):
    base_with_all_df = base_df.copy()

    # run multiple joins here!

def augment_with_topic_data(base_df):
    # Replace with source of states if changing from topic
    topics_df = pd.read_csv('aita_topics_5.csv', sep='\t').drop(['Unnamed: 0', 'id'], axis=1)
    topics_arr = topics_df.to_numpy()
    n_topics = 5
    topic_names = ['T0', 'T1', 'T2', 'T3', 'T4']
    # end block

    for row in topics_arr:
        num_zero = n_topics - np.count_nonzero(row)
        row_sum = np.sum(row)
        if not isclose(1.0, row_sum, rel_tol=1e-05): 
            row[row == 0] += (1 - row_sum) / num_zero

    topics_df_normalized = pd.DataFrame(topics_arr, columns=topic_names)
    base_with_topic_df = base_df.copy()
    # Step 2: append information to base dataset of choice
    base_with_topic_df['T0'], base_with_topic_df['T1'], base_with_topic_df['T2'], base_with_topic_df['T3'], base_with_topic_df['T4'] = (
        # swap with relevant normalized columns if changing from topic
        topics_df_normalized['T0'],
        topics_df_normalized['T1'],
        topics_df_normalized['T2'],
        topics_df_normalized['T3'],
        topics_df_normalized['T4'],
        # end block
    )
    return base_with_topic_df, topic_names

def multi_state_clf_performance(clfs, X_test, X_test_weights, y_test, state_names):
    # for each test set
    probs_under_each_state = np.zeros(len(X_test))
    for clf, state in zip(clfs, state_names):
        log_probs = clf.predict_log_proba(X_test)
        log_weights = np.log(X_test_weights[state].to_numpy())
        # log(p(X=1|t0)p(t0)) = log(p(X=1|t0)) + log(p(t0))
        probs_under_each_state += (np.exp((log_probs.T[1]).T + log_weights))
        
    print(probs_under_each_state)
    y_pred = [1 if prob > 0.5 else 0 for prob in probs_under_each_state]
    
    print("Had balanced accuracy: " + str(balanced_accuracy_score(y_test, y_pred)))

def train_on_multi_state_data(data_with_fuzz, state_names):
    params = {'C': [0.01, 0.1, 1, 10, 100], 'l1_ratio': [0, 0.2, 0.4, 0.6, 0.8, 1]}
    per_class_clfs = []
    X_train_all, X_test_all, y_train, y_test = train_test_split_data(data_with_fuzz)
    
    X_train = X_train_all.drop(state_names, axis=1)
    X_train_weights = X_train_all[state_names]
    X_test = X_test_all.drop(state_names, axis=1)
    X_test_weights = X_test_all[state_names]
    
    # train a classifier for each class, fitting fuzzily to each point based on the
    # probability that it came from the corresponding cluster
    for class_state in state_names:
        logistic_classifier = GridSearchCV(
            LogisticRegression(penalty = 'elasticnet', class_weight = 'balanced', solver = 'saga'), 
            params,
            n_jobs = 4, 
            cv = 5, 
            verbose = 3,
        )
        logistic_classifier.fit(X_train, y_train, **{'sample_weight': X_train_weights[class_state]})
        per_class_clfs.append(logistic_classifier)
        
    multi_state_clf_performance(per_class_clfs, X_test, X_test_weights, y_test, state_names)
        
    return per_class_clfs

if __name__ == "__main__":
    np.random.seed(448)

    print("Reading base...")
    base_df = pd.read_csv('aita_preprocessed_new.csv', sep='\t')
    X = base_df.processed_body.values

    print("Generating Ngrams...")
    unigram_vectorizer = TfidfVectorizer(ngram_range=(1,1), max_df=0.75, min_df=3)
    X_uni = unigram_vectorizer.fit_transform(X)
    uni_df = pd.DataFrame.sparse.from_spmatrix(X_uni)

    print("Reading doc2vec...")
    doc2vec_df = pd.read_csv('aita_doc2vec_300.csv', sep='\t')
    doc2vec_df = doc2vec_df.drop("Unnamed: 0", axis=1)

    print("Extending Dataframes...")
    uni_df['id'], uni_df['is_asshole'] = base_df['id'], base_df['is_asshole']
    doc2vec_df = doc2vec_df.merge(base_df[['id', 'is_asshole']], on='id')
    
    print("Generating topic datasets...")
    topic_uni_df, topic_names = augment_with_topic_data(uni_df)
    topic_d2v_df, _ = augment_with_topic_data(doc2vec_df)
        
    print("\nBeginning grid search of Hierarchical LR models...")
    for curr_df, states, df_name in [(topic_uni_df, topic_names, "HLR_topic_uni_"), (topic_d2v_df, topic_names, "HLR_topic_d2v300_")]:
        print('Assessing ' + df_name)
        per_class_clfs = train_on_multi_state_data(curr_df, states)
        print("Per state clf optimal parameters:")
        for curr_clf, state in zip(per_class_clfs, states):
            print(str(state) + ' had optimal parameters ' + str(curr_clf.best_params_))
            joblib.dump(curr_clf.best_estimator_, df_name + str(state) + '.pkl')
        print('\n')
        
    pass
