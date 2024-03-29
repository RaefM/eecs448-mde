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
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import StandardScaler

def train_test_split_data(df):
    X = df.drop(['is_asshole', 'id'], axis=1)
    y = df.is_asshole.values
    return train_test_split(X, y, test_size=0.2, random_state=448, stratify=y)

def train_on_df(df):
    X_train, X_test, y_train, y_test = train_test_split_data(df)
    params = {'C': [0.01, 0.1, 1, 10, 100], 'l1_ratio': [0, 0.2, 0.4, 0.6, 0.8, 1]}
    logistic_classifier = GridSearchCV(
        LogisticRegression(penalty = 'elasticnet', class_weight = 'balanced', solver = 'saga', tol=1e-4, max_iter=600, warm_start=True), 
        params, 
        scoring='balanced_accuracy',
        n_jobs = -1, 
        cv = 5, 
        verbose = 3
    )
    logistic_classifier.fit(X_train, y_train)
    y_pred = logistic_classifier.predict(X_test)
    print("Had balanced accuracy " + str(balanced_accuracy_score(y_test, y_pred)))
    return logistic_classifier

if __name__ == "__main__":
    np.random.seed(448)

    print("Reading base datasets...")
    base_df = pd.read_csv('aita_preprocessed_new.csv', sep='\t')
    liwc_df = pd.read_csv('liwc_with_id.csv')
    emotion_df = pd.read_csv('aita_emotion_final.csv')
    topic_df = pd.read_csv('aita_topics_5.csv', sep='\t')
    liwc_df = liwc_df.drop("Unnamed: 0", axis=1)
    liwc_df = liwc_df.rename(columns={'V1': 'id'})
    mapper = DataFrameMapper([(liwc_df.columns[1:], StandardScaler())])
    liwc_df[liwc_df.columns[1:]] = mapper.fit_transform(liwc_df.copy())

    X = base_df.processed_body.values

    print("Reading doc2vec...")
    doc2vec_df = pd.read_csv('aita_doc2vec_150.csv', sep='\t')
    doc2vec_df = doc2vec_df.drop("Unnamed: 0", axis=1)
    doc2vec_df = doc2vec_df.merge(
        base_df[['id', 'is_asshole']], 
        on='id'
    )

    print("Merging datasets...")
    doc2vec_df_with_liwc = doc2vec_df.copy()
    doc2vec_df_with_emotion = doc2vec_df.copy()
    doc2vec_df_with_topic = doc2vec_df.copy()

    doc2vec_df_with_emotion = doc2vec_df_with_emotion.merge(
        emotion_df[['id','Anger','Joy','Optimism','Sadness']], 
        on='id'
    )
    doc2vec_df_with_topic = doc2vec_df_with_topic.merge(
        topic_df[['id', 'TopicProb0', 'TopicProb1', 'TopicProb2', 'TopicProb3', 'TopicProb4']], 
        on='id'
    )
    doc2vec_df_with_liwc = doc2vec_df_with_liwc.merge(
        liwc_df, 
        on='id'
    )

    print("Beginning grid search of LR models...")
    for curr_df, df_name in [(doc2vec_df_with_liwc, 'D2VMix'), (doc2vec_df_with_emotion, 'D2VEmo'), (doc2vec_df_with_topic, 'D2VTopic5')]:
        print('Assessing ' + df_name)
        curr_clf = train_on_df(curr_df)
        print('Had optimal parameters' + str(curr_clf.best_params_))
        print('\n')
        joblib.dump(curr_clf.best_estimator_, df_name + '.pkl')
        
    pass
