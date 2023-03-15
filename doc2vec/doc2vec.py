from gensim.models.doc2vec import Doc2Vec
from nltk import word_tokenize
from sklearn.model_selection import train_test_split
import pandas as pd
from preprocess import *
from sklearn.model_selection import train_test_split
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from tqdm import tqdm
import numpy as np

def train_model(data_training, vec_size=300):
    model = Doc2Vec(
        vector_size=vec_size, 
        workers=20,
        min_count=3,
        seed=448,
        hs=1,
        negative=0
    )
    model.build_vocab(data_training)
    model.train(data_training, total_examples=model.corpus_count, epochs=model.epochs)
    
    return model
    
def infer_all_vecs(model, X_train, X_test, X_val, vec_size):
    d2v_df_train = X_train.apply(lambda r: [r['id']] + list(model.dv[r['id']]), axis=1, result_type='expand')
    d2v_df_val = X_val.apply(
        lambda r: [r['id']] + list(model.infer_vector(word_tokenize(r['processed_body']))), 
        axis=1, 
        result_type='expand'
    )
    d2v_df_test = X_test.apply(
        lambda r: [r['id']] + list(model.infer_vector(word_tokenize(r['processed_body']))), 
        axis=1, 
        result_type='expand'
    )
    d2v_df_train.columns = ['id'] + ['Dim ' + str(i) for i in range(vec_size)]
    d2v_df_val.columns = ['id'] + ['Dim ' + str(i) for i in range(vec_size)]
    d2v_df_test.columns = ['id'] + ['Dim ' + str(i) for i in range(vec_size)]

    return pd.concat([pd.concat([d2v_df_train, d2v_df_val], join='outer'), d2v_df_test])

def X_to_tagged_set(X):
    def to_tagged_doc(post):
        return TaggedDocument(word_tokenize(post['processed_body']), [post['id']])
    return X.apply(to_tagged_doc, axis=1).to_numpy()

def get_split_data(inp_file='aita_preprocessed_new.csv'):
    base_df = pd.read_csv(inp_file, sep='\t')
    X = base_df[['processed_body', 'id']]
    y = base_df.is_asshole.values
    X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=448, stratify=y)
    X_true_train, X_val, _, _ = train_test_split(X_train, y_train, test_size=0.1, random_state=448, stratify=y_train)
    data_training = X_to_tagged_set(X_true_train)
    data_val =  X.processed_body.values
    
    return X, data_training, data_val, X_true_train, X_test, X_val

def construct_d2v_of_size_v(X, X_train, X_test, X_val, data_training, data_val, v):
    print("Training a doc2vec model with vectors of size v = " + str(v) + "...")
    model = train_model(data_training, vec_size=v)

    print("Inferring vectors for all data and making df...")
    doc2vec_df = infer_all_vecs(model, X_train, X_test, X_val, v)
    
    print("Saving model and csv...")
    doc2vec_df.to_csv('aita_doc2vec_' + str(v) + '.csv', sep='\t', encoding='utf-8')
    model.save('aita_doc2vec_' + str(v) + '_model')
    
    val_score = np.mean(model.score(data_val, total_sentences=len(data_val)))
    print("Doc2Vec model with v = " + str(v) + " had a log prob of " + str(val_score) + " on the validation set")
    print('\n')
    
if __name__ == "__main__":
    np.random.seed(448)
    X, data_training, data_val, X_train, X_test, X_val = get_split_data()
    for v in tqdm([100, 150, 200, 300, 500, 750, 1000]):
        construct_d2v_of_size_v(X, X_train, X_test, X_val, data_training, data_val, v)
        
    pass
    
    
    
    
