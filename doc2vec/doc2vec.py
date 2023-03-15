from gensim.models.doc2vec import Doc2Vec
from nltk import word_tokenize
from sklearn.model_selection import train_test_split
import pandas as pd
from preprocess import *
from sklearn.model_selection import train_test_split
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from tqdm import tqdm

def train_model(data_training, vec_size=300):
    model = Doc2Vec(
        vector_size=vec_size, 
        workers=-1,
        min_count=3, 
        epochs=30, 
        seed=448,
        hs=1,
        negative=0
    )
    model.build_vocab(data_training)
    model.train(data_training, total_examples=model.corpus_count, epochs=model.epochs)
    
    return model
    
def infer_all_vecs(model, X):
    return [model.infer_vector(word_tokenize(post)) for post in X.processed_body.values]

def construct_d2v_df(model, vecs, X, vec_size=300):
    doc2vec_df = pd.DataFrame(vecs, columns = ['Dim ' + str(i) for i in range(vec_size)])
    doc2vec_df['id'] = X['id']
    
    return doc2vec_df, model.wv

def X_to_tagged_set(X):
    def to_tagged_doc(post): return TaggedDocument(word_tokenize(post['processed_body']), post['id'])
    return X.apply(to_tagged_doc, axis=1)

def get_data():
    base_df = pd.read_csv('aita_preprocessed_new.csv', sep='\t')
    X = base_df[['processed_body', 'id']]
    y = base_df.is_asshole.values
    X_train, X_test, _, _ = train_test_split(X, y, test_size=0.2, random_state=448, stratify=y)
    data_training = X_to_tagged_set(X_train)
    data_testing =  X_to_tagged_set(X_test)
    
    return X, data_training, data_testing

def construct_d2v_of_size_v(X, data_training, data_testing, v=300):
    print("Training a doc2vec model with vectors of size v = " + str(v) + "...")
    model = train_model(data_training, vec_size=150)
    vecs = infer_all_vecs(model, X)
    
    print("Saving results...")
    doc2vec_df, word_to_embedding = construct_d2v_df(model, vecs, X)
    doc2vec_df.to_csv('aita_doc2vec_' + str(v) + '.csv', sep='\t', encoding='utf-8')
    model.save('aita_doc2vec_' + + str(v) + '_model')
    
    training_score = model.score(data_testing, total_sentences=len(data_testing))
    print("Doc2Vec model with v = " + str(v) + " had a log prob of " + str(training_score))
    print('\n')
    
if __name__ == "__main__":
    np.random.seed(448)
    X, data_training, data_testing = get_data()
    for v in tqdm([100, 150, 200, 300, 500, 750, 1000]):
        construct_d2v_of_size_v(X, data_training, data_testing, v)
        
    pass
    
    
    
    