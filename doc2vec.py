from gensim.models.doc2vec import Doc2Vec
from nltk import word_tokenize
from sklearn.model_selection import train_test_split
import pandas as pd

def train_model(data_training, vec_size=300, num_threads=6):
    model = Doc2Vec(
        vector_size=vec_size, 
        workers=num_threads,
        min_count=2, 
        epochs=30, 
        seed=448
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