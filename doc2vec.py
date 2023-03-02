import Gensim
from nltk import word_tokenize


def base_df_to_doc2vec_df():
    data = [post for post in df.body] 
    vec_size = 40

    def tagged_document(list_of_list_of_words):
      for i, list_of_words in enumerate(list_of_list_of_words):
          yield Gensim.models.doc2vec.TaggedDocument(list_of_words, [i])
    data_training = list(tagged_document(data))

    model = Gensim.models.doc2vec.Doc2Vec(vector_size=vec_size, min_count=2, epochs=30)
    model.build_vocab(data)
    vecs = [model.infer_vector[word_tokenize(post)] for post in df.body]

    doc2vec_df = pd.DataFrame(doc_lda, columns = ['Dim ' + str(i) for i in range(vec_size)])
    doc2vec_df.to_csv('aita_doc2vec.csv', sep='\t', encoding='utf-8')
    
    return doc2vec_df