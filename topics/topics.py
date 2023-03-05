import re
import numpy as np
import pandas as  pd
from pprint import pprint# Gensim
import gensim
import gensim.corpora as corpora
from nltk.tokenize import word_tokenize
from gensim.models import CoherenceModel# spaCy for preprocessing
import spacy# Plotting tools
from nltk.corpus import stopwords

from nltk.corpus import stopwords

def process_sentence(sentence):
    stopwords_eng = stopwords.words('english')
    punc = [*'!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~']
    other_common_words = ['wibta', 'aita', '``']
    def not_in_stop(word):
        return word not in stopwords_eng and word not in punc and word not in other_common_words
    return [word for word in word_tokenize(sentence) if not_in_stop(word)]

def get_corpus(df):
    # Convert to list 
    stopwords_eng = stopwords.words('english')
    punc = [*'!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~']
    other_common_words = ['wibta', 'aita', '``']
    def not_in_stop(word):
        return word not in stopwords_eng and word not in punc and word not in other_common_words
    def tokenized_without_stopwords(sentence): 
        return [word for word in word_tokenize(sentence) if not_in_stop(word)]
    data_words = [tokenized_without_stopwords(sentence) for sentence in df.processed_body.values]
    print(data_words[:1])

    # Create Dictionary 
    id2word = corpora.Dictionary(data_words)  
    # Create Corpus 
    texts = data_words  
    # Term Document Frequency 
    corpus = [id2word.doc2bow(text) for text in texts]  
    # View 
    print(corpus[:1])
    
    return id2word, corpus

def train_lda_model(df, id2word, corpus, n_topics=20):
    lda_model = gensim.models.ldamulticore.LdaMulticore(corpus=corpus,
                                                    id2word=id2word,
                                                    workers=4,
                                                    num_topics=n_topics, 
                                                    random_state=100,
                                                    chunksize=100,
                                                    passes=5,
                                                    alpha='asymmetric',
                                                    per_word_topics=True)

    doc_lda = lda_model[corpus] 
    return lda_model, doc_lda

def get_scores(lda_model, id2word, corpus):
    # Compute Perplexity
    print('\nPerplexity: ', lda_model.log_perplexity(corpus))  
    # a measure of how good the model is. lower the better.

    # Compute Coherence Score
    coherence_model_lda = CoherenceModel(model=lda_model, texts=corpus, dictionary=id2word, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print('\nCoherence Score: ', coherence_lda)
    pass

def create_topic_df(df, doc_lda, n_topics=20):
    probs = np.zeros((len(doc_lda), n_topics))
    
    for i, doc_vec in enumerate(doc_lda):
        for (cluster_k, prob) in doc_vec[0]:
            probs[i][cluster_k] = prob
    
    topic_df = pd.DataFrame(probs, columns = ['TopicProb' + str(i) for i in range(n_topics)])
    topic_df['id'] = df['id']

    return topic_df
    