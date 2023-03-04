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

def get_corpus(df):
    # Convert to list 
    data_words = [word_tokenize(sentence) for sentence in df.body.values]
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

def base_df_to_topics_df(df, id2word, corpus, n_topics=20):
    lda_model = gensim.models.ldamulticore.LdaMulticore(corpus=corpus,
                                                    id2word=id2word,
                                                    workers=6,
                                                    num_topics=n_topics, 
                                                    random_state=100,
                                                    chunksize=50,
                                                    passes=8,
                                                    alpha='asymmetric',
                                                    per_word_topics=True)

    doc_lda = lda_model[corpus] 
    return lda_model, doc_lda

def get_scores(lda_model, id2word, corpus):
    # Compute Perplexity
    print('\nPerplexity: ', lda_model.log_perplexity(corpus))  
    # a measure of how good the model is. lower the better.

    # Compute Coherence Score
    coherence_model_lda = CoherenceModel(model=lda_model, texts=data, dictionary=id2word, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print('\nCoherence Score: ', coherence_lda)
    pass

def create_topic_df(df, doc_lda, n_topics=20):
    topic_df = pd.DataFrame(doc_lda, columns = ['Topic ' + str(i) for i in range(n_topics)])
    topic_df['id'] = df['id']

    return topic_df
    