import re
import numpy as np
import pandas as  pd
from pprint import pprint# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel# spaCy for preprocessing
import spacy# Plotting tools

from nltk.corpus import stopwords

# Define function for stopwords, bigrams, trigrams and lemmatization

def base_df_to_topics_df(df):
    # Convert to list 
    data = df.body.values.tolist()

    def sent_to_words(sentences):
      for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))            #deacc=True removes punctuations
    data_words = list(sent_to_words(data))
    print(data_words[:1])

    # Initialize spacy 'en' model, keeping only tagger component (for efficiency)
    # python3 -m spacy download en
    nlp = spacy.load('en', disable=['parser', 'ner'])

    # Create Dictionary 
    id2word = corpora.Dictionary(data)  
    # Create Corpus 
    texts = data  
    # Term Document Frequency 
    corpus = [id2word.doc2bow(text) for text in texts]  
    # View 
    print(corpus[:1])

    n_topics=20
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                id2word=id2word,
                                                num_topics=n_topics, 
                                                random_state=100,
                                                update_every=1,
                                                chunksize=100,
                                                passes=10,
                                                alpha='auto',
                                                per_word_topics=True)

    doc_lda = lda_model[corpus] 

    # Compute Perplexity
    print('\nPerplexity: ', lda_model.log_perplexity(corpus))  
    # a measure of how good the model is. lower the better.

    # Compute Coherence Score
    coherence_model_lda = CoherenceModel(model=lda_model, texts=data, dictionary=id2word, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print('\nCoherence Score: ', coherence_lda)

    # try in notebook to visualize:
    # pyLDAvis.enable_notebook()
    # vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
    # vis

    topic_df = pd.DataFrame(doc_lda, columns = ['Topic ' + str(i) for i in range(n_topics)])

    return topic_df, lda_model, corpus, id2word