# EECS 487 Intro to NLP
# Assignment 1

import math
import random
from collections import defaultdict
from itertools import product
from nltk.tokenize import word_tokenize
from nltk import ngrams
from sklearn.model_selection import train_test_split


class NGramLM:
    """N-gram language model."""

    def __init__(self, bos_token, eos_token, tokenizer, ngram_size):
        self.ngram_count = {}
        for i in range(ngram_size):
            # Each of these will be its own dictionary containing all of the (i-1)-grams
            # For instance, self.ngram_count[0] will be its own dictionary containing all unigrams
            self.ngram_count[i] = None

        self.ngram_size = ngram_size

        self.vocab_sum = None # could be useful in linear interpolation
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.tokenizer = tokenizer
        
    def tokenize_review(self, review, num_bos):
        return [self.bos_token] * num_bos + self.tokenizer(review.lower()) + [self.eos_token]

    def get_ngram_counts(self, reviews):

        #################################################################################
        # TODO: store counts in self.ngram_count
        #################################################################################
        
        # bookkeeping regarding if we're in word level or char level mode.
        is_word_level = len(self.tokenizer("test sentence")) == 2
        num_bos = 2 if is_word_level else 3
        needed_ngrams = [1, 2, 3] if is_word_level else [1, 2, 3, 4]
        
        # 1) PREPEND EOS AND BOS, CONVERT TO LOWERCASE, AND TOKENIZE ALL REVIEWS
        tokenized_reviews = [self.tokenize_review(review, num_bos) for review in reviews]
        
        # 2) REMOVE ALL WORDS THAT OCCUR < 2 TIMES AND REPLACE WITH UNK
        def zero_count():
            return 0
        
        token_counts = defaultdict(zero_count)
        
        for review in tokenized_reviews:
            for token in review:
                token_counts[token] += 1
            
        for i, review in enumerate(tokenized_reviews):
            for j, token in enumerate(review):
                if token_counts[token] < 2:
                    tokenized_reviews[i][j] = 'UNK'
            
        # 3) CALCULATE ALL NGRAM COUNTS
        def new_ngram(ngram_len, curr_ngram, new_token):
            updated_ngram = None
            meets_required_len = False
            
            # if a unigram, return the new token alone; else slice off first elt and and new token
            if ngram_len == 1:
                updated_ngram = new_token
            else:
                chopped_if_full = curr_ngram if len(curr_ngram) < ngram_len else curr_ngram[1:]
                updated_ngram = chopped_if_full + (new_token,)
            
            # if adding the new token doesn't reach length, still not long enough
            if len(curr_ngram) + 1 < ngram_len:
                meets_required_len = False
            else:
                meets_required_len = True
            
            return (meets_required_len, updated_ngram)
            
        for ngram_len in needed_ngrams:
            # initialize dicts and state
            self.ngram_count[ngram_len - 1] = defaultdict(zero_count)
            curr_ngram = ()
            
            for review in tokenized_reviews:
                for token in review:
                    # generate a new ngram of the desired length from the token
                    valid, curr_ngram = new_ngram(ngram_len, curr_ngram, token)
                    
                    # if enough words have been strung together, update count
                    if (valid):
                        self.ngram_count[ngram_len - 1][curr_ngram] += 1

        #################################################################################
    
    def replace_unknown_words_with_UNK(self, ngram, override_tupling=False):
        new_ngram = []
        for token in ngram:
            if token not in self.ngram_count[0]:
                new_ngram.append('UNK')
            else:
                new_ngram.append(token)
                
        return new_ngram if override_tupling else tuple(new_ngram) 
        
        
    def pr_unigram_given_last_j_of_ngram(self, unigram, ngram, j, k=0):
        if type(unigram) is not str:
            print("Non-string unigram supplied to pr_unigram_given_last_j_of_ngram")
            return None
        elif len(ngram) < j:
            print("Cannot remove last j from ngram as it is too short")
            return None
        
        unigram = self.replace_unknown_words_with_UNK((unigram,))[0]
        ngram = self.replace_unknown_words_with_UNK(ngram)
                
        # if condition is empty (aka if calculating unigrams standalone probability)
        if j == 0:
            # count of unigram frequency over the total number of tokens in corpus (not unique and w/o <s>)
            total_tokens = sum(self.ngram_count[0].values()) - self.ngram_count[0][self.bos_token]
            return 0 if unigram not in self.ngram_count[0] else self.ngram_count[0][unigram] / total_tokens
        
        vocab_size = len(self.ngram_count[0]) - 1
        
        def unwrap_if_unit_length(t):
            return t if len(t) != 1 else t[0]
        
        ngram_to_condition_on = ngram[-j:]
        ngram_condition_level = len(ngram_to_condition_on)
        
        count_of_condition_then_uni = (
            self.ngram_count[ngram_condition_level][ngram_to_condition_on + (unigram,)] + k
        )
        count_of_condition = (
            self.ngram_count[ngram_condition_level - 1][unwrap_if_unit_length(ngram_to_condition_on)] 
                + k * vocab_size
        )
        
#         print(str(ngram_to_condition_on) + str(unigram) + ": " + str(count_of_condition_then_uni))
#         print(str(ngram_to_condition_on)+ ": " + str(count_of_condition))
        
        return 0 if count_of_condition == 0 else count_of_condition_then_uni / count_of_condition
        
        
    def add_k_smooth_prob(self, n_minus1_gram, unigram, k):
        probability = 0

        #################################################################################
        # TODO: calculate probability using add-k smoothing
        #################################################################################
        probability = self.pr_unigram_given_last_j_of_ngram(unigram, n_minus1_gram, len(n_minus1_gram), k)
        #################################################################################

        return probability

    def linear_interp_prob(self, n_minus1_gram, unigram, lambdas):
        probability = 0

        #################################################################################
        # TODO: calculate probability using linear interpolation
        #################################################################################
        for i, lambda_i in enumerate(lambdas):
            num_prec = len(lambdas) - i - 1
            probability += lambda_i * self.pr_unigram_given_last_j_of_ngram(unigram, n_minus1_gram, num_prec)
        #################################################################################

        return probability
    
    def get_probability(self, n_minus1_gram, unigram, smoothing_args):
        probability = 0

        #################################################################################
        # TODO: calculate probability using add-k smoothing or linear interpolation
        #################################################################################
        if smoothing_args['method'] == 'add_k':
            probability = self.add_k_smooth_prob(n_minus1_gram, unigram, smoothing_args['k'])
        else:
            probability = self.linear_interp_prob(n_minus1_gram, unigram, smoothing_args['lambdas'])
        #################################################################################

        return probability
    
    def get_perplexity(self, text, smoothing_args):
        perplexity = 0

        #################################################################################
        # TODO: calculate perplexity for text
        #################################################################################
        # tokenize
        is_word_level = len(self.tokenizer("test sentence")) == 2
        num_precursors = 2 if is_word_level else 3
        tokenized_text = self.tokenize_review(text, num_precursors)
        
        # replace unknown words with UNK
        tokenized_text_with_unk = self.replace_unknown_words_with_UNK(tokenized_text, True)
                
        precursors = ()
        log_prob_sum = 0
        
        # for each word given prev two, compute the log of get_probability on it and add it up
        for token in tokenized_text_with_unk:
            if token != self.bos_token:
                prob = self.get_probability(precursors, token, smoothing_args)
                log_prob_sum += math.log(prob)
                
            # adjust precursor set so next has 'num_precursors' tokens before it
            chopped_if_full = precursors if len(precursors) < num_precursors else precursors[1:]
            precursors = chopped_if_full + (token,)
            
        # scale by the negative inverse of the size of the document
        perplexity = math.exp((-1 / (len(tokenized_text) - num_precursors)) * log_prob_sum)
        #################################################################################

        return perplexity
    
    def search_k(self, dev_data):
        best_k = 0

        #################################################################################
        # TODO: find best k value
        #################################################################################
        best_pp_avg = None
                              
        for k in [0.2, 0.4, 0.6, 0.8, 1]:
            pp_sum = 0

            for review in dev_data:
                pp_sum += self.get_perplexity(review, {'method': 'add_k', 'k':k})
                              
            pp_avg = pp_sum / len(dev_data)
            print(str(k) + ": " + str(pp_avg))
                              
            if best_pp_avg is None or pp_avg < best_pp_avg:
                best_k = k
                best_pp_avg = pp_avg
        #################################################################################

        return best_k
    
    def search_lambda(self, dev_data):
        best_lambda = [0, 0, 0]

        #################################################################################
        # TODO: find best lambda values
        #################################################################################
        best_pp_avg = None
        is_word_level = len(self.tokenizer("test sentence")) == 2
        i_lb = 3 if is_word_level else 1
        
        lambdas = set()
        for i in range(i_lb, 10):
            for j in range(1, 10):
                for k in range(1, 10):
                    if (len(self.tokenizer("test sentence")) == 2):
                        if (i + j + k == 10):
                            lambdas.add((i / 10, j / 10, k / 10))
                    else:
                        for l in range(1, 10):
                            if i + j + k + l == 10:
                                lambdas.add((i / 10, j / 10, k / 10, l / 10))
                        
        lambdas = [list(tup) for tup in lambdas]
            
        for i, curr_lambdas in enumerate(lambdas):
            pp_sum = 0

            for review in dev_data:
                pp_sum += self.get_perplexity(review, {'method': 'linear', 'lambdas': curr_lambdas})
                              
            pp_avg = pp_sum / len(dev_data)
                              
            if best_pp_avg is None or pp_avg < best_pp_avg:
                best_lambda = curr_lambdas
                best_pp_avg = pp_avg
                
            print(str(curr_lambdas)  + " had PP: " + str(pp_avg))
                
        print("\nBEST PP: " + str(best_pp_avg) + " from " + str(best_lambda) + "\n")
        #################################################################################

        return best_lambda
    
    def generate_text(self, prompt, smoothing_args):
        generated_text = prompt.copy()

        #################################################################################
        # TODO: generate text based on prompt
        #################################################################################
        is_word_level = len(self.tokenizer("test sentence")) == 2
        num_precursors = 2 if is_word_level else 3
        generated_text = [t.lower() for t in generated_text]
        generated_text = self.replace_unknown_words_with_UNK(generated_text, True)
        last_generated = generated_text[-1]
        
        while len(generated_text) < 15 and last_generated != self.eos_token:
            curr_precursors = generated_text[-num_precursors:]
            
            all_uni = list(self.ngram_count[0].keys())
            pr_of_all_uni = [self.get_probability(curr_precursors, uni, smoothing_args) for uni in all_uni]

            last_generated = random.choices(all_uni, weights=pr_of_all_uni)[0]
            generated_text.append(last_generated)
        #################################################################################

        print(' '.join(generated_text))

def load_new_data(df):

    df_class1 = None
    df_class2 = None

    #################################################################################
    # TODO: load the reviews based on a split of your choosing
    #################################################################################
    split_name = "stars"
    # split by starts <= 2 and > 2 (bad and not bad)
    df_class1 = df[df["stars"] > 2]
    df_class2 = df[df["stars"] <= 2]
    
    all_text1 = df_class1["text"]
    all_text2 = df_class2["text"]
    
    class1_trn, class1_dev = train_test_split(all_text1, test_size=0.2, random_state=42)
    class1_trn, class1_dev = class1_trn.reset_index(drop=True), class1_dev.reset_index(drop=True)

    class2_trn, class2_dev = train_test_split(all_text2, test_size=0.2, random_state=42)
    class2_trn, class2_dev = class2_trn.reset_index(drop=True), class2_dev.reset_index(drop=True)
    #################################################################################

    display(df_class1[["text", split_name]])
    display(df_class2[["text", split_name]])

    return (class1_trn, class1_dev, class2_trn, class2_dev)
    

def predict_class(test_file, class1_lm, class2_lm, smoothing_args):

    class1_ppl = 0
    class2_ppl = 0

    #################################################################################
    # TODO: load the review in test_file, predict its class
    #################################################################################
    text = ''
    with open(test_file, 'r') as f:
        text = f.readlines()[0]
    
    class1_ppl = class1_lm.get_perplexity(text, smoothing_args)
    class2_ppl = class2_lm.get_perplexity(text, smoothing_args)
    #################################################################################

    print(f"Perplexity for class1_lm: {class1_ppl}")
    print(f"Perplexity for class2_lm: {class2_ppl}")
    if class1_ppl < class2_ppl:
        print("It is in class 1")
    else:
        print("It is in class 2")
