import nltk
import os
from preprocessing import get_tokenized_data, get_words_with_nplus_frequency, replace_oov_words_by_unk
from n_grams import estimate_probabilities, count_n_grams

import constants

start_token = constants.start_token
minimum_freq = constants.minimum_freq
n_grams_up_to = constants.n_grams_up_to

#smoothing factor
k = constants.k

def suggest_a_word(previous_tokens, n_gram_counts, n_plus1_gram_counts, vocabulary, start_token=start_token, k= k, start_with=None):
    """
    Get suggestion for the next word
    
    Args:
        previous_tokens: The sentence you input where each token is a word. Must have length >= n 
        n_gram_counts: Dictionary of counts of n-grams
        n_plus1_gram_counts: Dictionary of counts of (n+1)-grams
        vocabulary: List of words
        k: positive constant, smoothing parameter
        start_with: If not None, specifies the first few letters of the next word
        
    Returns:
        A tuple of 
          - string of the most likely next word
          - corresponding probability
    """

    n = len(list(n_gram_counts.keys())[0])
    previous_tokens = [start_token] * n + previous_tokens
    previous_n_gram = previous_tokens[-n:]

    probabilities = estimate_probabilities(previous_n_gram,
                                           n_gram_counts, n_plus1_gram_counts,
                                           vocabulary, k=k)

    suggestion = None
    max_prob = 0
    
    for word, prob in probabilities.items():
        if start_with:
            if not word.startswith(start_with):
                continue
        
        if prob > max_prob:
            suggestion = word
            max_prob = prob
    
    return suggestion, max_prob


def get_suggestions(previous_tokens, n_gram_counts_list, vocabulary, k=k, start_with=None):

    model_counts = len(n_gram_counts_list)
    suggestions = []

    for i in range(model_counts-1):
        n_gram_counts = n_gram_counts_list[i]
        n_plus1_gram_counts = n_gram_counts_list[i+1]
        
        suggestion = suggest_a_word(previous_tokens, n_gram_counts,
                                    n_plus1_gram_counts, vocabulary,
                                    k=k, start_with=start_with)
        suggestions.append(suggestion)

    return suggestions


class AutoComplete:

    def __init__(self):
        
        if os.getcwd() not in nltk.data.path:
            nltk.data.path.append(os.getcwd())

        exists = False
        for path in nltk.data.path:
            if os.path.isfile(os.path.join(path,"tokenizers","punkt","english.pickle")) and os.path.isfile(os.path.join(path,"tokenizers","punkt","PY3","english.pickle")):
                exists = True
                break

        if not exists:
            print("downloading punkt to current working directory from nltk...")
            nltk.download('punkt', download_dir=os.getcwd())

    def fit(self, train_data, minimum_freq = minimum_freq, n_grams_up_to = n_grams_up_to):

        print("Tokenizing training data...")
        tokenized_data = get_tokenized_data(train_data)

        print("Extracting vocabulary...")
        self.vocab = get_words_with_nplus_frequency(tokenized_data, minimum_freq)

        print("Processing unknown Tokens in training data...")
        self.train_data_processed = replace_oov_words_by_unk(tokenized_data, self.vocab)

        self.n_gram_counts_list = []

        for n in range(1, n_grams_up_to + 1):

            print(f"Computing n-gram counts with n = {n} of {n_grams_up_to}...")
            n_model_counts = count_n_grams(self.train_data_processed, n)
            self.n_gram_counts_list.append(n_model_counts)
        
        print("The model has been successfully fit.")

        return self
    
    def predict(self, sentence, start_with= None, k= k):

        sentence = sentence.lower()
        previous_tokens = nltk.word_tokenize(sentence)
        suggest = get_suggestions(previous_tokens, self.n_gram_counts_list, self.vocab, k=k, start_with = start_with)

        return suggest


    

