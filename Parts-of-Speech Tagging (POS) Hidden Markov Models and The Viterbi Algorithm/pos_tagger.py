import numpy as np

from preprocessing import preprocess, create_dictionaries, create_transition_matrix, create_emission_matrix
from compute_accuracy import compute_accuracy
from verbati_algorithm import initialize, viterbi_forward, viterbi_backward

class PartsOfSpeechTaggerHMM:
    """
    Parts of Speech Tagger using Hidden Markov Models.

    Attributes:
    - alpha (float): Smoothing parameter for emission and transition probabilities.
    - vocab (set): A set containing the vocabulary of known words.
    - emission_counts (defaultdict): A dictionary containing emission counts for each (tag, word) pair.
    - transition_counts (defaultdict): A dictionary containing transition counts for each (prev_tag, tag) pair.
    - tag_counts (defaultdict): A dictionary containing counts of each tag.
    - states (list): A list of all possible tags.
    - transition_matrix (numpy.ndarray): Transition matrix.
    - emission_matrix (numpy.ndarray): Emission matrix.

    Methods:
    - __init__(self, alpha=0.001): Initializes the PartsOfSpeechTaggerHMM object with a given smoothing parameter.
    - fit(self, vocab, training_corpus): Fits the model to training data by creating dictionaries and matrices.
    - preprocess(self, corpus): Preprocesses a corpus.
    - predict_pos(self, corpus_words, test_data=None): Predicts the parts of speech for a given corpus using Viterbi algorithm.
    """

    def __init__(self, alpha = 0.001):
        """
        Initializes the PartsOfSpeechTaggerHMM object with a given smoothing parameter.

        Parameters:
        - alpha (float, optional): Smoothing parameter for emission and transition probabilities. Default is 0.001.
        """
        
        self.alpha = alpha

    def fit(self, vocab, training_corpus):
        """
        Fits the model to training data by creating dictionaries and matrices.

        Parameters:
        - vocab (set): A set containing the vocabulary of known words.
        - training_corpus (list): A list of (word, tag) tuples representing the training corpus.
        """

        self.vocab = vocab

        self.emission_counts, self.transition_counts, self.tag_counts = create_dictionaries(training_corpus, self.vocab)
        self.states = sorted(self.tag_counts.keys())

        self.transition_matrix = create_transition_matrix(self.alpha, self.tag_counts, self.transition_counts)
        self.emission_matrix = create_emission_matrix(self.alpha, self.tag_counts, self.emission_counts, list(self.vocab))
    
    def preprocess(self, corpus):
        """
        Preprocesses a corpus by replacing unknown words and tagging empty strings.

        Parameters:
        - corpus (list): A list of words from the corpus to preprocess.

        Returns:
        - prep (list): A list containing the preprocessed words.
        """

        _, prep = preprocess(self.vocab, corpus)

        return prep

    def predict_pos(self, corpus_words, test_data = None):  
        """
        Predicts the parts of speech for a given corpus using the Viterbi algorithm.

        Parameters:
        - corpus_words (list): A list of words from the corpus to predict parts of speech for.
        - test_data (list, optional): Test Corpus with labels for evaluation. Default is None.

        Returns:
        - pred (list): A list of predicted parts of speech for each word in the corpus.
        """
        
        prep = self.preprocess(corpus_words)

        best_probs, best_paths = initialize(
            self.states, self.tag_counts, self.transition_matrix,  self.emission_matrix,  prep, self.vocab)
        
        best_probs, best_paths = viterbi_forward(
            self.transition_matrix, self.emission_matrix,  prep,  best_probs,  best_paths, self.vocab)

        pred = viterbi_backward(best_probs, best_paths, prep, self.states)

        if test_data:
            print(f"Accuracy of the Viterbi algorithm is {compute_accuracy(pred, test_data):.4f}")

        return pred