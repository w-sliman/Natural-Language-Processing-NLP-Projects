import re
from preprocess import process_data, get_count, get_probs
from get_corrections import get_corrections

class AutoCorrect:
    def __init__(self):
        """
        Initializes an instance of the AutoCorrect class.
        """
        self.vocab = ()
        self.word_count_dict = {}
        self.probs = {}

    def fit(self, base_data):
        """
        Fits the AutoCorrect model to the provided base data.

        Args:
            base_data (str): The location of the base data file used for training the AutoCorrect model.

        Returns:
            AutoCorrect: The fitted AutoCorrect object.
        """
        word_l = process_data(base_data)
        self.vocab = set(word_l)

        print(f"There are {len(self.vocab)} unique words in the vocabulary.")

        self.word_count_dict = get_count(word_l)
        self.probs = get_probs(self.word_count_dict)
        
        return self

    def correct(self, input_text, n_best=1):
        """
        Corrects misspelled words in the input text.

        Args:
            input_text (str): The input text to be corrected.
            n_best (int): The number of best suggestions to return for each misspelled word.

        Returns:
            str: The corrected text.
        """
        words = input_text.split()

        corrected_words = []
        for word in words:
            punctuation = ''
            if not word[-1].isalnum():
                punctuation = word[-1]
                word = word[:-1]

            corrected_word = get_corrections(word, self.probs, self.vocab, n=n_best, verbose=False)[0][0]
            corrected_words.append(corrected_word + punctuation)

        corrected_text = ' '.join(corrected_words)

        return corrected_text
