import numpy as np
from collections import defaultdict
from constants import assign_unk

def preprocess(vocab, corpus_words):
    """
    Preprocesses a list of words from a corpus based on a vocabulary.

    Parameters:
    - vocab (set): A set containing the vocabulary of known words.
    - corpus_words (list): A list of words from the corpus to preprocess.

    Returns:
    - orig (list): A list containing the original words from the corpus.
    - prep (list): A list containing the preprocessed words with unknown words replaced by "--n--".

    """
    orig, prep = [], []

    for word in corpus_words:
        
        if not word.split():
            orig.append(word.strip())
            word = "--n--"
            prep.append(word)
            continue

        elif word.strip() not in vocab:
            orig.append(word.strip())
            word = assign_unk(word)
            prep.append(word)
            continue

        else:
            orig.append(word.strip())
            prep.append(word.strip())

    return orig, prep


def get_word_tag(line, vocab): 
    """
    Extracts word and tag from a line of text.

    Parameters:
    - line (str): A line of text containing a word and its tag separated by a space.
    - vocab (set): A set containing the vocabulary of known words.

    Returns:
    - word (str): The word extracted from the line. If line is empty, returns "--n--".
    - tag (str): The tag extracted from the line. If line is empty, returns "--s--".
    """

    if not line.split():
        word = "--n--"
        tag = "--s--"
        return word, tag
    else:
        word, tag = line.split()
        if word not in vocab: 
            word = assign_unk(word)
        return word, tag
    return None 


def create_dictionaries(training_corpus, vocab, verbose=True):

    """
    Creates dictionaries for emission, transition, and tag counts based on a training corpus.

    Parameters:
    - training_corpus (list): A list of (word, tag) tuples representing the training corpus.
    - vocab (set): A set containing the vocabulary of known words.
    - verbose (bool, optional): If True, prints word count progress. Default is True.

    Returns:
    - emission_counts (defaultdict): A dictionary containing emission counts for each (tag, word) pair.
    - transition_counts (defaultdict): A dictionary containing transition counts for each (prev_tag, tag) pair.
    - tag_counts (defaultdict): A dictionary containing counts of each tag.

    """
        
    emission_counts = defaultdict(int)
    transition_counts = defaultdict(int)
    tag_counts = defaultdict(int)
    
    prev_tag = '--s--' 
    i = 0 

    for word_tag in training_corpus:
        
        i += 1
        if i % 50000 == 0 and verbose:
            print(f"word count = {i}")
        
        word, tag = get_word_tag(word_tag, vocab)
        transition_counts[(prev_tag, tag)] += 1
        emission_counts[(tag, word)] += 1
        tag_counts[tag] += 1
        prev_tag = tag
        
    return emission_counts, transition_counts, tag_counts


def create_transition_matrix(alpha, tag_counts, transition_counts):

    """
    Creates a transition matrix based on transition counts.

    Parameters:
    - alpha (float): Smoothing parameter.
    - tag_counts (defaultdict): A dictionary containing counts of each tag.
    - transition_counts (defaultdict): A dictionary containing transition counts for each (prev_tag, tag) pair.

    Returns:
    - A (numpy.ndarray): Transition matrix where A[i, j] represents the probability of transitioning from tag i to tag j.

    """

    all_tags = sorted(tag_counts.keys())
    num_tags = len(all_tags)
    A = np.zeros((num_tags,num_tags))
    trans_keys = set(transition_counts.keys())

    for i in range(num_tags):
        for j in range(num_tags):
            count = 0
            key = (all_tags[i], all_tags[j])

            if key in transition_counts:
                count = transition_counts[key]                

            count_prev_tag = tag_counts[all_tags[i]]
            A[i,j] = (count + alpha)/(count_prev_tag + alpha * num_tags)

    return A


def create_emission_matrix(alpha, tag_counts, emission_counts, vocab):
    
    """
    Creates an emission matrix based on emission counts.

    Parameters:
    - alpha (float): Smoothing parameter.
    - tag_counts (defaultdict): A dictionary containing counts of each tag.
    - emission_counts (defaultdict): A dictionary containing emission counts for each (tag, word) pair.
    - vocab (list): A list of all words in the vocabulary.

    Returns:
    - B (numpy.ndarray): Emission matrix where B[i, j] represents the probability of emitting word j given tag i.
    """
    
    num_tags = len(tag_counts)
    all_tags = sorted(tag_counts.keys())
    num_words = len(vocab)
    B = np.zeros((num_tags, num_words))
    emis_keys = set(list(emission_counts.keys()))

    for i in range(num_tags): 
        for j in range(num_words): 
            count = 0
            key = (all_tags[i],vocab[j]) 
            
            if key in emission_counts:
                count = emission_counts[key]

            count_tag = tag_counts[all_tags[i]]
            B[i,j] = (count + alpha) / (count_tag + alpha * num_words)

    return B



