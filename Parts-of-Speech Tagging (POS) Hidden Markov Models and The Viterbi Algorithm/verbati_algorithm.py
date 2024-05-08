import numpy as np
import math

def initialize(states, tag_counts, A, B, corpus, vocab):
    """
    Initializes the Verbati algorithm by setting up initial probabilities and paths.

    Parameters:
    - states (list): A list of all possible tags.
    - tag_counts (defaultdict): A dictionary containing counts of each tag.
    - A (numpy.ndarray): Transition matrix.
    - B (numpy.ndarray): Emission matrix.
    - corpus (list): A list of words from the input text.
    - vocab (dict): A dictionary mapping words to their indices.

    Returns:
    - best_probs (numpy.ndarray): Matrix containing the best probabilities for each tag at each position in the corpus.
    - best_paths (numpy.ndarray): Matrix containing the best paths for each tag at each position in the corpus.

    """
   
    num_tags = len(tag_counts)
    best_probs = np.zeros((num_tags, len(corpus)))
    best_paths = np.zeros((num_tags, len(corpus)), dtype=int)
    s_idx = states.index("--s--")
   
    for i in range(num_tags): 
        best_probs[i,0] = math.log(A[s_idx,i]) + math.log(B[i, vocab[corpus[0]]])
            
    return best_probs, best_paths


def viterbi_forward(A, B, test_corpus, best_probs, best_paths, vocab, verbose=True):
    """
    Performs the forward step of the Viterbi algorithm.

    Parameters:
    - A (numpy.ndarray): Transition matrix.
    - B (numpy.ndarray): Emission matrix.
    - test_corpus (list): A list of words from the input text.
    - best_probs (numpy.ndarray): Matrix initialized in the initialize function.
    - best_paths (numpy.ndarray): Matrix initialized in the initialize function.
    - vocab (dict): A dictionary mapping words to their indices.
    - verbose (bool, optional): If True, prints progress during execution. Default is True.

    Returns:
    - best_probs (numpy.ndarray): Updated matrix of best probabilities.
    - best_paths (numpy.ndarray): Updated matrix of best paths.
    """

    num_tags = best_probs.shape[0]
    
    for i in range(1, len(test_corpus)): 
        
        if i % 5000 == 0 and verbose:
            print("Words processed: {:>8}".format(i))
            
        for j in range(num_tags):
            
            best_prob_i = float("-inf")
            best_path_i = None
            
            for k in range(num_tags): 
            
                prob = best_probs[k,i-1] + math.log(A[k,j]) + math.log(B[j,vocab[test_corpus[i]]])

                if prob > best_prob_i: 
                    
                    best_prob_i = prob
                    best_path_i = k

            best_probs[j,i] = best_prob_i
            best_paths[j,i] = best_path_i

    return best_probs, best_paths


def viterbi_backward(best_probs, best_paths, corpus, states):

    """
    Performs the backward step of the Viterbi algorithm to obtain the predicted tags.

    Parameters:
    - best_probs (numpy.ndarray): Matrix containing the best probabilities for each tag at each position in the corpus.
    - best_paths (numpy.ndarray): Matrix containing the best paths for each tag at each position in the corpus.
    - corpus (list): A list of words from the input text.
    - states (list): A list of all possible tags.

    Returns:
    - pred (list): A list of predicted tags for each word in the corpus.
    """
    
    m = best_paths.shape[1]
    z = [None] * m
    num_tags = best_probs.shape[0]
    best_prob_for_last_word = float('-inf')
    pred = [None] * m

    for k in range(num_tags):

        if best_probs[k,m-1] > best_prob_for_last_word: 
            
            best_prob_for_last_word = best_probs[k,m-1]

            z[m - 1] = k
            
    pred[m - 1] = states[z[m - 1]]
    
    for i in range(m - 1, 0 , -1): 
        
        pos_tag_for_word_i = z[i]
        z[i - 1] = best_paths[pos_tag_for_word_i,i]
        pred[i - 1] = states[z[i-1]]

    return pred


