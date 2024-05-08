from collections import Counter
from string_manipulation import delete_letter, insert_letter, replace_letter, switch_letter

def edit_one_letter(word, allow_switches = True):
    """
    Input:
        word: the string/word for which we will generate all possible wordsthat are one edit away.
    Output:
        edit_one_set: a set of words with one possible edit. Please return a set. and not a list.
    """
    
    edit_one_set = set()
    
    ### START CODE HERE ###
    if allow_switches:
    
        edit_one_set = set(delete_letter(word)+insert_letter(word)+replace_letter(word)+switch_letter(word))
    else:
        edit_one_set = set(delete_letter(word)+insert_letter(word)+replace_letter(word))
    ### END CODE HERE ###
    
    # return this as a set and not a list
    return set(edit_one_set)


def edit_two_letters(word, allow_switches = True):
    '''
    Input:
        word: the input string/word 
    Output:
        edit_two_set: a set of strings with all possible two edits
    '''
    
    edit_two_set = set()
    
    ### START CODE HERE ###
    edit_one_set = edit_one_letter(word, allow_switches = allow_switches)
    edit_two_set = edit_two_set.union(edit_one_set)
    for w in edit_one_set:
        edit_two_set = edit_two_set.union(edit_one_letter(w, allow_switches = allow_switches))
    ### END CODE HERE ###
    
    # return this as a set instead of a list
    return set(edit_two_set)


def get_corrections(word, probs, vocab, n=2, verbose = False):
    '''
    Input: 
        word: a user entered string to check for suggestions
        probs: a dictionary that maps each word to its probability in the corpus
        vocab: a set containing all the vocabulary
        n: number of possible word corrections you want returned in the dictionary
    Output: 
        n_best: a list of tuples with the most probable n corrected words and their probabilities.
    '''
    
    suggestions = []
    n_best = []
    
    if word in vocab:
        suggestions = [word] 
    elif vocab.intersection(edit_one_letter(word)):
        suggestions = [word for word in vocab.intersection(edit_one_letter(word))]
    elif vocab.intersection(edit_two_letters(word)):
        suggestions = [word for word in vocab.intersection(edit_two_letters(word))]
    else: 
        suggestions = []
                    
    if suggestions:
        suggestions_dict = {key: probs[key] for key in suggestions}
        counter_obj = Counter(suggestions_dict)
    
    if suggestions:
        n_best = counter_obj.most_common(n)
    else:
        n_best = (word,0)
    
    if verbose: print("entered word = ", word, "\nsuggestions = ", suggestions)

    return n_best

