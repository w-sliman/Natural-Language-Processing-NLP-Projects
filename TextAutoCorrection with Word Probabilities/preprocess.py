import re
from collections import Counter

def process_data(file_name):
    """
    Input: 
        A file_name which is found in your current directory. You just have to read it in. 
    Output: 
        words: a list containing all the words in the corpus (text file you read) in lower case. 
    """
    words = []

    with open(file_name, 'r') as file:
        txt = file.read()
    
    txt = txt.lower()
    words = re.findall(r'\w+',txt)

    return words


def get_count(word_l):
    '''
    Input:
        word_l: a set of words representing the corpus. 
    Output:
        word_count_dict: The wordcount dictionary where key is the word and value is its frequency.
    '''

    word_count_dict = {}
    word_count_dict = Counter(word_l)
    
    return word_count_dict


def get_probs(word_count_dict):
    '''
    Input:
        word_count_dict: The wordcount dictionary where key is the word and value is its frequency.
    Output:
        probs: A dictionary where keys are the words and the values are the probability that a word will occur. 
    '''
    probs = {}
    m = sum(word_count_dict.values())

    for word in word_count_dict.keys():
        probs[word] = word_count_dict[word]/m
    
    return probs