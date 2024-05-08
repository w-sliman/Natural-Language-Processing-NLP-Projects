import constants

unknown_token = constants.unknown_token
start_token = constants.start_token
end_token = constants.end_token

#smoothing parameter
k = constants.k

def count_n_grams(data, n, start_token= start_token, end_token = end_token):
    """
    Count all n-grams in the data
    
    Args:
        data: List of lists of words
        n: number of words in a sequence
    
    Returns:
        A dictionary that maps a tuple of n-words to its frequency
    """

    n_grams = {}

    for sentence in data:
        sentence = [start_token] * n + sentence + [end_token]
           
        for i in range(len(sentence)-n+1):
            n_gram = tuple(sentence[i:i+n])

            if n_gram in n_grams:
                n_grams[n_gram] += 1

            else:
                n_grams[n_gram] = 1
    
    return n_grams


def estimate_probability(word, previous_n_gram, n_gram_counts, n_plus1_gram_counts, vocabulary_size, k= k):
    """
    Estimate the probabilities of a next word using the n-gram counts with k-smoothing
    
    Args:
        word: next word
        previous_n_gram: A sequence of words of length n
        n_gram_counts: Dictionary of counts of n-grams
        n_plus1_gram_counts: Dictionary of counts of (n+1)-grams
        vocabulary_size: number of words in the vocabulary
        k: positive constant, smoothing parameter
    
    Returns:
        A probability
    """

    previous_n_gram = tuple(previous_n_gram)
    previous_n_gram_count = n_gram_counts.get(previous_n_gram,0)

    denominator = previous_n_gram_count + k * vocabulary_size

    n_plus1_gram = previous_n_gram +(word,)   
    n_plus1_gram_count = n_plus1_gram_counts.get(n_plus1_gram,0)

    numerator = n_plus1_gram_count + k
    probability = numerator / denominator
    
    return probability



def estimate_probabilities(previous_n_gram, n_gram_counts, n_plus1_gram_counts, vocabulary, end_token=end_token, unknown_token= unknown_token,  k= k):
    """
    Estimate the probabilities of next words using the n-gram counts with k-smoothing
    
    Args:
        previous_n_gram: A sequence of words of length n
        n_gram_counts: Dictionary of counts of n-grams
        n_plus1_gram_counts: Dictionary of counts of (n+1)-grams
        vocabulary: List of words
        k: positive constant, smoothing parameter
    
    Returns:
        A dictionary mapping from next words to the probability.
    """

    previous_n_gram = tuple(previous_n_gram)    
    vocabulary = vocabulary + [end_token, unknown_token]    
    vocabulary_size = len(vocabulary)    
    
    probabilities = {}
    for word in vocabulary:
        probability = estimate_probability(word, previous_n_gram, 
                                           n_gram_counts, n_plus1_gram_counts, 
                                           vocabulary_size, k=k)
                
        probabilities[word] = probability

    return probabilities