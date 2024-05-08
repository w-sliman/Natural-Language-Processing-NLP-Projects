unknown_token= "<unk>"
start_token='<s>'
end_token = '<e>'

#n_grams smoothing parameter
k = 1.0

#Minimum frequency of a word in the training text to be included in the vocabulary
minimum_freq = 2

# max number of n_grams used for prediction. Choosing 5 as below means we will use 1-gram, 2-gram, up to 5-gram.
n_grams_up_to = 5