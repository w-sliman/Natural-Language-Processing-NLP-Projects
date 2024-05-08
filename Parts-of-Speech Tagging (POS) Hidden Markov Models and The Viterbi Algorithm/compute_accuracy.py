def compute_accuracy(pred, y):
    """
    Computes the accuracy of predictions compared to true labels.

    Parameters:
    - pred (list): A list of predicted tags.
    - y (list): A list of true labels.

    Returns:
    - accuracy (float): The accuracy of the predictions.
    """
    num_correct = 0
    total = 0
    
    for prediction, y in zip(pred, y):
        word_tag_tuple = y.split()

        if len(word_tag_tuple) !=2 :
            continue

        word, tag = word_tag_tuple
        
        if prediction == tag: 
            num_correct += 1
            
        total += 1

    return num_correct/total



