from string import punctuation, digits
import numpy as np
import random



#==============================================================================
#===  PART I  =================================================================
#==============================================================================



def get_order(n_samples):
    try:
        with open(str(n_samples) + '.txt') as fp:
            line = fp.readline()
            return list(map(int, line.split(',')))
    except FileNotFoundError:
        random.seed(1)
        indices = list(range(n_samples))
        random.shuffle(indices)
        return indices



def hinge_loss_single(feature_vector, label, theta, theta_0):
    """
    Finds the hinge loss on a single data point given specific classification
    parameters.

    Args:
        `feature_vector` - numpy array describing the given data point.
        `label` - float, the correct classification of the data
            point.
        `theta` - numpy array describing the linear classifier.
        `theta_0` - float representing the offset parameter.
    Returns:
        the hinge loss, as a float, associated with the given data point and
        parameters.
    """
    z = np.dot(theta, feature_vector) + theta_0
    
    # Compute the margin
    margin = label * z
    
    # Compute the hinge loss
    hinge_loss = max(0, 1 - margin)
    
    return hinge_loss



def hinge_loss_full(feature_matrix, labels, theta, theta_0):
    """
    Finds the hinge loss for given classification parameters averaged over a
    given dataset

    Args:
        `feature_matrix` - numpy matrix describing the given data. Each row
            represents a single data point.
        `labels` - numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        `theta` - numpy array describing the linear classifier.
        `theta_0` - real valued number representing the offset parameter.
    Returns:
        the hinge loss, as a float, associated with the given dataset and
        parameters.  This number should be the average hinge loss across all of
    """
    n = feature_matrix.shape[0]  # number of data points
    losses = np.zeros(n)
    
    # Compute the raw predictions for all data points
    predictions = np.dot(feature_matrix, theta) + theta_0
    
    # Compute the margins
    margins = labels * predictions
    
    # Compute the individual hinge losses
    losses = np.maximum(0, 1 - margins)
    
    # Compute the average hinge loss
    average_loss = np.mean(losses)
    
    return average_loss






def perceptron_single_step_update(
        feature_vector,
        label,
        current_theta,
        current_theta_0):
    """
    Updates the classification parameters `theta` and `theta_0` via a single
    step of the perceptron algorithm.  Returns new parameters rather than
    modifying in-place.

    Args:
        feature_vector - A numpy array describing a single data point.
        label - The correct classification of the feature vector.
        current_theta - The current theta being used by the perceptron
            algorithm before this update.
        current_theta_0 - The current theta_0 being used by the perceptron
            algorithm before this update.
    Returns a tuple containing two values:
        the updated feature-coefficient parameter `theta` as a numpy array
        the updated offset parameter `theta_0` as a floating point number
    """
     # Compute prediction
    z = np.dot(current_theta, feature_vector) + current_theta_0
    
    # Check if prediction is correct
    if label * z <= 0:
        # Update theta and theta_0
        new_theta = current_theta + label * feature_vector
        new_theta_0 = current_theta_0 + label
    else:
        new_theta = current_theta
        new_theta_0 = current_theta_0
    
    return new_theta, new_theta_0




def perceptron(feature_matrix, labels, T):
    """
    Runs the full perceptron algorithm on a given set of data. Runs T
    iterations through the data set: we do not stop early.

    NOTE: Please use the previously implemented functions when applicable.
    Do not copy paste code from previous parts.

    Args:
        `feature_matrix` - numpy matrix describing the given data. Each row
            represents a single data point.
        `labels` - numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        `T` - integer indicating how many times the perceptron algorithm
            should iterate through the feature matrix.

    Returns a tuple containing two values:
        the feature-coefficient parameter `theta` as a numpy array
            (found after T iterations through the feature matrix)
        the offset parameter `theta_0` as a floating point number
            (found also after T iterations through the feature matrix).
    """
    n_samples, n_features = feature_matrix.shape
    theta = np.zeros(n_features)
    theta_0 = 0
    
    for t in range(T):
        for i in get_order(n_samples):
            if labels[i] * (np.dot(feature_matrix[i], theta) + theta_0) <= 0:
                theta += labels[i] * feature_matrix[i]
                theta_0 += labels[i]
    
    return theta, theta_0




def average_perceptron(feature_matrix, labels, T):
    n_samples, n_features = feature_matrix.shape
    theta = np.zeros(n_features)
    theta_0 = 0
    theta_sum = np.zeros(n_features)
    theta_0_sum = 0
    count = 0
    
    for t in range(T):
        for i in get_order(n_samples):
            if labels[i] * (np.dot(feature_matrix[i], theta) + theta_0) <= 0:
                theta += labels[i] * feature_matrix[i]
                theta_0 += labels[i]
            theta_sum += theta
            theta_0_sum += theta_0
            count += 1
    
    theta_avg = theta_sum / count
    theta_0_avg = theta_0_sum / count
    
    return theta_avg, theta_0_avg

def pegasos_single_step_update(
        feature_vector,
        label,
        L,
        eta,
        theta,
        theta_0):
    """
    Updates the classification parameters `theta` and `theta_0` via a single
    step of the Pegasos algorithm.  Returns new parameters rather than
    modifying in-place.

    Args:
        `feature_vector` - A numpy array describing a single data point.
        `label` - The correct classification of the feature vector.
        `L` - The lamba value being used to update the parameters.
        `eta` - Learning rate to update parameters.
        `theta` - The old theta being used by the Pegasos
            algorithm before this update.
        `theta_0` - The old theta_0 being used by the
            Pegasos algorithm before this update.
    Returns:
        a tuple where the first element is a numpy array with the value of
        theta after the old update has completed and the second element is a
        real valued number with the value of theta_0 after the old updated has
        completed.
    """
    prediction = np.dot(theta, feature_vector) + theta_0
    
    if label * prediction <= 1:
        theta = (1 - eta * L) * theta + eta * label * feature_vector
        theta_0 = theta_0 + eta * label
    else:
        theta = (1 - eta * L) * theta
    
    return theta, theta_0




def pegasos(feature_matrix, labels, T, L):
    """
    Runs the Pegasos algorithm on a given set of data. Runs T iterations
    through the data set, there is no need to worry about stopping early.  For
    each update, set learning rate = 1/sqrt(t), where t is a counter for the
    number of updates performed so far (between 1 and nT inclusive).

    NOTE: Please use the previously implemented functions when applicable.  Do
    not copy paste code from previous parts.

    Args:
        `feature_matrix` - A numpy matrix describing the given data. Each row
            represents a single data point.
        `labels` - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        `T` - An integer indicating how many times the algorithm
            should iterate through the feature matrix.
        `L` - The lamba value being used to update the Pegasos
            algorithm parameters.

    Returns:
        a tuple where the first element is a numpy array with the value of the
        theta, the linear classification parameter, found after T iterations
        through the feature matrix and the second element is a real number with
        the value of the theta_0, the offset classification parameter, found
        after T iterations through the feature matrix.
    """
    n_samples, n_features = feature_matrix.shape
    
    # Initialize theta and theta_0
    theta = np.zeros(n_features)
    theta_0 = 0
    
    # Initialize update count
    update_count = 0
    
    for t in range(T):
        for i in get_order(n_samples):
            update_count += 1
            
            # Compute learning rate eta
            eta = 1.0 / np.sqrt(update_count)
            
            # Perform Pegasos update
            theta, theta_0 = pegasos_single_step_update(feature_matrix[i], labels[i], L, eta, theta, theta_0)
    
    return theta, theta_0





#==============================================================================
#===  PART II  ================================================================
#==============================================================================



##  #pragma: coderesponse template
##  def decision_function(feature_vector, theta, theta_0):
##      return np.dot(theta, feature_vector) + theta_0
##  def classify_vector(feature_vector, theta, theta_0):
##      return 2*np.heaviside(decision_function(feature_vector, theta, theta_0), 0)-1
##  #pragma: coderesponse end



def classify(feature_matrix, theta, theta_0):
    """
    A classification function that uses given parameters to classify a set of
    data points.

    Args:
        `feature_matrix` - numpy matrix describing the given data. Each row
            represents a single data point.
        `theta` - numpy array describing the linear classifier.
        `theta_0` - real valued number representing the offset parameter.

    Returns:
        a numpy array of 1s and -1s where the kth element of the array is the
        predicted classification of the kth row of the feature matrix using the
        given theta and theta_0. If a prediction is GREATER THAN zero, it
        should be considered a positive classification.
    """
    predictions = np.dot(feature_matrix, theta) + theta_0
    classifications = np.where(predictions > 0, 1, -1)
    return classifications



def classifier_accuracy(
        classifier,
        train_feature_matrix,
        val_feature_matrix,
        train_labels,
        val_labels,
        **kwargs):
    """
    Trains a linear classifier and computes accuracy.  The classifier is
    trained on the train data.  The classifier's accuracy on the train and
    validation data is then returned.

    Args:
        `classifier` - A learning function that takes arguments
            (feature matrix, labels, **kwargs) and returns (theta, theta_0)
        `train_feature_matrix` - A numpy matrix describing the training
            data. Each row represents a single data point.
        `val_feature_matrix` - A numpy matrix describing the validation
            data. Each row represents a single data point.
        `train_labels` - A numpy array where the kth element of the array
            is the correct classification of the kth row of the training
            feature matrix.
        `val_labels` - A numpy array where the kth element of the array
            is the correct classification of the kth row of the validation
            feature matrix.
        `kwargs` - Additional named arguments to pass to the classifier
            (e.g. T or L)

    Returns:
        a tuple in which the first element is the (scalar) accuracy of the
        trained classifier on the training data and the second element is the
        accuracy of the trained classifier on the validation data.
    """
    # Train the classifier on the training data
    theta, theta_0 = classifier(train_feature_matrix, train_labels, **kwargs)
    
    # Predict on training and validation data
    train_predictions = classify(train_feature_matrix, theta, theta_0)
    val_predictions = classify(val_feature_matrix, theta, theta_0)
    
    # Compute accuracy
    training_accuracy = accuracy(train_predictions, train_labels)
    validation_accuracy = accuracy(val_predictions, val_labels)
    
    return training_accuracy, validation_accuracy



def extract_words(text):
    """
    Helper function for `bag_of_words(...)`.
    Args:
        a string `text`.
    Returns:
        a list of lowercased words in the string, where punctuation and digits
        count as their own words.
    """
    
    for c in punctuation + digits:
        text = text.replace(c, ' ' + c + ' ')
    return text.lower().split()


def load_stopwords(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        stopword_list = [line.strip() for line in file]
    return set(stopword_list)

stopwords_file = 'stopwords.txt'
stopwords = load_stopwords(stopwords_file)

def bag_of_words(texts, remove_stopword=False):
    indices_by_word = {}  # maps word to unique index

    for text in texts:
        word_list = extract_words(text)
        for word in word_list:
            if word in stopwords:
                continue
            if word not in indices_by_word:
                indices_by_word[word] = len(indices_by_word)

    return indices_by_word



def extract_bow_feature_vectors(reviews, indices_by_word, binarize=True):
    """
    Args:
        `reviews` - a list of natural language strings
        `indices_by_word` - a dictionary of uniquely-indexed words.
    Returns:
        a matrix representing each review via bag-of-words features.  This
        matrix thus has shape (n, m), where n counts reviews and m counts words
        in the dictionary.
    """
    feature_matrix = np.zeros([len(reviews), len(indices_by_word)], dtype=np.float64)
    
    for i, text in enumerate(reviews):
        word_list = extract_words(text)
        for word in word_list:
            if word in indices_by_word:
                feature_matrix[i, indices_by_word[word]] += 1
    return feature_matrix



def accuracy(preds, targets):
    """
    Given length-N vectors containing predicted and target labels,
    returns the fraction of predictions that are correct.
    """
    return (preds == targets).mean()
