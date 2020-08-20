import numpy as np

def kNN(TRAIN, TEST, method, k, representation = None):
    """
    Approximate or exact k-nearest neighbors algorithm depending on the representation
    :param TRAIN: The training set to get the neighbors from
    :param TEST: The test set whose neighbors we get
    :param method: The correlation or distance measure being used
    :param k: how many nearest neighbors to return
    :param representation: The representation being used if any, for instance GRAIL
    :return: a matrix of size row(TEST)xk
    """
    rowTEST = TEST.shape[0]
    rowTRAIN = TRAIN.shape[0]
    colTEST = TEST.shape[1]
    colTRAIN = TRAIN.shape[1]

    neighbors = np.zeros((rowTEST, k))
    if representation == "GRAIL":
        pass

    if method == "Pearson":
        similarity = True
    elif method == "ED":
        similarity = False
    elif method == "NCC":
        similarity = True
    elif method == "SINK":
        similarity = True
    elif method == "SINK_compressed":
        similarity = True
    elif method == "NCC_compressed":
        similarity = True

    for i in range(rowTEST):
        x = TEST[i]

        pass

    return neighbors