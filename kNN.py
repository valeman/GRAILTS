import numpy as np
import math
from Correlation import Correlation
import exceptions
from Representation import Representation
import heapq


def kNN(TRAIN, TEST, method, k, representation=None, **kwargs):
    """
    Approximate or exact k-nearest neighbors algorithm depending on the representation
    :param TRAIN: The training set to get the neighbors from
    :param TEST: The test set whose neighbors we get
    :param method: The correlation or distance measure being used
    :param k: how many nearest neighbors to return
    :param representation: The representation being used if any, for instance GRAIL. This is a representation object.
    :param **kwargs: arguments for the correlator
    :return: a matrix of size row(TEST)xk
    """
    rowTEST = TEST.shape[0]
    rowTRAIN = TRAIN.shape[0]
    colTEST = TEST.shape[1]
    colTRAIN = TRAIN.shape[1]

    neighbors = np.zeros((rowTEST, k))
    if representation:
        TRAIN = representation.get_representation(TRAIN)
        TEST = representation.get_representation(TEST)

    for i in range(rowTEST):
        x = TEST[i, :]
        corr_array = np.zeros(rowTRAIN)
        for j in range(rowTRAIN):
            y = TRAIN[j, :]
            correlation = Correlation(x, y, correlation_protocol_name=method, **kwargs)
            corr_array[j] = correlation.correlate()
        if Correlation.is_similarity(method):
            neighbors[i,:] = heapq.nlargest(k, range(len(corr_array)), corr_array.take())
        else:
            neighbors[i,:] = heapq.nsmallest(k, range(len(corr_array)), corr_array.take())
    return neighbors
