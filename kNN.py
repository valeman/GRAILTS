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
    correlations = np.zeros((rowTEST, k))
    if representation:
        together = np.vstack((TRAIN, TEST))
        rep_together = representation.get_representation(together)
        TRAIN = rep_together[0:rowTRAIN, :]
        TEST = rep_together[rowTRAIN:, :]

    for i in range(rowTEST):
        x = TEST[i, :]
        corr_array = np.zeros(rowTRAIN)
        for j in range(rowTRAIN):
            y = TRAIN[j, :]
            correlation = Correlation(x, y, correlation_protocol_name=method, **kwargs)
            corr_array[j] = correlation.correlate()
        if Correlation.is_similarity(method):
            temp = np.array(heapq.nlargest(k, enumerate(corr_array), key = lambda x: x[1]))
            neighbors[i,:] = temp[:, 0]
            correlations[i,:] = temp[:,1]
        else:
            temp = np.array(heapq.nsmallest(k, enumerate(corr_array), key = lambda x: x[1]))
            neighbors[i,:] = temp[:, 0]
            correlations[i,:] = temp[:,1]
    return neighbors, correlations
