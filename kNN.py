import numpy as np
import math
from Correlation import Correlation
import exceptions
from Representation import Representation
import heapq
import OPQ
import PQ


def kNN(TRAIN, TEST, method, k, representation=None, pq_method = None, **kwargs):
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
    if pq_method:
        return kNN_with_pq(TRAIN, TEST, method, k, representation=None, pq_method = pq_method, **kwargs)

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
    neighbors = neighbors.astype(int)
    return neighbors, correlations

#check the returned distances
def kNN_with_pq(TRAIN, TEST, method, k, representation=None, pq_method = "opq", **kwargs):
    if method != "ED":
        raise ValueError("Product Quantization can only be used with ED.")

    rowTEST = TEST.shape[0]
    rowTRAIN = TRAIN.shape[0]

    neighbors = np.zeros((rowTEST, k))
    distances = np.zeros((rowTEST, k))
    if representation:
        together = np.vstack((TRAIN, TEST))
        rep_together = representation.get_representation(together)
        TRAIN = rep_together[0:rowTRAIN, :]
        TEST = rep_together[rowTRAIN:, :]

    TRAIN = TRAIN.astype(np.float32)
    TEST = TEST.astype(np.float32)

    if rowTRAIN < 256:
        code_word_num = rowTRAIN - 1

    if pq_method == "opq":
        pq = OPQ.OPQ(M=TRAIN.shape[1], Ks= code_word_num, verbose=False)
    elif pq_method == "pq":
        pq = PQ.PQ(M=TRAIN.shape[1], Ks= code_word_num, verbose=False)
    else:
        raise ValueError("Product quantization method not found.")

    pq.fit(vecs=TRAIN)
    TRAIN_code = pq.encode(vecs=TRAIN)

    for i in range(rowTEST):
        query = TEST[i, :]
        dists = pq.dtable(query=query).adist(codes=TRAIN_code)
        temp = np.array(heapq.nsmallest(k, enumerate(dists), key = lambda x: x[1]))
        neighbors[i,:] = temp[:, 0]
        distances[i,:] = temp[:,1]
    neighbors = neighbors.astype(int)
    return neighbors, distances

def kNN_classifier(TRAIN, train_labels, TEST, method, k, representation=None, pq_method = None, **kwargs):
    neighbors, _ = kNN(TRAIN, TEST, method, k, representation, pq_method, **kwargs)
    return_labels = np.zeros(TEST.shape[0])
    for i in range(TEST.shape[0]):
        nearest_labels = np.zeros(k)
        for j in range(k):
            nearest_labels[j] = train_labels[neighbors[i,j]]
        unique, counts = np.unique(nearest_labels, return_counts=True)
        mx = 0
        mx_label = 0
        for j in range(unique.shape[0]):
            if counts[j] > mx:
                mx = counts[j]
                mx_label = unique[j]
        return_labels[i] = mx_label
    return return_labels
