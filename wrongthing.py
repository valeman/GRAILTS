from SINK import SINK
import numpy as np
from numpy.linalg import svd as SVD
import time
from scipy import stats
import random
from FrequentDirections import FrequentDirections
from kshape import matlab_kshape
import Representation
from scipy import linalg
import math

#returns the pearson correlation coefficient
def pearson(x,y):
    return stats.pearsonr(x,y)[0]

def spearman(x,y):
    return stats.spearmanr(x,y)[0]

def kendall_tau(x,y)
    tau, pval = stats.kendalltau(x,y)
    return tau

#returns the same function with tuned parameters helper for Grail
def tune_parameters(func, *args):
    return lambda x,y: func(x,y,args)

def ED(x,y):
    return np.linalg.norm(x-y)

def kShape_general(A, k,initialization = "random partitions", similarity = "SINK", *args):
    '''
    shape based clustering algorithm
    :param X: nxm matrix containing time series that are z-normalized
    :param k: number of clusters
    :param similarity: the similarity function for clustering
    :return: index is the length n array containing the index of the clusters to which
    the series are assigned. centroids is the kxm matrix containing the centroids of
    the clusters
    '''

    m = A.shape[0]
    if initialization == "random partitions":
        mem = np.zeros(m)
        for i in range(m):
            mem[i] = random.randrange(k)
        cent = np.zeros((k, A.shape[1]))

    if similarity == "SINK":
        kernel = tune_parameters(SINK, args)

    for iter in range(100):
        prev_mem = mem.copy()
        D = math.inf * np.ones((m,k))

        for i in range(k):
            cent[i,:] = kshape_centroid(A, mem ,cent[i,:], i)
            cent[i,:] = stats.zscore(cent[i,:],ddof= 1)

        for i in range(m):
            for j in range(k):
                dist = 1 - max(SINK.NCC(A[i,:], cent[j,:]) )
                D[i, j] = dist

        for i in range(m):
            mem[i] = np.argmin(D[i,:])

        if linalg.norm(prev_mem - mem) == 0:
            break

    return [mem, cent]


def GRAIL_general(X, d, f, similarity = "SINK", *args):
    """

    :param X: nxm matrix that contains n time series
    :param d: number of landmark series to extract from kshape
    :param f: scalar to tune the dimensionality k of Z_k
    :param r: parameter for tuning gamma, taken as 20 in the paper
    :param args: parameters for the similarity function
    :return: Z_k nxk matrix of low dimensional reduced representation
    """

    if similarity == "SINK":
        kernel = tune_parameters(SINK, args)
    elif similarity == "Pearson":
        kernel = pearson
    elif similarity == "Spearman":
        kernel = spearman
    elif similarity == "Kendall":
        kernel = kendall_tau
    elif similarity == "ED":
        kernel = lambda x,y: 1/(1 + ED(x,y))
    else:
        kernel = similarity

    n = X.shape[0]
    [mem, Dictionary] = matlab_kshape(X,d)
    W = np.zeros((d, d))

    for i in range(d):
        for j in range(d):
            W[i, j] = kernel(Dictionary[i, :], Dictionary[j, :])

    E = np.zeros((n, d))

    for i in range(n):
        for j in range(d):
            E[i, j] = kernel(X[i, :], Dictionary[j, :])

    [eigenvalvector, eigenvecMatrix] = np.linalg.eigh(W)
    inVa = np.diag(np.power(eigenvalvector, -0.5))
    Zexact = E @ eigenvecMatrix @ inVa

    Zexact = Representation.CheckNaNInfComplex(Zexact)
    Zexact = np.real(Zexact)

    BSketch = Representation.fd(Zexact, int(np.ceil(0.5 * d)))

    # eigh returns sorted eigenvalues in ascending order. We reverse this.
    [eigvalues, Q] = np.linalg.eigh(np.matrix.transpose(BSketch) @ BSketch)
    eigvalues = np.real(eigvalues)
    Q = np.real(Q)
    eigvalues = np.flip(eigvalues)
    Q = np.flip(Q)

    VarExplainedCumSum = np.divide(np.cumsum(eigvalues), np.sum(eigvalues))
    k = np.argwhere(VarExplainedCumSum >= f)[0, 0] + 1
    Z_k = Representation.CheckNaNInfComplex(Zexact @ Q[0:d, 0:k])
    return Z_k

