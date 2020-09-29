from cpython cimport array
import array

import numpy as np
from numpy.linalg import svd as SVD
import time
import scipy as sp
import random
from FrequentDirections import FrequentDirections
from kshape import matlab_kshape, kshape_with_centroid_initialize
from GRAIL import CheckNaNInfComplex, approx_gte, fd
import exceptions
import gc

cdef extern from "headers.h":
     double kdtw(double* x, int xlen, double* y, int ylen, double sigma)

def GRAIL_rep(X, d, f, r, GV, sigma = None, eigenvecMatrix = None, inVa = None):
    """

    :param X: nxm matrix that contains n time series
    :param d: number of landmark series to extract from kshape
    :param f: scalar to tune the dimensionality k of Z_k
    :param r: parameter for tuning gamma, taken as 20 in the paper
    :param GV: vector of gammas to select the best gamma
    :param fourier_coeff: number of fourier coeffs to keep
    :param e: preserved energy in Fourier domain
    :return: Z_k nxk matrix of low dimensional reduced representation
    """

    n = X.shape[0]
    random.seed(1)
    Dictionary_indices = random.sample(range(n), d)
    Dictionary = X[Dictionary_indices, :]

    print("here")
    if sigma == None:
        print("here")
        [score, sigma] = sigma_select(Dictionary, GV, r)

    E = np.zeros((n, d))

    for i in range(n):
        for j in range(d):
            E[i, j] = compute_kdtw(X[i, :], Dictionary[j, :], sigma)

    if eigenvecMatrix == None and inVa == None:
        W = np.zeros((d, d))
        for i in range(d):
            for j in range(d):
                W[i, j] = compute_kdtw(Dictionary[i, :], Dictionary[j, :], sigma)

        [eigenvalvector, eigenvecMatrix] = np.linalg.eigh(W)
        inVa = np.diag(np.power(eigenvalvector, -0.5))

    Zexact = E @ eigenvecMatrix @ inVa
    Zexact = CheckNaNInfComplex(Zexact)
    Zexact = np.real(Zexact)

    BSketch = fd(Zexact, int(np.ceil(0.5 * d)))

    # eigh returns sorted eigenvalues in ascending order. We reverse this.
    [eigvalues, Q] = np.linalg.eigh(np.matrix.transpose(BSketch) @ BSketch)
    eigvalues = np.real(eigvalues)
    Q = np.real(Q)
    eigvalues = np.flip(eigvalues)
    Q = np.flip(Q)

    VarExplainedCumSum = np.divide(np.cumsum(eigvalues), np.sum(eigvalues))
    k = np.argwhere(approx_gte(VarExplainedCumSum, f))[0, 0] + 1
    Z_k = CheckNaNInfComplex(Zexact @ Q[0:d, 0:k])
    return Z_k, Zexact

def compute_kdtw(x,y, sigma):
    #print("inside kdtw")
    cdef array.array xc = array.array('d', x)
    cdef array.array yc = array.array('d', y)

    cdef double[:] xa = xc
    cdef double[:] ya  = yc

    return kdtw(&xa[0], len(x), &ya[0], len(y), sigma)

def sigma_select(Dictionary, GV, r):
    """
    Parameter Tuning function. Tunes the parameters for GRAIL_kdtw
    :param Dictionary: Dictionary to summarize the dataset.
    :param GV: A vector of sigma values to choose from.
    :param r: The number of top eigenvalues to be considered. This is 20 in the paper.
    :return: the tuned parameter sigma and its score
    """
    d = Dictionary.shape[0]
    GVar = np.zeros(len(GV))
    var_top_r = np.zeros(len(GV))
    score = np.zeros(len(GV))
    for i in range(len(GV)):
        print(i)
        sigma = GV[i]
        W = np.zeros((d, d))
        for j in range(d):
            for k in range(d):
                W[j, k] = compute_kdtw(Dictionary[j, :], Dictionary[k, :], sigma)
        GVar[i] = np.var(W)
        [eigenvalvector, eigenvecMatrix] = np.linalg.eigh(W)
        eigenvalvector = np.flip(eigenvalvector)
        eigenvecMatrix = np.flip(eigenvecMatrix)
        var_top_r[i] = np.sum(eigenvalvector[0:r]) / np.sum(eigenvalvector)

    score = GVar * var_top_r
    best_sigma_index = np.argmax(score)
    best_sigma = GV[best_sigma_index]
    best_score = score[best_sigma_index]
    print("sigma = ", best_sigma)
    return [best_score, best_sigma]
