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

cdef extern from "headers.h":
	double kdtw(double* x, int xlen, double* y, int ylen, double sigma)

def GRAIL_rep(X, d, f, r, sigma = 1, eigenvecMatrix = None, inVa = None, initialization_method = "partition"):
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
    if initialization_method == "partition":
        [mem, Dictionary] = matlab_kshape(X,d)
    elif initialization_method == "centroid_uniform":
        [mem, Dictionary] = kshape_with_centroid_initialize(X, d, is_pp=False)
    elif initialization_method == "k-shape++":
        [mem, Dictionary] = kshape_with_centroid_initialize(X, d, is_pp=True)
    else:
        raise exceptions.InitializationMethodNotFound

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
    cdef array.array xc = array.array('d', x)
    cdef array.array yc = array.array('d', y)

    cdef double[:] xa = xc
    cdef double[:] ya  = yc

    return kdtw(&xa[0], len(x), &ya[0], len(y), sigma)