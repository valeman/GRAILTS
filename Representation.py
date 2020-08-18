from SINK import SINK
import numpy as np
from numpy.linalg import svd as SVD
import time
import scipy as sp
import random
from FrequentDirections import FrequentDirections
from kshape import matlab_kshape


def GRAIL(X, d, f, r, GV, e = -1):
    """

    :param X: nxm matrix that contains n time series
    :param d: number of landmark series to extract from kshape
    :param f: scalar to tune the dimensionality k of Z_k
    :param r: parameter for tuning gamma, taken as 20 in the paper
    :param GV: vector of gammas to select the best gamma
    :param e: preserved energy in Fourier domain
    :return: Z_k nxk matrix of low dimensional reduced representation
    """
    n = X.shape[0]
    [mem, Dictionary] = matlab_kshape(X,d)
    [score, gamma] = gamma_select(Dictionary, GV, r)
    W = np.zeros((d, d))

    for i in range(d):
        for j in range(d):
            W[i, j] = SINK(Dictionary[i, :], Dictionary[j, :], gamma, e)

    E = np.zeros((n, d))

    for i in range(n):
        for j in range(d):
            E[i, j] = SINK(X[i, :], Dictionary[j, :], gamma, e)

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
    k = np.argwhere(VarExplainedCumSum >= f)[0, 0] + 1
    Z_k = CheckNaNInfComplex(Zexact @ Q[0:d, 0:k])
    return Z_k

def repLearn(X, Dictionary, gamma, k=-1):
    """
    :param K: Optional. Number of Fourier coefficients to keep when computing the SINK kernel
    :param X: A matrix containing time series in its rows.
    :param Dictionary: A Dictionary of time series to summarize the underlying dataset. Provided by kshape.
    :param gamma: SINK kernel parameter
    :return: Approximated Kernel Matrix
    """
    t = time.time()
    W = np.zeros((Dictionary.shape[0], Dictionary.shape[0]))
    DistComp = 0

    for i in range(Dictionary.shape[0]):
        for j in range(Dictionary.shape[0]):
            W[i, j] = SINK(Dictionary[i, :], Dictionary[j, :], gamma, k)
            DistComp = DistComp + 1

    E = np.zeros((X.shape[0], Dictionary.shape[0]))

    for i in range(X.shape[0]):
        for j in range(Dictionary.shape[0]):
            E[i, j] = SINK(X[i, :], Dictionary[j, :], gamma, k)
            DistComp = DistComp + 1

    [eigenvalvector, eigenvecMatrix] = np.linalg.eigh(W)
    inVa = np.diag(np.power(eigenvalvector, -0.5))
    Zexact = E @ eigenvecMatrix @ inVa
    RuntimeNystrom = time.time() - t

    Zexact = CheckNaNInfComplex(Zexact)
    Zexact = np.real(Zexact)

    t = time.time()
    BSketch = fd(Zexact, int(np.ceil(0.5 * Zexact.shape[1])))

    # eigh returns sorted eigenvalues in ascending order. We reverse this.
    [eigvalues, Q] = np.linalg.eigh(np.matrix.transpose(BSketch) @ BSketch)
    eigvalues = np.real(eigvalues)
    Q = np.real(Q)
    eigvalues = np.flip(eigvalues)
    Q = np.flip(Q)
    RuntimeFD = time.time() - t

    VarExplainedCumSum = np.divide(np.cumsum(eigvalues), np.sum(eigvalues))

    DimFor99 = np.argwhere(VarExplainedCumSum >= 0.99)[0, 0] + 1
    DimFor98 = np.argwhere(VarExplainedCumSum >= 0.98)[0, 0] + 1
    DimFor97 = np.argwhere(VarExplainedCumSum >= 0.97)[0, 0] + 1
    DimFor95 = np.argwhere(VarExplainedCumSum >= 0.95)[0, 0] + 1
    DimFor90 = np.argwhere(VarExplainedCumSum >= 0.90)[0, 0] + 1
    DimFor85 = np.argwhere(VarExplainedCumSum >= 0.85)[0, 0] + 1
    DimFor80 = np.argwhere(VarExplainedCumSum >= 0.80)[0, 0] + 1

    Ztop5 = CheckNaNInfComplex(Zexact @ Q[:, 0:5])
    Ztop10 = CheckNaNInfComplex(Zexact @ Q[:, 0:10])
    Ztop20 = CheckNaNInfComplex(Zexact @ Q[:, 0:20])

    Z99per = CheckNaNInfComplex(Zexact @ Q[:, 0:DimFor99])
    Z98per = CheckNaNInfComplex(Zexact @ Q[:, 0:DimFor98])
    Z97per = CheckNaNInfComplex(Zexact @ Q[:, 0:DimFor97])
    Z95per = CheckNaNInfComplex(Zexact @ Q[:, 0:DimFor95])
    Z90per = CheckNaNInfComplex(Zexact @ Q[:, 0:DimFor90])
    Z85per = CheckNaNInfComplex(Zexact @ Q[:, 0:DimFor85])
    Z80per = CheckNaNInfComplex(Zexact @ Q[:, 0:DimFor80])

    print(time.time() - t)

    return [Zexact, Ztop5, Ztop10, Ztop20, Z99per, Z98per, Z97per, Z95per, Z90per, Z85per, Z80per, DistComp,
            RuntimeNystrom, RuntimeFD]


def CheckNaNInfComplex(Z):
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            if np.isnan(Z[i, j]) or np.isinf(Z[i, j]) or (not np.isreal(Z[i, j])):
                Z[i, j] = 0

    return Z


def repLearnKM(KM):
    t = time.time()
    [eigenvalvector, Q] = np.linalg.eigh(KM)
    eigenvalvector = np.flip(eigenvalvector)
    Q = np.flip(Q)

    VarExplainedCumSum = np.divide(np.cumsum(eigenvalvector), np.sum(eigenvalvector))

    DimFor99 = np.argwhere(VarExplainedCumSum >= 0.99)[0, 0] + 1
    DimFor98 = np.argwhere(VarExplainedCumSum >= 0.98)[0, 0] + 1
    DimFor97 = np.argwhere(VarExplainedCumSum >= 0.97)[0, 0] + 1
    DimFor95 = np.argwhere(VarExplainedCumSum >= 0.95)[0, 0] + 1
    DimFor90 = np.argwhere(VarExplainedCumSum >= 0.90)[0, 0] + 1
    DimFor85 = np.argwhere(VarExplainedCumSum >= 0.85)[0, 0] + 1
    DimFor80 = np.argwhere(VarExplainedCumSum >= 0.80)[0, 0] + 1

    RepLearnTime = time.time() - t

    Z99per = CheckNaNInfComplex(Q[:, 0: DimFor99] @ np.sqrt(np.diag(eigenvalvector[0: DimFor99])))
    Z98per = CheckNaNInfComplex(Q[:, 0: DimFor98] @ np.sqrt(np.diag(eigenvalvector[0: DimFor98])))
    Z97per = CheckNaNInfComplex(Q[:, 0: DimFor97] @ np.sqrt(np.diag(eigenvalvector[0: DimFor97])))
    Z95per = CheckNaNInfComplex(Q[:, 0: DimFor95] @ np.sqrt(np.diag(eigenvalvector[0: DimFor95])))
    Z90per = CheckNaNInfComplex(Q[:, 0: DimFor90] @ np.sqrt(np.diag(eigenvalvector[0: DimFor90])))
    Z85per = CheckNaNInfComplex(Q[:, 0: DimFor85] @ np.sqrt(np.diag(eigenvalvector[0: DimFor85])))
    Z80per = CheckNaNInfComplex(Q[:, 0: DimFor80] @ np.sqrt(np.diag(eigenvalvector[0: DimFor80])))

    Ztop20 = CheckNaNInfComplex(Q[:, 0: 20] @ np.sqrt(np.diag(eigenvalvector[0: 20])))
    Ztop10 = CheckNaNInfComplex(Q[:, 0: 10] @ np.sqrt(np.diag(eigenvalvector[0: 10])))
    Ztop5 = CheckNaNInfComplex(Q[:, 0: 5] @ np.sqrt(np.diag(eigenvalvector[0: 5])))
    return [Z99per, Z98per, Z97per, Z95per, Z90per, Z85per, Z80per, Ztop20, Ztop10, Ztop5, RepLearnTime]


# python version of matlab fd
# def fd(A, ell):
#     rows = A.shape[0]
#     d = A.shape[1]
#     m = 2 * ell
#
#     if rows <= m:
#         [U, s, Vt] = np.linalg.svd(A, full_matrices=False)
#         S = sp.linalg.diagsvd(s, A.shape[0], A.shape[1])
#         vout = Vt[:, 0]
#         sketch = S @ np.transpose(Vt)
#         return
#
#     sketch = np.zeros(m, d, dtype=complex)
#     nextZeroRow = 0
#
#     for i in range(rows):
#         vector = A[i,:]     # append row
#
#         if (nextZeroRow > m): # rotate
#             [U, s, Vt] = np.linalg.svd(sketch, full_matrices=False) # economy SVD: sketch = U S V
#             S = sp.linalg.diagsvd(s, A.shape[0], A.shape[1])
#             vout = Vt
#             length = len(s)
#             if length >= ell: # if rank is greater than ell, then shrink
#                 sShrunk = np.sqrt(np.power(s[0:ell],2) - np.power(s(ell),2))
#                 sketch[1: ell,:] = diag(sShrunk) * np.transpose(Vt[:, 0: ell])
#                 sketch[(ell + 1): end,:] = 0
#                 nextZeroRow = ell + 1 # maintain invariant that row l is zeros
#
#             else: # otherwise fewer than ell non-zero rows
#                 sketch[1: length,:] = S @ np.transpose(Vt[:, 0:length])
#                 sketch[(length + 1): end,:] = 0
#                 nextZeroRow = length + 1
#
#         sketch[nextZeroRow,:] = vector # append row
#         nextZeroRow = nextZeroRow + 1
#     sketch = sketch[0:ell,:]
#     return [sketch, vout]


def gamma_select(Dictionary, GV, r, k=-1):
    """
    Parameter Tuning function. Tunes the parameters for GRAIL
    This function does not work at the moment for gamma values that don't correspond
    to a range(n)
    :param Dictionary: Dictionary to summarize the dataset. Provided by KShape
    :param GV: A vector of Gamma values to choose from.
    :param k: Number of Fourier coeffs to keep when computing the SINK function.
    :param r: The number of top eigenvalues to be considered. This is 20 in the paper.
    :return: the tuned parameter gamma and its score
    """
    d = Dictionary.shape[0]
    GVar = np.zeros(len(GV))
    var_top_r = np.zeros(len(GV))
    score = np.zeros(len(GV))
    for gamma in GV:
        W = np.zeros((d, d))
        for i in range(d):
            for j in range(d):
                W[i, j] = SINK(Dictionary[i, :], Dictionary[j, :], gamma, k)
        GVar[gamma] = np.var(W)
        [eigenvalvector, eigenvecMatrix] = np.linalg.eigh(W)
        eigenvalvector = np.flip(eigenvalvector)
        eigenvecMatrix = np.flip(eigenvecMatrix)
        var_top_r[gamma] = np.cumsum(eigenvalvector[0:r]) / np.sum(eigenvalvector)

    score = GVar * var_top_r
    best_gamma = np.argmax(score)
    best_score = score[best_gamma]

    return [best_score, best_gamma]


# Frequent directions helper function returns sketch matrix of size (ell x d)
def fd(A, ell):
    """
    Returns a sketch matrix of size ell x A.shape[1]
    :param A: Matrix to be sketched
    :param ell:
    :return:
    """
    d = A.shape[1]
    sketcher = FrequentDirections(d, ell)
    for i in range(A.shape[0]):
        sketcher.append(A[i, :])
    sketch = sketcher.get()
    return sketch


x = np.array([[0.33169908, 0.79247418, 0.20331952, 0.34972748, 0.42525741, 0.33914221],
              [0.44189778, 0.23751129, 0.10359036, 0.55644621, 0.3216573, 0.27341989],
              [0.85078943, 0.9351495, 0.92466953, 0.20721836, 0.17450002, 0.36438568],
              [0.35611616, 0.60584238, 0.35174043, 0.1855819, 0.57233715, 0.51038871],
              [0.90560198, 0.07615764, 0.12331417, 0.49759322, 0.37303985, 0.07340795],
              [0.26056805, 0.74227914, 0.80298907, 0.2873181, 0.44221004, 0.5717879]])

print(repLearnKM(x))
