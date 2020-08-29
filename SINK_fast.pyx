import numpy as np
cimport numpy as np
import math
from libcpp.vector cimport vector
from cpython cimport array
import array

cdef extern from "headers.h":
    double sink(double * x, int xlen, double* y, int ylen, double gamma)



cdef extern from "headers.h":
	void hello(double *v, int size)

cdef array.array a = array.array('d', [1, 2, 3])
cdef double[:] ca = a

hello(&ca[0], 3)


# # Helper function for SINK
# def sumExpNCC(x, y, gamma, k=-1, e=-1):
#     check_arguments(k, e)
#     return np.sum(np.exp(gamma * NCC(x, y, k, e)))
#
#
# def NCC(x, y, k=-1, e=-1):
#     '''
#     Normalized cross correlation
#     :param x: time series
#     :param y: time series
#     :param k: optional parameter for compression
#     :param e: preserved energy in the Fourier Domain
#     :return: cross correlation sequence
#     '''
#
#     check_arguments(k, e)
#     length = max(len(x), len(y))
#
#     fftlength = int(pow(2, nextPow2(2 * length - 1)))
#
#     if k == -1 and e == -1:
#         fftx = np.fft.fft(x, n=fftlength)
#         ffty = np.fft.fft(y, n=fftlength)
#     elif k == -1 and e != -1:
#         fftx = preserved_energy(x, e, fftlength)
#         ffty = preserved_energy(x, e, fftlength)
#     else:
#         fftx = leadingFourier(np.fft.fft(x, n=fftlength), k)
#         ffty = leadingFourier(np.fft.fft(y, n=fftlength), k)
#     r = np.fft.ifft(np.multiply(fftx, np.conj(ffty)))
#
#     end = len(r) - 1
#     r = np.concatenate((r[(end - length + 2):len(r)], r[0: length]))
#
#     result = np.divide(np.real(r), (np.linalg.norm(x) * np.linalg.norm(y)))
#     return np.nan_to_num(result)
#
#
# # Helper function for NCC
# def nextPow2(x):
#     return np.ceil(np.log2(abs(x)))
#
#
# def preserved_energy(x, e, fftlength):
#     fftx = np.fft.fft(x, n=fftlength)
#     norm_cumsum = np.divide(np.cumsum(np.power(np.abs(fftx), 2)), np.sum(np.power(np.abs(fftx), 2)))
#     k = np.argwhere(norm_cumsum >= e / 2)[0, 0] + 1
#     return leadingFourier(x, k)
#
#
# # Helper function for NCC
# def leadingFourier(x, k):
#     m = int(np.floor(len(x) / 2) + 1)
#     x[k: (m - 1 + m - k)] = 0
#     return x
#
#
# def check_arguments(k, e):
#     if k != -1 and e != -1:
#         raise ValueError(
#             "Preserved energy and number of Fourier coefficients should not be specified at the same time.")
