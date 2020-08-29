import numpy as np
from TimeSeries import TimeSeries
from kNN import kNN
import Representation
from SINK import SINK, NCC
import heapq
from sklearn.neighbors import NearestNeighbors
import Correlation
import kshape
import math
import random
import hello

# def NCC_new(x, y):
#     length = len(x);
#     fftlen = 2 ** math.ceil(math.log2(abs(2 * length - 1)));
#     r = np.fft.ifft(np.multiply(np.fft.fft(x, fftlen), np.conj(np.fft.fft(y, fftlen))))
#
#     lenr = len(r) - 1;
#
#     result = np.append(r[lenr - length + 2:lenr + 1], r[0:length])
#
#     return result;
#
#
# def NCCc(x, y):
#     length = len(x);
#     fftlen = 2 ** math.ceil(math.log2(abs(2 * length - 1)));
#     r = np.fft.ifft(np.multiply(np.fft.fft(x, fftlen), np.conj(np.fft.fft(y, fftlen))))
#
#     lenr = len(r) - 1;
#
#     result = np.append(r[lenr - length + 2:lenr + 1], r[0:length])
#
#     return np.divide(np.real(result), np.linalg.norm(x) * np.linalg.norm(y))
#
# for i in range(10):
#     x = np.random.rand(5)
#     y = np.random.rand(5)
#     print("x = ", x)
#     print("y = ", y)
#     print(NCC(x,y))
#     print(NCCc(x,y))
#     print(NCC(x,y) == NCCc(x,y))
#
#
# #
# #TRAIN = TimeSeries.load("ECG200_TRAIN", "UCR")
# # TEST = TimeSeries.load("ECG200_TEST", "UCR")
# #
# #
# # grail = Representation.GRAIL(d = 100)
# # result = kNN(TRAIN, TEST, method = "ED", representation=grail, k = 5)
# # print(result[0])
# #
# # print("second")
# # a = kNN(TRAIN, TEST, method = "SINK", k = 5, gamma = 1)
# # print(a[0])
#
#
# #print(kshape.kshape_with_centroid_initialize(TRAIN, 10))

