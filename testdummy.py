import numpy as np
from TimeSeries import TimeSeries
from kNN import kNN, kNN_with_pq, kNN_classifier
import Representation
from SINK import SINK, NCC
import heapq
from sklearn.neighbors import NearestNeighbors
import Correlation
import kshape
import time


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
TRAIN, train_labels = TimeSeries.load("ECG200_TRAIN", "UCR")
TEST, test_labels = TimeSeries.load("ECG200_TEST", "UCR")

grail = Representation.GRAIL(d = 100)
returned_labels = kNN_classifier(TRAIN, train_labels, TEST, method = "ED", pq_method="opq", k =10, representation=grail)

cnt = 0
for i in range(test_labels.shape[0]):
    if test_labels == returned_labels[i]:
        cnt = cnt + 1
print(cnt/ test_labels.shape[0])

#
#
# grail = Representation.GRAIL(d = 100)
# t = time.time()
# print(kNN(TRAIN, TEST, method = "ED", representation=grail, pq_method= "opq", k = 5)[0])
# print(time.time() - t)
# print(kNN(TRAIN, TEST, method = "ED", representation=grail, k = 5)[0])

# #
# # print("second")
# # a = kNN(TRAIN, TEST, method = "SINK", k = 5, gamma = 1)
# # print(a[0])
#
#
# #print(kshape.kshape_with_centroid_initialize(TRAIN, 10))

