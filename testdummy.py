import numpy as np
from TimeSeries import TimeSeries
from kNN import kNN, kNN_with_pq, kNN_classifier, kNN_classification_precision_test
import Representation
from SINK import SINK, NCC
import heapq
from sklearn.neighbors import NearestNeighbors
import Correlation
import kshape
import time
import csv


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
# #

a = np.loadtxt('elasticmeasures/'+'repTRAIN')
print(a.shape)

# # #returned_labels = kNN_classifier(TRAIN, train_labels, TEST, method = "ED", pq_method="opq", k =10, representation=grail)
# # returned_labels, precision = kNN_classification_precision_test(TRAIN, train_labels, TEST, use_exact_rep=True,
# #                                                    method = "ED", pq_method="opq", k =1, representation=grail)
# #
# # print(precision)
#
# repTRAIN, repTEST = grail.get_rep_train_test(TRAIN, TEST, exact=True)
# exact_neighbors, _, _ = kNN(TRAIN, TEST, method="SINK", k=5, representation=None, gamma = grail.best_gamma)

# for i in range(3):
#     with open('final_results.csv', 'a') as csvfile:
#         csvwriter = csv.writer(csvfile)
#         csvwriter.writerows([1,2,3])


# cnt = 0
# for i in range(test_labels.shape[0]):
#     if test_labels[i] == returned_labels[i]:
#         cnt = cnt + 1
# print(cnt/ test_labels.shape[0])


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

