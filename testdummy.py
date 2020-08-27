import numpy as np
from TimeSeries import TimeSeries
from kNN import kNN
import Representation
from SINK import SINK
import heapq
from sklearn.neighbors import NearestNeighbors
import Correlation
import kshape

#
# TRAIN = TimeSeries.load("ECG200_TRAIN", "UCR")
# TEST = TimeSeries.load("ECG200_TEST", "UCR")
#
#
# grail = Representation.GRAIL(d = 100)
# result = kNN(TRAIN, TEST, method = "ED", representation=grail, k = 5)
# print(result[0])
#
# print("second")
# a = kNN(TRAIN, TEST, method = "SINK", k = 5, gamma = 1)
# print(a[0])


TRAIN = TimeSeries.load("ECG200_TRAIN", "UCR")
print(kshape.kshape_with_centroid_initialize(TRAIN, 10))




