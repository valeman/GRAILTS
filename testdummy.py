import numpy as np
from TimeSeries import TimeSeries
from kNN import kNN
import Representation
from SINK import SINK
import heapq
from sklearn.neighbors import NearestNeighbors
import Correlation


TRAIN = TimeSeries.load("BeetleFly_TRAIN", "UCR")
TEST = TimeSeries.load("BeetleFly_TEST", "UCR")


grail = Representation.GRAIL(d = 40)
result = kNN(TRAIN, TEST, method = "ED", representation=grail, k = 5)
print(result[0])
print(result[1])

print("second")
a = kNN(TRAIN, TEST, method = "ED", k = 5)
for i in range(a.shape[0]):
    print()



