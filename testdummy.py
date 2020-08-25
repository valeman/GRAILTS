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

# x = np.array([0,0,0])
# y = np.array([0,0,0])
#
# grail = Representation.GRAIL()
# print(kNN(TRAIN, TEST, method = "ED", representation=grail, k = 2))



