import numpy as np
import Correlation
import SINK
from TimeSeries import TimeSeries
from kNN import kNN
from sklearn.neighbors import NearestNeighbors
import Representation


TRAIN = TimeSeries.load("BeetleFly_TRAIN", "UCR")
TEST = TimeSeries.load("BeetleFly_TEST", "UCR")

grail = Representation.GRAIL()
print(kNN(TRAIN, TEST, method = "ED", representation=grail, k = 2))