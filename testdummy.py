import numpy as np
from TimeSeries import TimeSeries
from kNN import kNN
import Representation


TRAIN = TimeSeries.load("BeetleFly_TRAIN", "UCR")
TEST = TimeSeries.load("BeetleFly_TEST", "UCR")

grail = Representation.GRAIL()
print(kNN(TRAIN, TEST, method = "ED", representation=grail, k = 2))
