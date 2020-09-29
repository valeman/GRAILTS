import grail_kdtw
from TimeSeries import TimeSeries
import numpy as np


TRAIN, train_labels = TimeSeries.load("ECG200_TRAIN", "UCR")

print(grail_kdtw.GRAIL_rep(TRAIN, 50, .99, 10, GV = [np.power(float(2),-x) for x in range(16)]))
