import grail_kdtw
from TimeSeries import TimeSeries



TRAIN, train_labels = TimeSeries.load("ECG200_TRAIN", "UCR")

print(grail_kdtw.GRAIL_rep(TRAIN, 100, .99))
