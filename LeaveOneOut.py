from TimeSeries import TimeSeries
from kNN import kNN, kNN_with_pq, kNN_classifier
import Representation
from sklearn.model_selection import LeaveOneOut
import numpy as np

# TRAIN, train_labels = TimeSeries.load("/tartarus/DATASETS/UCR2018/ACSF1/ACSF1_TRAIN", "UCR")
# TEST, test_labels = TimeSeries.load("/tartarus/DATASETS/UCR2018/ACSF1/ACSF1_TEST", "UCR")

orgTRAIN, orgtrain_labels = TimeSeries.load("ECG200_TRAIN", "UCR")
TEST, test_labels = TimeSeries.load("ECG200_TEST", "UCR")

loo = LeaveOneOut()
split_size = loo.get_n_splits(orgTRAIN)
err_arr = []

for gamma in range(1, 21):
    avg_error = 0
    for train_index, cc_index in loo.split(orgTRAIN):
        TRAIN = orgTRAIN[train_index]
        train_labels = orgtrain_labels[train_index]
        CC = orgTRAIN[cc_index]
        cc_labels = orgtrain_labels[cc_index]

        grail = Representation.GRAIL(gamma = gamma)
        returned_labels = kNN_classifier(TRAIN, train_labels, CC, method="ED", pq_method="opq", k=1, representation=grail)
        if returned_labels[0] != cc_labels[0]:
            avg_error += 1
    avg_error = avg_error / split_size
    err_arr.append(avg_error)
print(err_arr)
print(np.argmin(err_arr) + 1)


# grail = Representation.GRAIL(d = 100)
# returned_labels = kNN_classifier(TRAIN, train_labels, TEST, method = "ED", pq_method="opq", k =1, representation=grail)
#
# cnt = 0
# for i in range(test_labels.shape[0]):
#     if test_labels == returned_labels[i]:
#         cnt = cnt + 1
# print(cnt/ test_labels.shape[0])