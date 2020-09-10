from TimeSeries import TimeSeries
from kNN import kNN, kNN_with_pq, kNN_classifier
import Representation
from time import time

TRAIN, train_labels = TimeSeries.load("/tartarus/DATASETS/UCR2018/ACSF1/ACSF1_TRAIN", "UCR")
TEST, test_labels = TimeSeries.load("/tartarus/DATASETS/UCR2018/ACSF1/ACSF1_TEST", "UCR")

ls = []
for gamma in range(1,21):
    grail = Representation.GRAIL(d = 100, gamma = gamma)
    returned_labels = kNN_classifier(TRAIN, train_labels, TEST, method = "ED", pq_method="opq", k =1, representation=grail)

    cnt = 0
    for i in range(test_labels.shape[0]):
        if test_labels[i] == returned_labels[i]:
            cnt = cnt + 1
    ls.append(cnt/ test_labels.shape[0])
    print(cnt/ test_labels.shape[0])
print("max: ", max(ls))