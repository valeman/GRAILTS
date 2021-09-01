from TimeSeries import TimeSeries
from kNN import kNN, kNN_with_pq, kNN_classifier
import Representation
from time import time
from kshape import matlab_kshape
import numpy as np


datasets = ["ACSF1", "Adiac", "AllGestureWiimoteX", "AllGestureWiimoteY", "AllGestureWiimoteZ",
           "ArrowHead", "Beef", "BeetleFly", "BirdChicken", "BME", "Car", "CBF", "Chinatown",
            "ChlorineConcentration", "CinCECGTorso", "Coffee", "Computers", "CricketX", "CricketY",
            "CricketZ", "Crop", "DiatomSizeReduction", "DistalPhalanxOutlineAgeGroup", "DistalPhalanxOutlineCorrect",
            "DistalPhalanxTW", "DodgerLoopDay", "DodgerLoopGame", "DodgerLoopWeekend"]


def test(dataset):
    path1 = "/tartarus/DATASETS/UCR2018/" + dataset + "/" + dataset + "_TRAIN"
    path2 = "/tartarus/DATASETS/UCR2018/" + dataset + "/" + dataset + "_TEST"

    TRAIN, train_labels = TimeSeries.load(path1, "UCR")
    TEST, test_labels = TimeSeries.load(path2, "UCR")

    d = int(min(max(4*len(np.unique(train_labels)), np.ceil(0.4 * (TRAIN.shape[0] + TEST.shape[0])), 20), 100))
    grail = Representation.GRAIL(d = d)
    returned_labels = kNN_classifier(TRAIN, train_labels, TEST, method = "ED", k =1, representation=grail, use_exact_rep=True)

    cnt = 0
    for i in range(test_labels.shape[0]):
        if test_labels[i] == returned_labels[i]:
            cnt = cnt + 1
    print("Accuracy for", dataset, ": ", cnt/ test_labels.shape[0])


for dataset in datasets:
    test(dataset)

