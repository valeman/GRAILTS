from TimeSeries import TimeSeries
from kNN import kNN, kNN_with_pq, kNN_classifier
import Representation
from time import time
from kshape import matlab_kshape
import numpy as np
import csv


with open("datasets.out", "r+") as dataset_file:
    dataset_names = dataset_file.readlines()
    datasets = [line.rstrip('\n') for line in dataset_names]

results = []

def test(dataset):
    path1 = "/tartarus/DATASETS/UCR2018/" + dataset + "/" + dataset + "_TRAIN"
    path2 = "/tartarus/DATASETS/UCR2018/" + dataset + "/" + dataset + "_TEST"

    TRAIN, train_labels = TimeSeries.load(path1, "UCR")
    TEST, test_labels = TimeSeries.load(path2, "UCR")

    for NN in [1, 3, 5, 10]:

    d = int(min(max(4*len(np.unique(train_labels)), np.ceil(0.4 * (TRAIN.shape[0] + TEST.shape[0])), 20), 100))
    grail = Representation.GRAIL(d = d)
    returned_labels_16 = kNN_classifier(TRAIN, train_labels, TEST, method = "ED", k =1,
                                     pq_method="opq", Ks = 64, M = 16, representation=grail, use_exact_rep=True)

    returned_labels_32 = kNN_classifier(TRAIN, train_labels, TEST, method = "ED", k =1,
                                     pq_method="opq", Ks = 128, M = 32, representation=grail, use_exact_rep=True)

    cnt16 = 0
    cnt32 = 0
    for i in range(test_labels.shape[0]):
        if test_labels[i] == returned_labels_16[i]:
            cnt16 = cnt16 + 1
        if test_labels[i] == returned_labels_32[i]:
            cnt32 = cnt32 + 1
    print("Accuracy for", dataset, " with M = 16: ", cnt16/ test_labels.shape[0])
    print("Accuracy for", dataset, " with M = 32: ", cnt32 / test_labels.shape[0])


    with open('final_results.csv', 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows()

for dataset in datasets:
    test(dataset)
