from TimeSeries import TimeSeries
from kNN import kNN, kNN_with_pq, kNN_classifier, kNN_classification_precision_test
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

    d = int(min(max(4 * len(np.unique(train_labels)), np.ceil(0.4 * (TRAIN.shape[0] + TEST.shape[0])), 20), 100))
    grail = Representation.GRAIL(d=d)
    repTRAIN, repTEST = grail.get_rep_train_test(TRAIN, TEST, exact=True)

    for NN in [1, 3, 5, 10]:
        exact_neighbors, _, _ = kNN(TRAIN, TEST, method="SINK", k=NN, representation=None, gamma=grail.best_gamma)
        exact_results, pq_results, opq_results_64, opq_results_128 = [1,2,3]

        exact_results[0], exact_results[1], exact_results[2] = kNN_classification_precision_test(exact_neighbors, repTRAIN, train_labels, repTEST, test_labels,
                                        method = "ED", k =NN,pq_method=None, representation=None)

        pq_results[0], pq_results[1], pq_results[2] = kNN_classification_precision_test(exact_neighbors, repTRAIN, train_labels, repTEST, test_labels,
                                        method = "ED", k =NN,pq_method='pq',Ks = 4, M = 16, representation=None)

        opq_results_64[0], opq_results_64[1], opq_results_64[2] = kNN_classification_precision_test(exact_neighbors, repTRAIN, train_labels, repTEST, test_labels,
                                        method = "ED", k =NN,pq_method='opq',Ks = 4, M = 16, representation=None)

        opq_results_128[0], opq_results_128[1], opq_results_128[2] = kNN_classification_precision_test(exact_neighbors, repTRAIN, train_labels, repTEST, test_labels,
                                        method = "ED", k =NN,pq_method='opq',Ks = 4, M = 32, representation=None)


        with open('final_results.csv', 'a') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['Dataset', 'NN', 'GRAIL exact classification precision', 'GRAIL exact precision', 'GRAIL exact time',
                                'PQ exact classification precision', 'PQ exact precision', 'PQ exact time',
                                'OPQ exact classification precision', 'OPQ exact precision', 'OPQ exact time'])
            csvwriter.writerows([dataset, NN] + exact_results + pq_results + opq_results_64 + opq_results_128)

for dataset in datasets:
    test(dataset)
