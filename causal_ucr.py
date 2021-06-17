from TimeSeries import TimeSeries
#import pmdarima as pm
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from Causal_inference import add_causality_dataset
import sys
import csv
from Causal_Test import test

import Representation
from kNN import kNN, kNN_with_pq_NCC, kNN_with_pq_SINK, MAP, avg_recall_measure
import numpy as np
# from Causal_inference import generate_synthetic


def load_dataset(foldername, dataset):
    path1 = "{}_TRAIN".format(dataset)
    path2 = "{}_TEST".format(dataset)
    TRAIN, train_labels = TimeSeries.load(path1, "UCR")
    TEST, test_labels = TimeSeries.load(path2, "UCR")

    return TRAIN, train_labels, TEST, test_labels

# def fit_arima(train):
#     print(train.shape)
#     model = pm.auto_arima(train, seasonal=False)
#     newstuff = model.arima_res_.simulate(train.shape[0])
#     x = np.arange(newstuff.shape[0])
#     plt.plot(x, newstuff, c='blue')
#     plt.show()
#     order = model.order
#     model = ARIMA(train, order = order)
#     #model.simulate()
#     print(model.param_terms)



if __name__ == '__main__':

    # dataset = 'ECG200'
    # arglen = len(sys.argv)
    # test_env = None
    # if arglen == 2:
    #     test_env = sys.argv[1]
    # if test_env == "windows":
    #     foldername = '.'
    # elif test_env == 'chaos':
    #     foldername = '/tartarus/DATASETS/{}'.format(dataset)
    # else:
    #     foldername = '.'
    #
    # TRAIN, train_labels, TEST, test_labels = load_dataset(foldername, dataset)
    # TS = np.vstack((TRAIN, TEST))
    # print(TS.shape)
    # n,m = TS.shape
    # lag = 2
    # best_gamma = 5
    # TS, trueMat = add_causality_dataset(TS, lag=lag )

    TRAIN = np.zeros((50, 128))  # generate_synthetic(200, m = 128, lag = 2, ar = [1, 1],ma = [0.01])
    TEST = np.zeros((50, 128))

    for i in range(50):
        TRAIN[i, 0] = np.random.normal(0, 1)
        for j in range(1, 128):
            TRAIN[i, j] = TRAIN[i, j - 1] + 1
            TEST[i, j] = TRAIN[i, j]
    TS = np.vstack((TRAIN, TEST))
    print(TS.shape)
    n,m = TS.shape
    lag = 2
    best_gamma = 5
    TS, trueMat = add_causality_dataset(TS, lag=lag )
    representation = Representation.GRAIL(kernel="SINK", d=100, gamma=best_gamma)

    TRAIN_TS, TEST_TS = representation.get_rep_train_test(TS, TS)

    neighbors, _, _ = kNN(TRAIN_TS, TEST_TS, method="ED", k=10, representation=None, use_exact_rep=True,
                          pq_method=None)

    exact_neighbors, _, _ = kNN(TS, TS, method="SINK", k=10, representation=None, gamma_val=best_gamma)

    knn_map_accuracy = MAP(exact_neighbors, neighbors)
    knn_recall_accuracy = avg_recall_measure(exact_neighbors, neighbors)

    print(knn_recall_accuracy, knn_map_accuracy)

    # print(TS)
    # csvfile = open('causal_ucr1.csv', 'w')
    # csvwriter = csv.writer(csvfile)
    # brute_results, result_by_neighbor = test(TS, trueMat, best_gamma = best_gamma, neighbor_param= [10],lag = lag)
    # csvwriter.writerow([n] + [lag] + ['brute'] + list(brute_results.values()))
    # for n_num in result_by_neighbor:
    #     csvwriter.writerow([n] + [lag] + [n_num] + list(result_by_neighbor[n_num].values()))
    # csvfile.close()

