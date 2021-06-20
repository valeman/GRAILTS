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
from Causal_inference import generate_synthetic, preprocess_dataset

best_gamma = 5

def load_dataset(foldername, dataset, testenv):
    if testenv == 'windows':
        path1 = "{}_TRAIN".format(foldername, dataset)
        path2 = "{}_TEST".format(foldername, dataset)
    else:
        path1 = "{}/{}_TRAIN".format(foldername,dataset)
        path2 = "{}/{}_TEST".format(foldername, dataset)
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

#weight was 2 pval was 0.0001
def prepare_and_test_ucr(foldername, dataset, method = 'standard', lag = 2, weight = 2, pval = 0.0001, testenv = 'windows'):
    TRAIN, train_labels, TEST, test_labels = load_dataset(foldername, dataset, testenv='windows')
    TS = np.vstack((TRAIN, TEST))
    n = TS.shape[0]
    TS = preprocess_dataset(TS)
    TS, trueMat = add_causality_dataset(TS, lag=lag, weight=weight, method = method)

    csvfile = open('{}_direct_causality_{}.csv'.format(dataset, method), 'w')
    csvwriter = csv.writer(csvfile)
    brute_results, result_by_neighbor = test(TS, trueMat, best_gamma = best_gamma, neighbor_param= [10, 100],lag = lag, pval=pval)
    csvwriter.writerow([n] + [lag] + ['brute'] + list(brute_results.values()))
    for n_num in result_by_neighbor:
        csvwriter.writerow([n] + [lag] + [n_num] + list(result_by_neighbor[n_num].values()))
    csvfile.close()





if __name__ == '__main__':

    dataset = sys.argv[2]
    arglen = len(sys.argv)
    test_env = sys.argv[1]
    if test_env == "windows":
        foldername = '.'
    elif test_env == 'chaos':
        foldername = '/tartarus/DATASETS/UCR2018/{}'.format(dataset)
    else:
        foldername = '.'

    prepare_and_test_ucr(foldername, dataset, method = 'standard')






    # TRAIN = np.zeros((50, 128))  # generate_synthetic(200, m = 128, lag = 2, ar = [1, 1],ma = [0.01])
    # TEST = np.zeros((50, 128))
    #
    # for i in range(50):
    #     TRAIN[i, 0] = np.random.normal(0, 1)
    #     for j in range(1, 128):
    #         TRAIN[i, j] = TRAIN[i, j - 1] + 1
    #         TEST[i, j] = TRAIN[i, j]
    # TS = np.vstack((TRAIN, TEST))
    # print(TS.shape)
    # n,m = TS.shape
    # lag = 2
    # best_gamma = 5
    # TS, trueMat = add_causality_dataset(TS, lag=lag )
    # representation = Representation.GRAIL(kernel="SINK", d=100, gamma=best_gamma)
    #
    # TRAIN_TS, TEST_TS = representation.get_rep_train_test(TS, TS)
    # #np.random.seed(0)
    # neighbors, _, _ = kNN(TRAIN_TS, TEST_TS, method="ED", k=10, representation=None, use_exact_rep=True,
    #                       pq_method=None)
    #
    # exact_neighbors, _, _ = kNN(TS, TS, method="SINK", k=10, representation=None, gamma_val=best_gamma)
    #
    # knn_map_accuracy = MAP(exact_neighbors, neighbors)
    # knn_recall_accuracy = avg_recall_measure(exact_neighbors, neighbors)
    #
    # print(knn_recall_accuracy, knn_map_accuracy)


