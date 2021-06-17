from TimeSeries import TimeSeries
import pmdarima as pm
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

    dataset = 'ECG200'
    test_env = sys.argv[1]
    if test_env == "windows":
        foldername = '.'
    elif test_env == 'chaos':
        foldername = '/tartarus/DATASETS/{}'.format(dataset)

    TRAIN, train_labels, TEST, test_labels = load_dataset(foldername, dataset)
    TS = np.vstack((TRAIN, TEST))
    print(TS.shape)
    n,m = TS.shape
    lag = 5
    best_gamma = 5
    TRAIN, trueMat = add_causality_dataset(TS, lag=lag )

    csvfile = open('causal_ucr.csv', 'w')
    csvwriter = csv.writer(csvfile)
    brute_results, result_by_neighbor = test(TS, trueMat, best_gamma = best_gamma, neighbor_param= [10, 100], n = n, lag = lag, m = m)
    csvwriter.writerow([n] + [lag] + ['brute'] + list(brute_results.values()))
    for n_num in result_by_neighbor:
        csvwriter.writerow([n] + [lag] + [n_num] + list(result_by_neighbor[n_num].values()))
    csvfile.close()

