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

def load_dataset(foldername, dataset):
    path1 = "{}_TRAIN".format(dataset)
    path2 = "{}_TEST".format(dataset)
    TRAIN, train_labels = TimeSeries.load(path1, "UCR")
    TEST, test_labels = TimeSeries.load(path2, "UCR")

    return TRAIN, train_labels, TEST, test_labels


if __name__ == '__main__':

    n = 200
    lag = 2
    with open('ecgarima{}.npy'.format(n), 'rb') as f:
        TS = np.load(f)
    with open('ecgarima{}.npy'.format(n), 'rb') as f:
        trueMat = np.load(f)


    csvfile = open('ecgarima{}.csv'.format(n), 'w')
    csvwriter = csv.writer(csvfile)
    brute_results, result_by_neighbor = test(TS, trueMat, best_gamma = best_gamma, neighbor_param= [10],lag = lag, pval=0.001)
    csvwriter.writerow([n] + [lag] + ['brute'] + list(brute_results.values()))
    for n_num in result_by_neighbor:
        csvwriter.writerow([n] + [lag] + [n_num] + list(result_by_neighbor[n_num].values()))
    csvfile.close()

