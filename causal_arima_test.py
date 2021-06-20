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

def ucr_new_method(foldername, dataset, testenv, lag = 5, pval = 1e-9):
    TS = np.load('ecgarima_200.npy')
    trueMat = np.load('ecgarima_200_truemat.npy')
    n = TS.shape[0]

    csvfile = open('{}_arima_causality.csv'.format(dataset), 'w')
    csvwriter = csv.writer(csvfile)
    brute_results, result_by_neighbor = test(TS, trueMat, best_gamma = best_gamma, neighbor_param= [10, 100],lag = lag, pval=pval)
    csvwriter.writerow([n] + [lag] + ['brute'] + list(brute_results.values()))
    for n_num in result_by_neighbor:
        csvwriter.writerow([n] + [lag] + [n_num] + list(result_by_neighbor[n_num].values()))
    csvfile.close()


if __name__ == '__main__':
    arglen = len(sys.argv)
    if arglen > 1:
        n = int(sys.argv[1])
    else:
        n = 200
    lag = 2
    with open('ecgarima{}.npy'.format(n), 'rb') as f:
        TS = np.load(f)
    with open('ecgarima{}_truemat.npy'.format(n), 'rb') as f:
        trueMat = np.load(f)


    csvfile = open('ecgarima{}_new.csv'.format(n), 'w')
    csvwriter = csv.writer(csvfile)
    brute_results, result_by_neighbor = test(TS, trueMat, best_gamma = best_gamma, neighbor_param= [10, 100],lag = lag, pval=1e-30) # used to be different
    csvwriter.writerow([n] + [lag] + ['brute'] + list(brute_results.values()))
    for n_num in result_by_neighbor:
        csvwriter.writerow([n] + [lag] + [n_num] + list(result_by_neighbor[n_num].values()))
    csvfile.close()

