import Representation
import numpy as np
from kNN import kNN, kNN_with_pq_NCC, kNN_with_pq_SINK
from time import time
from Causal_inference import generate_synthetic, granger_matrix, check_with_original, granger_causality
import csv
from Causal_Test import load_ts_truemat
from tqdm import tqdm



def findgammaforgranger(TS, trueMat, gamma, lag = 2, neighbor_param = [10,100]):
    #try not setting gamma
    #d = int(min(max(np.ceil(0.4 * 2*n), 20), 100))

    n = TS.shape[0]
    representation = Representation.GRAIL(kernel="SINK", d = 100, gamma = gamma)
    grailMat = np.zeros((n, n))

    result_by_neighbor = {}

    TRAIN_TS, TEST_TS = representation.get_rep_train_test(TS, TS)

    for neighbor_num in neighbor_param:
        if neighbor_num >= n:
            continue
        np.random.seed(0)
        neighbors, _, _ = kNN(TRAIN_TS, TEST_TS, method="ED", k=neighbor_num, representation=None, use_exact_rep=True,
                              pq_method="opq")

        t = time()
        for i in range(n):
           for j in neighbors[i]:
               if j != i:
                  grailMat[i,j] = granger_causality(TS[j], TS[i], lag)
        prunedtime = time() - t

        grail_results = check_with_original(trueMat, grailMat)
        result_by_neighbor[neighbor_num] = {'precision' : grail_results[0], 'recall' : grail_results[1],
                                            'fscore' : grail_results[2], 'runtime' : prunedtime}

    return result_by_neighbor

if __name__ == '__main__':
    csvfile = open('findgammaforgranger_ecgarima.csv', 'w')
    csvwriter = csv.writer(csvfile)

    lags = [2,5,10]

    for lag in tqdm(lags):
        #TS, trueMat = load_ts_truemat(lag, n, ar = 'ar05')
        TS = np.load('ecgarima_200.npy')
        trueMat = np.load('ecgarima_200_truemat.npy')
        for gamma in range(1,20):
            result_by_neighbor = findgammaforgranger(TS, trueMat, lag = lag, gamma = gamma)
            for n_num in result_by_neighbor:
                csvwriter.writerow([TS.shape[0]] + [lag] + [n_num] + list(result_by_neighbor[n_num].values()) + [gamma])
            csvfile.flush()
    csvfile.close()

    # csvfile = open('findgammaforgranger_fixdataset.csv', 'w')
    # csvwriter = csv.writer(csvfile)
    #
    # ns = [200,500]
    # lags = [2,5,10]
    #
    # for n in ns:
    #     for lag in lags:
    #         #TS, trueMat = load_ts_truemat(lag, n, ar = 'ar05')
    #         TS = np.load('ecgarima_200.npy')
    #         trueMat = np.load('ecgarima_200_truemat.npy')
    #         for gamma in range(1,20):
    #             result_by_neighbor = findgammaforgranger(TS, trueMat, n = n, lag = lag, gamma = gamma)
    #             for n_num in result_by_neighbor:
    #                 csvwriter.writerow([n] + [lag] + [n_num] + list(result_by_neighbor[n_num].values()) + [gamma])
    #             csvfile.flush()
    # csvfile.close()