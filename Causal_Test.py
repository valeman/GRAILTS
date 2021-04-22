import Representation
import numpy as np
from kNN import kNN, kNN_with_pq_NCC, kNN_with_pq_SINK
from time import time
from Causal_inference import generate_synthetic, granger_matrix, check_with_original, granger_causality
import csv


def test(n = 100, lag = 2, m = 128):
    #try not setting gamma
    representation = Representation.GRAIL(kernel="SINK", d = 50, gamma = 1)

    TS, trueMat = generate_synthetic(n, m = m, lag = lag, ar = [1, 0.5]) #try changing ar

    grailMat = np.zeros((n, n))
    controlMat = np.zeros((n, n))
    t = time()
    bruteMat = granger_matrix(TS, lag)
    bruteTime = time() - t


    neighbor_param = [2, 5, 10, 100]

    result_by_neighbor = {}

    TRAIN_TS, TEST_TS = representation.get_rep_train_test(TS, TS)
    brute_res = check_with_original(trueMat, bruteMat)
    brute_results = {'precision' : brute_res[0], 'recall' : brute_res[1],
                                            'fscore' : brute_res[2], 'runtime' : bruteTime}

    for neighbor_num in neighbor_param:
        if neighbor_num >= n:
            continue
    #    neighbors, _, _ = kNN(TRAIN_TS, TEST_TS, method = "ED", k = neighbor_num, representation=None, use_exact_rep = True, pq_method = "pq", M = 16)
        np.random.seed(0)
        neighbors, _, _ = kNN(TRAIN_TS, TEST_TS, method="ED", k=neighbor_num, representation=None, use_exact_rep=True,
                              pq_method="opq")

        # control_neighbors = np.array([np.random.choice(n, neighbor_num, replace= False) for i in range(n)])
        # print(control_neighbors)
        #neighbors, _, _ = kNN_with_pq_NCC(TS, TS, k = neighbor_num, use_exact_rep=True, Ks=4, M = 16)
        #neighbors, _, _ = kNN(TS, TS, method = "NCC", k = neighbor_num, representation=None, use_exact_rep = True, pq_method = None, M = 16)

        #neighbors, _, _ = kNN_with_pq_SINK(TS, TS, k = neighbor_num, use_exact_rep=True, Ks=4, M = 16, gamma_val = 1)

        t = time()
        for i in range(n):
           for j in neighbors[i]:
               if j != i:
                  grailMat[i,j] = granger_causality(TS[j], TS[i], lag)
        prunedtime = time() - t

        grail_results = check_with_original(trueMat, grailMat)

        #print("k number for kNN:", neighbor_num)
        #print("Pruned VL time: ", prunedtime)
        #print("prec, rec, and F1 for GRAIL: ", compare_results)
        # print("prec, rec, and F1 for control: ", check_with_original(trueMat, controlMat))
        #print("prec, rec, and F1 for Brute Granger Causality:", brute_results)
        result_by_neighbor[neighbor_num] = {'precision' : grail_results[0], 'recall' : grail_results[1],
                                            'fscore' : grail_results[2], 'runtime' : prunedtime}

    #print("time for VL:", bruteTime )
    return brute_results, result_by_neighbor



def compare_with_standard():
    csvfile = open('causal_inference.csv', 'w')
    csvwriter = csv.writer(csvfile)
    m =128
    for n in range(100, 5000, 100):
        for lag in range(1, 5):
            print(n, lag)
            brute_results, result_by_neighbor, grail_results = test(n,lag,m)
            csvwriter.writerow([n] + [lag] + ['brute'] + list(brute_results.values()))
            for n_num in result_by_neighbor:
                csvwriter.writerow([n] + [lag] + [n_num] + list(result_by_neighbor[n_num].values()))
    csvfile.close()


compare_with_standard()









