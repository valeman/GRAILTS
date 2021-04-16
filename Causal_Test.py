import Representation
import numpy as np
from kNN import kNN, kNN_with_pq_NCC, kNN_with_pq_SINK
from time import time
from Causal_inference import generate_synthetic, granger_matrix, check_with_original, granger_causality

n = 100
lag = 2
#can try kdtw here
representation = Representation.GRAIL(kernel="SINK", d = 50, gamma = 1)

TS, trueMat = generate_synthetic(n, m = 128, lag = lag, ar = [1, 0.5]) #try changing ar

grailMat = np.zeros((n, n))
controlMat = np.zeros((n, n))
t = time()
bruteMat = granger_matrix(TS, lag)
bruteTime = time() - t


neighbor_param = [2, 5, 10]

TRAIN_TS, TEST_TS = representation.get_rep_train_test(TS, TS)

for neighbor_num in neighbor_param:
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

    # for i in range(n):
    #     for j in control_neighbors[i]:
    #         if j != i:
    #             controlMat[i, j] = granger_causality(TS[j], TS[i], lag)

    print("k number for kNN:", neighbor_num)
    print("Pruned VL time: ", prunedtime)
    print("prec, rec, and F1 for GRAIL: ", check_with_original(trueMat, grailMat))
    # print("prec, rec, and F1 for control: ", check_with_original(trueMat, controlMat))
    print("prec, rec, and F1 for Brute Granger Causality:", check_with_original(trueMat, bruteMat))

print("time for VL:", bruteTime )


