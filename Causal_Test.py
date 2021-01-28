import Representation
import numpy as np
from kNN import kNN
from time import time
from Causal_inference import generate_synthetic, granger_matrix, check_with_original, granger_causality

n = 100
lag = 5
#can try kdtw here
representation = Representation.GRAIL(kernel="SINK", d = 10) #change

TS, trueMat = generate_synthetic(n, m = 100, lag = 5, ar = [1,0.5])

grailMat = np.zeros((n, n))
t = time()
bruteMat = granger_matrix(TS, lag)
bruteTime = time() - t


neighbor_param = [2, 5, 10, 100]

#TRAIN_TS, TEST_TS = representation.get_rep_train_test(TS, TS)

for neighbor_num in neighbor_param:
    #neighbors, _, _ = kNN(TRAIN_TS, TEST_TS, method = "ED", k = neighbor_num, representation=None, use_exact_rep = True, pq_method = "opq", M = 16)
    neighbors, _, _ = kNN(TS, TS, method="NCC", k=neighbor_num, representation=None, use_exact_rep=True,
                          pq_method=None, gamma_val = 1)

    t = time()
    for i in range(n):
       for j in neighbors[i]:
           if j != i:
              grailMat[i,j] = granger_causality(TS[j], TS[i], lag)
    prunedtime = time() - t
    print("k number for kNN:", neighbor_num)
    print("Pruned VL time: ", prunedtime)
    print("prec, rec, and F1 for GRAIL: ", check_with_original(trueMat, grailMat))
    print("prec, rec, and F1 for VL:", check_with_original(trueMat, bruteMat))

print("time for VL:", bruteTime )


