from MultipleTSsimulation import MultipleSimulationVLtimeseries, \
    recurList, genMultipleSimulation, checkMultipleSimulationVLtimeseries, gen_from_density
import Representation
import rpy2
import rpy2.robjects as robjects
from rpy2.robjects import numpy2ri
import pandas as pd
from rpy2.robjects.conversion import localconverter
from rpy2.robjects import pandas2ri
import numpy as np
from kNN import kNN
from time import time

ts_num = 100
#can try kdtw here
grail = Representation.GRAIL(kernel="SINK", d = 100)
vl = robjects.r("VLTimeCausality::VLGrangerFunc")
numpy2ri.activate()
pandas2ri.activate()

r_source = robjects.r['source']
r_source('helper.R')
vlhelperfunc = robjects.globalenv['vlhelperfunc']
vlmultigranger = robjects.r("VLTimeCausality::multipleVLGrangerFunc")

TS, trueMat = gen_from_density(ts_num, caused_neighbor_num=1, seedVal=1)

grailMat = np.zeros((ts_num, ts_num))
t = time()
vlOut = vlmultigranger(np.transpose(TS))
VLtime = time() - t
vlOut_py = recurList(vlOut)
vlMat = vlOut_py["adjMat"]

neighbor_param = [2]
density_param = [1]

together = np.vstack((TS, TS))
rep_together = grail.get_exact_representation(together)
TRAIN_TS = rep_together[0:ts_num, :]
TEST_TS = rep_together[ts_num:, :]


for neighbor_num in neighbor_param:
    neighbors, _ = kNN(TRAIN_TS, TEST_TS, method = "ED", k = neighbor_num, representation=None, use_exact_rep = True, pq_method = "opq", M = 16)

    t = time()
    for i in range(ts_num):
       for j in neighbors[i]:
           if j != i:
              out = vl(TS[j,:], TS[i,:]) # second one causes the first
              grailMat[i,j] = vlhelperfunc(out)
    prunedtime = time() - t
    print("k number for kNN:", neighbor_num)
    print("Pruned VL time: ", prunedtime)
    print("prec, rec, and F1 for GRAIL: ", checkMultipleSimulationVLtimeseries(trueMat, grailMat))
    print("prec, rec, and F1 for VL:", checkMultipleSimulationVLtimeseries(trueMat, vlMat))

print("time for VL:", VLtime )

#TS_GRAIL = grail.get_exact_representation(np.transpose(bigTS))
#print(TS_GRAIL.shape)
#vl = robjects.r("VLTimeCausality::multipleVLGrangerFunc")
#out = vl(np.transpose(TS_GRAIL))
#print(vl(np.transpose(TS)))
#print(out)
#out_py = recurList(out)
