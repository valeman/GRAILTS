from MultipleTSsimulation import MultipleSimulationVLtimeseries, recurList, genMultipleSimulation, checkMultipleSimulationVLtimeseries
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

ts_num = 50
#can try kdtw here
grail = Representation.GRAIL(kernel="SINK", d = 40)
vl = robjects.r("VLTimeCausality::VLGrangerFunc")
numpy2ri.activate()
pandas2ri.activate()

r_source = robjects.r['source']
r_source('helper.R')
vlhelperfunc = robjects.globalenv['vlhelperfunc']
vlmultigranger = robjects.r("VLTimeCausality::multipleVLGrangerFunc")

TS, trueMat = genMultipleSimulation(half_ts_num = int(ts_num/2))

grailMat = np.zeros((ts_num, ts_num))
t = time()
vlOut = vlmultigranger(np.transpose(TS))
VLtime = time() - t
vlOut_py = recurList(vlOut)
vlMat = vlOut_py["adjMat"]

neighbors, _ = kNN(TS, TS, method = "ED", k = 10, representation=grail, use_exact_rep = True, pq_method = "opq", M = 16)

t = time()
for i in range(ts_num):
    for j in neighbors[i]:
        if j != i:
            out = vl(TS[j,:], TS[i,:]) # second one causes the first
            grailMat[i,j] = vlhelperfunc(out)
prunedtime = time() - t
            
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
