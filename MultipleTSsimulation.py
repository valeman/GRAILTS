import numpy as np
from TSsimulation import SimpleSimulationVLtimeseries
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri
import pandas as pd
from rpy2.robjects.conversion import localconverter
from rpy2.robjects import pandas2ri
from rpy2.robjects.vectors import DataFrame, FloatVector, IntVector, StrVector, ListVector
import numpy
from collections import OrderedDict

#R listvector to python dictionary
def recurList(data):
    rDictTypes = [ DataFrame,ListVector]
    rArrayTypes = [FloatVector,IntVector]
    rListTypes=[StrVector]
    if type(data) in rDictTypes:
        return OrderedDict(zip(data.names, [recurList(elt) for elt in data]))
    elif type(data) in rListTypes:
        return [recurList(elt) for elt in data]
    elif type(data) in rArrayTypes:
        return numpy.array(data)
    else:
        if hasattr(data, "rclass"): # An unsupported r class
            raise KeyError('Could not proceed, type {} is not defined'.format(type(data)))
        else:
            return data # We reached the end of recursion


# For this one times series are stored in rows.
def genMultipleSimulation(half_ts_num = 5, prob_of_causing = 0.5, n=200, lag=5, YstFixInx=110, YfnFixInx=170, XpointFixInx=100, arimaFlag=True, seedVal=-1):
    TS = np.zeros((half_ts_num*2, n))  
    trueMatrix = np.zeros((half_ts_num * 2, half_ts_num * 2))
    if seedVal == -1:
        seeds = np.ones(half_ts_num) * (-1)
    else:
        np.random.seed(seedVal)
        seeds = np.random.uniform(1000, 250000, 2*half_ts_num)
        
    for i in range(half_ts_num):
        causalFlag = False
        causes = np.random.choice(2, p = [1-prob_of_causing, prob_of_causing])
        if causes == 1:
            causalFlag = True
            trueMatrix[2*i,2*i + 1] = 1
        TS[2*i], TS[2*i + 1] = SimpleSimulationVLtimeseries(n=n, lag=lag, YstFixInx=YstFixInx, YfnFixInx=YfnFixInx,
                                          XpointFixInx=XpointFixInx, arimaFlag=arimaFlag, seedVal=seeds[i],causalFlag=causalFlag)
    return TS, trueMatrix

def gen_from_density(ts_num, caused_neighbor_num = 1, n = 200, lag = 5, YstFixInx=110,YfnFixInx=170, seedVal = -1):
    TS = np.zeros((ts_num, n + lag))
    trueMatrix = np.zeros((ts_num, ts_num))

    if seedVal != -1:
        np.random.seed(seedVal)

    for i in range(ts_num):
        TS[i,:] = np.random.normal(0,1,n+lag)

    for i in range(ts_num):
        causes = np.random.choice(ts_num, caused_neighbor_num)
        trueMatrix[i, causes] = 1
        for neighbor in causes:
            for j in range(n):
                TS[neighbor, j + lag] += TS[i, j]

    for i in range(ts_num):
        TS[i, YstFixInx:YfnFixInx] = TS[i, YstFixInx]

    return TS[:, lag:], trueMatrix


def MultipleSimulationVLtimeseries(ts_num = 10, prob_of_causing = 0.1, n=200, lag=5, YstFixInx=110, YfnFixInx=170, XpointFixInx=100, arimaFlag=True, seedVal=-1):
    TS = np.zeros((n, 10))  # 10 time series
    if seedVal == -1:
        seeds = np.array([-1, -1, -1])
    else:
        np.random.seed(seedVal)
        seeds = np.random.uniform(1000, 250000, 3)
    AX, AY = SimpleSimulationVLtimeseries(n=n, lag=lag, YstFixInx=YstFixInx, YfnFixInx=YfnFixInx,
                                          XpointFixInx=XpointFixInx, arimaFlag=arimaFlag, seedVal=seeds[0])
    BX, BY = SimpleSimulationVLtimeseries(n=n, lag=lag + 5, YstFixInx=YstFixInx, YfnFixInx=YfnFixInx,
                                          XpointFixInx=XpointFixInx, arimaFlag=arimaFlag, seedVal=seeds[1])
    CX, CY = SimpleSimulationVLtimeseries(n=n, lag=lag + 8, YstFixInx=YstFixInx, YfnFixInx=YfnFixInx,
                                          XpointFixInx=XpointFixInx, arimaFlag=arimaFlag, seedVal=seeds[2])
    TS[:, 0] = AX
    TS[:, 1] = BX
    TS[:, 2] = CX
    TS[:, 3] = AY
    TS[:, 4] = BY
    TS[:, 5] = CY
    TS[:, 6] = AY + BY
    TS[:, 7] = AY + CY
    TS[:, 8] = CY + BY
    TS[:, 9] = AY + BY + CY
    return TS


def checkMultipleSimulationVLtimeseries(trueAdjMat, adjMat):
    assert adjMat.shape == trueAdjMat.shape
    TP = 0
    FP = 0
    FN = 0
    for i in range(adjMat.shape[0]):
        for j in range(adjMat.shape[1]):
            if trueAdjMat[i, j] and adjMat[i, j]:
                TP = TP + 1
            elif (not trueAdjMat[i, j]) and adjMat[i, j]:
                FP = FP + 1
            elif trueAdjMat[i, j] and (not adjMat[i, j]):
                FN = FN + 1
    prec = TP / (TP + FP)
    rec = TP / (TP + FN)
    F1 = 2 * prec * rec / (prec + rec)
    return [prec, rec, F1]
'''
importr("VLTimeCausality")
vl = robjects.r("VLTimeCausality::multipleVLGrangerFunc")


numpy2ri.activate()
pandas2ri.activate()

G = np.zeros((10,10))
G[0, [3,6,7,9]] = True
G[1,[4,6,8,9]] = True
G[2, [5,7,8,9]] =True
#TS = MultipleSimulationVLtimeseries()
mulsim = robjects.r("VLTimeCausality::MultipleSimulationVLtimeseries")
TS = mulsim(seedVal = 1)
#TSR = robjects.r.matrix(TS, nrow = TS.shape[0],ncol = TS.shape[1] )
#robjects.r.assign("TS", TSR)
out = vl(TS)
#with localconverter(robjects.default_converter + pandas2ri.converter):
#  out_pd = robjects.conversion.rpy2py(out)
out_py = recurList(out)
#print(type(out))
print(checkMultipleSimulationVLtimeseries(trueAdjMat=G,adjMat=out_py["adjMat"])) '''
