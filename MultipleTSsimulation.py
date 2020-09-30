import numpy as np
from TSsimulation import SimpleSimulationVLtimeseries


def MultipleSimulationVLtimeseries(n=200, lag=5, YstFixInx=110, YfnFixInx=170, XpointFixInx=100, arimaFlag=True,
                                   seedVal=-1):
    TS = np.zeros((n, 10))  # 10 time series
    if seedVal == -1:
        seeds = np.array([-1, -1, -1])
    else:
        np.random.seed(seedVal)
        seeds = np.random.uniform(1000, 250000, 3)
    AX, AY = SimpleSimulationVLtimeseries(n=n, lag=lag, YstFixInx=YstFixInx, YfnFixInx=YfnFixInx,
                                          XpointFixInx=XpointFixInx, arimaFlag=arimaFlag, seedVal=seeds[1])
    BX, BY = SimpleSimulationVLtimeseries(n=n, lag=lag + 5, YstFixInx=YstFixInx, YfnFixInx=YfnFixInx,
                                          XpointFixInx=XpointFixInx, arimaFlag=arimaFlag, seedVal=seeds[2])
    CX, CY = SimpleSimulationVLtimeseries(n=n, lag=lag + 8, YstFixInx=YstFixInx, YfnFixInx=YfnFixInx,
                                          XpointFixInx=XpointFixInx, arimaFlag=arimaFlag, seedVal=seeds[3])
    TS[:, 1] = AX
    TS[:, 2] = BX
    TS[:, 3] = CX
    TS[:, 4] = AY
    TS[:, 5] = BY
    TS[:, 6] = CY
    TS[:, 7] = AY + BY
    TS[:, 8] = AY + CY
    TS[:, 9] = CY + BY
    TS[:, 10] = AY + BY + CY
    return TS


def checkMultipleSimulationVLtimeseries(trueAdjMat, adjMat):
    TP = 0
    FP = 0
    FN = 0
    for i in range(10):
        for j in range(10):
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
