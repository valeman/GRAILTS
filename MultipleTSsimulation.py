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
