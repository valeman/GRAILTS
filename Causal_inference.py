from statsmodels.tsa.stattools import grangercausalitytests
import numpy as np
import pandas as pd
from statsmodels.tsa.arima_process import arma_generate_sample
from statsmodels.tsa.stattools import adfuller



def adf_test(x):
    """
    Test if the time series is stationary
    :param x: time series
    :return:
    """
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(x, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    print (dfoutput)

def preprocess(TS):
    n,m = TS.shape
    for x in TS:
        if not adf_test(x):
            for i in range(1,m):
                x[i] -= x[i-1]



def granger_causality(y, x, lag, pval = 0.05):
    """
    Return true if x causes y
    :param y: tseries
    :param x: tseries
    :param lag: max lag to check
    :return: Return true if x causes y
    """
    df = pd.DataFrame(columns=["t2", "t1"], data=zip(y, x))
    res = grangercausalitytests(df, lag, verbose=False)
    pvals = []
    for lg in range(1, lag + 1):
        pvals.append(res[lg][0]['ssr_ftest'][1])
    min_pval = min(pvals)
    if min_pval < pval:
        return True
    return False

def add_causality_dataset(TS, lag, labels = None, method  = 'standard'):
    n, m = TS.shape
    true_causality_matrix = np.zeros((n, n))
    if method == 'standard':
        for i in range(0,n,2):
            for j in range(m-lag):
                TS[i+1, j + lag] += TS[i, j]
            true_causality_matrix[i, i + 1] = 1
    elif method == 'same_group':
        label_set = set(labels)
        for group in label_set:
            group_indices = [i for i in range(n) if labels[i] == group]
            for i in range(0,len(group_indices),2):
                idx = group_indices[i]
                nextidx = group_indices[i+1]
                for j in range(m-lag):
                    TS[nextidx, j + lag] += TS[idx, j]
            true_causality_matrix[idx, nextidx] = 1

    elif method == 'between_groups':
        used = [False for i in range(n)]
        for idx in range(n):
            for nextidx in range(idx+1, n):
                if used[idx] or used[nextidx]:
                    continue
                if labels[idx] != labels[nextidx]:
                    for j in range(m - lag):
                        TS[nextidx, j + lag] += TS[idx, j]
                    used[idx] = True
                    used[nextidx] = True
                    true_causality_matrix[idx, nextidx] = 1
                    break

    return TS, true_causality_matrix

def add_causality(x, y, lag):
    m = y.shape[0]
    for i in range(m):
        y[i + lag] += x[i]


def generate_synthetic( n = 100, m = 200,  lag = 5, ar = [1,0.5], ma = [0.1]):
    """
    Generates synthetic data such that time series i causes i+1 for even i
    :param n: number of time series
    :param m: length of one time series
    :param lag: causation lag
    :param ar: ar parameters for arma model
    :param ma: ma parameteres for arma model
    :return: nxm matrix of time series and true causal matrix
    """
    TS = np.zeros((n, m + lag))
    true_causality_matrix = np.zeros((n,n))
    ar = np.asarray(ar)
    ma = np.asarray(ma)
    # due to signal.lfilter sign is negated
    ar[1:] = -ar[1:]

    for i in range(n):
        TS[i] = arma_generate_sample(ar, ma, m + lag)

    for i in range(0,n,2):
        for j in range(m):
            TS[i+1, j + lag] += TS[i, j]

        true_causality_matrix[i, i+1] = 1

    return TS[:, lag:], true_causality_matrix


def granger_matrix(TS, lag = 5):
    """
    computes causal matrix
    :param TS: nxm time series matrix
    :param lag: max lag to check
    :return: nxn causality matrix
    """
    n = TS.shape[0]
    gr_mat = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            gr_mat[i,j] = granger_causality(TS[j], TS[i], lag)

    return gr_mat

def check_with_original(trueAdjMat, adjMat):
    """
    Compares guessed causality matrix with the true causal matrix
    :param trueAdjMat: original causal matrix
    :param adjMat: guessed matrix
    :return: precision,recall and f-score
    """
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
    if TP + FP == 0:
        prec = 0
    else:
        prec = TP / (TP + FP)
    rec = TP / (TP + FN)
    if prec+rec != 0:
        F1 = 2 * prec * rec / (prec + rec)
    else:
        F1 = 0
    return [prec, rec, F1]
