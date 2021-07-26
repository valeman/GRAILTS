import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from Causal_inference import granger_causality,add_causality_dataset, generate_synthetic
from time import time
import pmdarima as pm
from tqdm import tqdm
from statsmodels.tsa.arima_model import ARMA
from TimeSeries import TimeSeries
from copy import deepcopy


def plot(x):
    '''
    plot time series
    :param x: time series
    :return:
    '''
    l = np.arange(x.shape[0])
    plt.plot(l, x, c='blue')
    plt.show()

def add_noise(TS, sd):
    '''
    Add Gaussian noise to dataset
    :param TS: dataset
    :param sd: standard deviation
    :return: dataset with noise
    '''
    n, m = TS.shape
    newTS = deepcopy(TS)
    for i in range(n):
        newTS[i, :] += np.random.normal(0, sd, m)
    return newTS

def random_walk(drift = 0, m = 128, sd = 1):
    '''
    Generate random walk
    :param drift:
    :param m: length of random walk
    :param sd: standard deviation for gaussian
    :return: random walk
    '''
    ls = [None] * m
    ls[0] = 0
    for i in range(1, m):
        ls[i] = ls[i-1] + drift + np.random.normal(0,sd)
    return np.array(ls)

def fit_arima_dataset(TS):
    '''
    Fit auto arima to dataset and double the dataset with arima fits
    :param TS: base dataset
    :return: new dataset with size 2n where i+1 is the arima fit of i
    '''
    n, m = TS.shape
    newts = []
    for x in tqdm(TS):
        newts.append(x)
        model = pm.auto_arima(x, seasonal=False)
        res = model.arima_res_
        fitted = res.fittedvalues
        newts.append(fitted)
    newts = np.array(newts)

    return newts


def split_cause_effect_truemat(TS):
    '''
    Fit arima and split to cause and effect datasets
    :param TS:
    :return:
    '''
    newts = deepcopy(TS)
    newts, _ = fit_arima_dataset(newts)
    causal_db = np.array([newts[i, :] for i in range(0, newts.shape[0], 2)])
    effect_db = np.array([newts[i, :] for i in range(1, newts.shape[0], 2)])
    assert causal_db.shape == effect_db.shape
    truemat = np.zeros((causal_db.shape[0], effect_db.shape[0]))
    for i in range(causal_db.shape[0]):
        truemat[i, i] = 1

    return causal_db, effect_db, truemat


def arima_dataset_to_noisy(TS, sd_list, name):
    '''
    Generate test sets from TS using the ARIMA process and added noise with standard deviations in sd_list
    :param TS: base time series dataset
    :param sd_list: standard deviations list
    :param name: name for the generated files
    :return:
    '''

    causal_db, effect_db, truemat = split_cause_effect_truemat(TS)


    for sd in sd_list:
        effect_db = add_noise(effect_db, sd)
        with open('{}_randomsd{}_causaldb.npy'.format(name, sd), 'wb') as f:
            np.save(f, causal_db)
        with open('{}_randomsd{}_effectdb.npy'.format(name, sd), 'wb') as f:
            np.save(f, effect_db)

    with open('{}_random_split_truemat.npy'.format(name), 'wb') as f:
        np.save(f, truemat)

    if len(sd_list) == 1:
        return causal_db, effect_db, truemat



def arima_dataset_to_rwalk(TS, sd_list, name):
    '''
    Generate test sets from TS using the ARIMA process and added noise (as random walks) with standard deviations in sd_list
    :param TS: base time series dataset
    :param sd_list: standard deviations list for random walks
    :param name: name for the generated files
    :return:
    '''
    causal_db, effect_db, truemat = split_cause_effect_truemat(TS)

    for sd in sd_list:
        for i in range(effect_db.shape[0]):
            walk = random_walk(m=effect_db.shape[1], sd=sd)
            effect_db[i] += walk

        with open('{}_rwalksd{}_causaldb.npy'.format(name, sd), 'wb') as f:
            np.save(f, causal_db)
        with open('{}_rwalksd{}_effectdb.npy'.format(name, sd), 'wb') as f:
            np.save(f, effect_db)

    with open('{}_rwalk_split_truemat.npy'.format(name), 'wb') as f:
        np.save(f, truemat)

    if len(sd_list) == 1:
        return causal_db, effect_db, truemat


# def arima_dataset_to_mix(TS, randsd_list, rwalksd_list, name):
#     '''
#     Generate test sets but with mixed noise from both gaussian and random walk
#     :param TS: : base time series dataset
#     :param randsd_list: standard deviations for gaussian noise
#     :param rwalksd_list: standard deviations for the random walk
#     :param name: name for the generated files
#     :return:
#     '''
#     newts = deepcopy(TS)
#     newts, _ = fit_arima_dataset(newts)
#     causal_db = np.array([newts[i, :] for i in range(0, newts.shape[0], 2)])
#     effect_db = np.array([newts[i, :] for i in range(1, newts.shape[0], 2)])
#     assert causal_db.shape == effect_db.shape
#
#     for
#     effect_db = add_noise(effect_db, randsd)
#     for i in range(effect_db.shape[0]):
#         walk = random_walk(m=effect_db.shape[1], sd=rwalksd)
#         effect_db[i] += walk
#
#     truemat = np.zeros((causal_db.shape[0], effect_db.shape[0]))
#     for i in range(causal_db.shape[0]):
#         truemat[i, i] = 1
#
#     with open('{}_mixsd{}_{}_causaldb.npy'.format(name, randsd, rwalksd), 'wb') as f:
#         np.save(f, causal_db)
#     with open('{}_mixsd{}_{}_effectdb.npy'.format(name, randsd, rwalksd), 'wb') as f:
#         np.save(f, effect_db)
#     with open('{}_mix_split_truemat.npy'.format(name), 'wb') as f:
#         np.save(f, truemat)
#
#     return causal_db, effect_db, truemat


if __name__ == '__main__':
    source_path = '/tartarus/DATASETS/UCR2018'
    dst_path = './test_files'

    sd_list = [0.1, 0.2]
    randomwalk_sd_list = [0.1, 0.05]
    mixed_sd_list = [(0.1,0.1),(0.1,0.05),(0.2,0.1),(0.2,0.05)]

    #load datasets
    for dataset_name in os.listdir(source_path):
        print(dataset_name)

        trainfile = '{}_TRAIN'.format(dataset_name)
        testfile = '{}_TEST'.format(dataset_name)
        trainpath = os.path.join(source_path, trainfile)
        testpath = os.path.join(source_path, testfile)
        TRAIN, train_labels = TimeSeries.load(trainpath, "UCR")
        TEST, test_labels = TimeSeries.load(testpath, "UCR")
        merged_ts = np.vstack((TRAIN, TEST))
        causal_db, effect_db, truemat = split_cause_effect_truemat(merged_ts)
        assert causal_db.shape == effect_db.shape

        with open(os.path.join(dst_path, '{}_causaldb.npy'.format(dataset_name)), 'wb') as f:
            np.save(f, causal_db)

        with open(os.path.join(dst_path, '{}_split_truemat.npy'.format(dataset_name)), 'wb') as f:
            np.save(f, truemat)

        #add gaussian noise
        for sd in sd_list:
            noisy_effect_db = add_noise(effect_db, sd)
            with open(os.path.join(dst_path, '{}_randomsd{}_effectdb.npy'.format(dataset_name, sd)), 'wb') as f:
                np.save(f, noisy_effect_db)

        #add random walk noise
        for sd in randomwalk_sd_list:
            rwalk_effect_db = deepcopy(effect_db)
            for i in range(effect_db.shape[0]):
                walk = random_walk(m=effect_db.shape[1], sd=sd)
                rwalk_effect_db[i] += walk

            with open(os.path.join(dst_path, '{}_rwalksd{}_effectdb.npy'.format(dataset_name, sd)), 'wb') as f:
                np.save(f, effect_db)

        #add mixed noise
        for sd1, sd2 in mixed_sd_list:
            noisy_effect_db = add_noise(effect_db, sd1)
            mixed_db = deepcopy(noisy_effect_db)
            for i in range(effect_db.shape[0]):
                walk = random_walk(m=effect_db.shape[1], sd=sd2)
                mixed_db[i] += walk
            with open(os.path.join(dst_path, '{}_mixsd{}_{}_causaldb.npy'.format(dataset_name, sd1, sd2)), 'wb') as f:
                np.save(f, causal_db)








