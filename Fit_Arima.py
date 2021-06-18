from TimeSeries import TimeSeries
import pmdarima as pm
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA


# import Representation
# from kNN import kNN, kNN_with_pq_NCC, kNN_with_pq_SINK, MAP, avg_recall_measure
import numpy as np
# from Causal_inference import generate_synthetic

best_gamma = 5

def load_dataset(foldername, dataset):
    path1 = "{}_TRAIN".format(dataset)
    path2 = "{}_TEST".format(dataset)
    TRAIN, train_labels = TimeSeries.load(path1, "UCR")
    TEST, test_labels = TimeSeries.load(path2, "UCR")

    return TRAIN, train_labels, TEST, test_labels

def fit_arima(train):
    print(train.shape)
    model = pm.auto_arima(train, seasonal=False)
    newstuff = model.arima_res_.simulate(train.shape[0])
    x = np.arange(newstuff.shape[0])
    plt.plot(x, newstuff, c='blue')
    plt.show()
    order = model.order
    model = ARIMA(train, order = order)
    #model.simulate()
    print(model.param_terms)
    return newstuff



if __name__ == '__main__':
    foldername = "."
    dataset = 'ECG200'
    TRAIN, train_labels, TEST, test_labels = load_dataset(foldername, dataset)
    # x = np.arange(TRAIN.shape[1])
    # plt.plot(x, TRAIN[0,:], c='blue')
    # plt.show()
    print(train_labels[:10])
    fit_arima(TRAIN[0,:])
    # fit_arima(TRAIN[1, :])
    # fit_arima(TRAIN[2, :])
    # fit_arima(TRAIN[3, :])
    # fit_arima(TRAIN[4, :])
    # fit_arima(TRAIN[5, :])
    # fit_arima(TRAIN[6, :])

