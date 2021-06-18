from TimeSeries import TimeSeries
import Representation
from kNN import kNN, kNN_with_pq_NCC, kNN_with_pq_SINK, MAP, avg_recall_measure
import numpy as np
from Causal_inference import generate_synthetic
from Causal_inference import add_causality_dataset

best_gamma = 5



TRAIN, train_labels = TimeSeries.load("ECG200_TRAIN", "UCR")
TEST, test_labels = TimeSeries.load("ECG200_TEST", "UCR")

# with open('datasets_ar05/series2_200.npy', 'rb') as f:
#     TRAIN = np.load(f)
#     TEST = TRAIN[:,:]
#

# TRAIN = np.zeros((50,128))#generate_synthetic(200, m = 128, lag = 2, ar = [1, 1],ma = [0.01])
# TEST = np.zeros((50,128))
#
# for i in range(50):
#     for j in range(128):
#         TRAIN[i,j] = np.random.normal(0,1)
#         TEST[i,j] = TRAIN[i,j]

# for i in range(50):
#     TRAIN[i,0] = np.random.normal(0,1)
#     TEST[i, 0] = TRAIN[i, 0]
#     for j in range(1,128):
#         TRAIN[i,j] = TRAIN[i,j-1]+1
#         TEST[i,j] = TRAIN[i,j]

print(np.sum(TRAIN - TEST))

add_causality_dataset(TRAIN, lag = 5)
#add_causality_dataset(TEST, lag = 5)

print(TRAIN)
print(TRAIN.shape)

representation = Representation.GRAIL(kernel="SINK", d=100,gamma = best_gamma)

TRAIN_TS, TEST_TS = representation.get_rep_train_test(TRAIN, TRAIN)

neighbors, _, _ = kNN(TRAIN_TS, TEST_TS, method="ED", k=10, representation=None, use_exact_rep=True,
                      pq_method=None)

exact_neighbors, _, _ = kNN(TRAIN, TRAIN, method="SINK", k=10, representation=None, gamma_val=best_gamma)

#
# TRAIN_TS, TEST_TS = representation.get_rep_train_test(TRAIN, TRAIN)
#
# neighbors, _, _ = kNN(TRAIN_TS, TEST_TS, method="ED", k=10, representation=None, use_exact_rep=True,
#                       pq_method=None)
#
# exact_neighbors, _, _ = kNN(TRAIN, TRAIN, method="SINK", k=10, representation=None, gamma_val=best_gamma)

knn_map_accuracy = MAP(exact_neighbors, neighbors)
knn_recall_accuracy = avg_recall_measure(exact_neighbors, neighbors)

print(knn_recall_accuracy, knn_map_accuracy)