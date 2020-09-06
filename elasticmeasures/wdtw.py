import numpy as np

def wdtw(x,y,g):


    xlen = len(x);
    ylen = len(y);

    weight_vector = [1 / (1 + np.exp(-g * (i - m / 2))) for i in
                         range(0, xlen)]


    pairwise_distances = np.asarray([[(x_,y_) for y_ in y] for x_ in x ])
    distances = np.full([n,m],np.inf);

    for i in range(0, xlen):
            for j in range(0, ylen):
                if i + j == 0:
                    distances[0][0] = weight_vector[0] * pairwise_distances[0][0];
                elif i == 0:
                    distances[0][j] = distances[0][j - 1] + weight_vector[j] * \
                              pairwise_distances[0][j];
                elif j == 0:
                     distances[i][0] = distances[i - 1][0] + weight_vector[i] * \
                              pairwise_distances[i][0];
                else:
                    min_dist = np.min([distances[i][j - 1], distances[i - 1][j],
                                   distances[i - 1][j - 1]])
                    distances[i][j] = (min_dist + weight_vector[np.abs(i - j)] *
                                   pairwise_distances[i][j])
    return distances[m - 1][n - 1]



