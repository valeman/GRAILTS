from Causal_inference import generate_synthetic
import numpy as np


if __name__ == '__main__':

    for n in range(100, 1000000, 100):
        m = 128
        lags = [2,5,10]
        for lag in lags:
            TS, trueMat = generate_synthetic(n, m = m, lag = lag, ar = [1, 0.5])
            with open('./datasets_ar05/series'+str(lag)+'_'+str(n)+'.npy', 'wb') as f:
                np.save(f, TS)
            with open('./datasets_ar05/truemat'+str(lag)+'_'+str(n)+'.npy', 'wb') as f:
                np.save(f, trueMat)