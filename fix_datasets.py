from Causal_inference import generate_synthetic
import numpy as np


def generate_fixed_data(arparams, maparams):
    for n in range(100, 2000, 100):
        m = 128
        lags = [2,5,10]
        for lag in lags:
            TS, trueMat = generate_synthetic(n, m = m, lag = lag, arparams = arparams, maparams = maparams)
            with open('./datasets_ar4/series'.format()+str(lag)+'_'+str(n)+'.npy', 'wb') as f:
                np.save(f, TS)
            with open('./datasets_ar4/truemat'+str(lag)+'_'+str(n)+'.npy', 'wb') as f:
                np.save(f, trueMat)


if __name__ == '__main__':

    # for n in range(100, 1000000, 100):
    #     m = 128
    #     lags = [2,5,10]
    #     for lag in lags:
    #         TS, trueMat = generate_synthetic(n, m = m, lag = lag, ar = [1, 0.5])
    #         with open('./datasets_ar05/series'+str(lag)+'_'+str(n)+'.npy', 'wb') as f:
    #             np.save(f, TS)
    #         with open('./datasets_ar05/truemat'+str(lag)+'_'+str(n)+'.npy', 'wb') as f:
    #             np.save(f, trueMat)

    #generate_fixed_data(arparams = [.75, -.25], maparams = [])
    generate_fixed_data(arparams = [.75, -.5, .25, -.15], maparams = [])
