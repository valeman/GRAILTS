import numpy as np;

def fftlbtopcoeff(x,y,coeff):
    fx = np.divide(np.fft.fft(x),(len(x) ** (1/2)));
    fy = np.divide(np.fft.fft(y),(len(x) ** (1/2)));

    return (np.sum(np.square(np.subtract(fx[0:coeff],fy[0:coeff]))) ** (1/2))