import numpy as np;
import math

def NCC(x,y):

    length = len(x);
    fftlen = 2 ** math.ceil(math.log2(abs(2*length-1)));
    r = np.fft.ifft(np.multiply(np.fft.fft(x,fftlen),np.conj(np.fft.fft(y,fftlen))))
    
    lenr = len(r) - 1;

    result = np.append(r[lenr-length+2:lenr + 1],r[0:length])

    return result;
