import numpy as np;




def BestCoeff(x,coeff):
    Y = np.square(x);

    Ysorted = np.sort(np.multiply(Y,-1));
    Yorder  = np.argsort(np.multiply(Y,-1));
    Ysorted = np.divide(np.cumsum(np.multiply(Ysorted,-1)),np.sum(Y));

    x[Yorder[coeff+1:len(Yorder)-1]] = 0;

    return x;



def fftlbestcoeff(x,y,coeff):
    
    fx = np.divide(np.fft.fft(x),(len(x) ** (1/2)))
    fy = np.divide(np.fft.fft(y),(len(x) ** (1/2)))

    Xred = BestCoeff(fx,coeff);
    Yred = BestCoeff(fy,coeff);



    return (np.sum(np.subtract(Xred,Yred) ** 2) ** (1/2))