import math
import numpy as np;

"""Throughout:
    x is the first time series as an array.
    y is the second time series as an array"""
def abs_euclidean(x,y):
    if len(x) != len(y):
        return -1;
    sum = 0
    for i in range(len(x)):
        sum += abs(x[i] - y[i]) ** 2;

    return math.sqrt(sum);

def additive_symm_chi(x,y):
    if len(x) != len(y):
        return -1;

    
    sum = 0;
    for i in range(len(x)):
        if x[i] == 0:
            return -1;
        if y[i] == 0:
            return -1;
        sum += (x[i] - y[i]) ** 2 * (x[i] + y[i]) / (x[i] * y[i]);

    return sum;

def avg_l1_linf(x,y):
    if len(x) != len(y):
        return -1;
    sum = 0;
    max = 0;
    for i in range(len(x)):
        dif = abs(x[i] - y[i]);
        sum += dif;
        if (dif > max):
            max = dif;

    return (sum + max)/2;

def bhattacharyya(x,y):
    if len(x) != len(y):
        return -1;
    sum = 0;
    for i in range(len(x)):
        if x[i] * y[i] < 0:
            return -1;
        sum += math.sqrt(x[i] * y[i]);

    return - math.log(sum);

def canberra(x,y):
    if len(x) != len(y):
        return -1;
    sum = 0;
    for i in range(len(x)):
        if x[i] + y[i] == 0:
            return -1;

        sum += abs(x[i] - y[i]) / (x[i] + y[i]);

    return sum;

def chebyshev(x,y):
    if len(x) != len(y):
        return -1;
    max = 0
    for i in range(len(x)):
        dif = abs(x[i] - y[i]);
        if max < dif:
            max = dif;
    return max;

def clark(x,y):
    if len(x) != len(y):
        return -1;
    sum = 0;
    for i in range(len(x)):
        if x[i] + y[i] == 0:
            return -1;
        sum += (abs(x[i] - y[i]) ** 2) / (x[i] + y[i])

    return math.sqrt(sum);

def cosine(x,y):
    if len(x) != len(y):
        return -1;
    sum = 0;
    sumx = 0
    sumy = 0
    for i in range(len(x)):
        sumy += y[i] ** 2;
        sumx += x[i] ** 2;
        sum += x[i] * y[i];
    if sumx < 0:
        return -1;
    if sumy < 0:
        return -1;

    return 1 - (sum/ (math.sqrt(sumx) * math.sqrt(sumy)));

def czekanowski(x,y):
    if len(x) != len(y):
        return -1;
    sum_add = 0;
    sum_dif = 0;
    for i in range(len(x)):
        sum_add += (x[i] + y[i]);
        sum_dif += abs(x[i] - y[i]);

    if sum_add == 0:
        return -1;

    return sum_dif/sum_add;

def dice(x,y):
    if len(x) != len(y):
        return -1;
    sum_dif = 0;
    sum_add = 0;
    for i in range(len(x)):
        sum_dif += (x[i] - y[i]) ** 2;
        sum_add += (x[i] ** 2 + y[i] ** 2);

    if (sum_add == 0):
        return -1;
    return sum_dif/sum_add;

def divergence(x,y):
    if len(x) != len(y):
        return -1;
    sum = 0;
    for i in range(len(x)):
        if (x[i] + y[i]) == 0:
            return -1;
        sum += ((x[i] - y[i]) ** 2)/ ((x[i] + y[i]) ** 2);

    

    return 2 * sum;

def ED(x,y):
    if len(x) != len(y):
        return -1;
    sum = 0
    for i in range(len(x)):
        sum += (x[i] - y[i]) ** 2;


    if sum < 0:
        return -1;

    return math.sqrt(sum);

def emanon2(x,y):
    if len(x) != len(y):
        return -1;
    sum = 0
    comp = 0;
    for i in range(len(x)):
        mind = min(x[i],y[i]);
        if mind == 0:
            return -1;
        sum += ((x[i] - y[i]) ** 2) / mind ** 2;
    return sum;

    def emanon3(x,y):
        if len(x) != len(y):
            return -1;
        sum = 0
        comp = 0;
        for i in range(len(x)):
            sum += ((x[i] - y[i]) ** 2) / min(x[i],y[i]);
        return sum;

def emanon4(x,y):
    if len(x) != len(y):
        return -1;
    sum = 0
    comp = 0;
    maxd = max(x + y);
    if maxd == 0:
        return -1;
    for i in range(len(x)):
        sum += ((x[i] - y[i]) ** 2) / maxd;
    return sum;

def fidelity(x,y):
    if len(x) != len(y):
        return -1;
    sum = 0;
    for i in range(len(x)):
        if x[i] * y[i] < 0:
            return -1;
        sum += math.sqrt(x[i] * y[i]);

    return sum;

def gower(x,y):
    if len(x) != len(y):
        return -1;
    if len(x) == 0:
        return -1;
    sum = 0;
    for  i in range(len(x)):
        sum += abs(x[i] - y[i]);
    return 1/len(x) * sum;

def harmonicmean(x,y):
    if len(x) != len(y):
        return -1;

    a = np.multiply(x,y);
    b = np.linalg.pinv([np.add(x,y)]);

    return 2 * np.sum(np.dot(a,b))
    
def hellinger(x,y):
    if len(x) != len(y):
        return -1;
    sum = 1;
    for i in range(len(x)):
        if x[i] * y[i] < 0:
            return -1;
        sum -= math.sqrt(x[i] * y[i]);

    if (sum < 0):
        return -1
    return 2 * math.sqrt(sum);

def innerproduct(x,y):
    if len(x) != len(y):
        return -1;
    sum = 0;
    for i in range(len(x)):
        sum += (x[i] * y[i]);

    return sum;

def intersection(x,y):
    if len(x) != len(y):
        return -1;
    sum = 0;
    for i in range(len(x)):
        sum += abs(x[i] - y[i]);
    
    return 1/2 * sum;

def jaccard(x,y):
    if len(x) != len(y):
        return -1;
    return np.sum(np.square(np.subtract(x,y))) / np.sum(np.square(x) + np.square(y) + np.multiply(np.add(x,y),-1));

def jansen_shannon(x,y):
    if len(x) != len(y):
        return -1;
    logxy = [];
    for i in range(len(x)):
        if x[i] + y[i] <= 0:
            return -1;
        logxy.append(math.log(x[i] + y[i]));

    sum = 0
    for i in range(len(x)):
        if x[i] <= 0:
            return -1;
        if y[i] <= 0:
            return -1;
        sum += x[i] * (math.log(2*x[i]) - logxy[i]) + y[i] * (math.log(2*y[i]) - logxy[i])
    
    return .5 * sum;

def jeffrey(x,y):
    if len(x) != len(y):
        return -1;
    return np.sum(np.multiply((np.subtract(x,y)),np.log(np.divide(x,y))));

    return sum;

def jensen_difference(x,y):
    xyavg = [];
    for i in range(len(x)):
        if x[i] + y[i] <= 0:
            return -1;
        xyavg.append((x[i] + y[i])/2);

    sum = 0;
    for i in range(len(x)):
        if y[i] <= 0:
            return -1;
        if x[i] <= 0:
            return -1;
        sum += (x[i] * math.log(x[i]) + y[i] * math.log(y[i])) / 2 - xyavg[i] * math.log(xyavg[i]);

    return sum;

def k_divergence(x,y):
    if len(x) != len(y):
        return -1;
    for i in range(len(x)):
        if y[i] <= 0:
            return -1;
    return np.sum(np.multiply(x,np.log(np.divide(np.multiply(x,2),np.add(x,y)))));

def kulczynski(x,y):
    if len(x) != len(y):
        return -1;
    suma = 0;
    sumb = 0;
    for i in range(len(x)):
        suma += abs(x[i] - y[i]);
        sumb += min(x[i], y[i]);
    if sumb == 0:
        return -1;
    return suma/sumb;

def kullback(x,y):
    if len(x) != len(y):
        return -1;
    sum = 0;
    for i in range(len(x)):
        if x[i]/y[i] == 0:
            return -1;
        sum+= x[i] * math.log(x[i]/y[i]);

    return sum;

def kumar_johnson(x,y):
    if len(x) != len(y):
        return -1;
    sum = 0;
    for i in range(len(x)):
        sum += ((x[i] ** 2 - y[i] ** 2) ** 2) / (2 * (x[i] * y[i]) ** (3/2));
    return sum;

def kumarhassebrook(x,y):
    if len(x) != len(y):
        return -1;
    return np.sum(np.multiply(x,y)) /np.subtract(np.add(np.sum(np.square(x)),np.sum(np.square(y))),np.sum(np.multiply(x,y)))

def lorentzian(x,y):
    if len(x) != len(y):
        return -1;
    sum = 0;
    for i in range(len(x)):
        sum += math.log(1 + abs(x[i] - y[i]))
    return sum;

def manhattan(x,y):
    if len(x) != len(y):
        return -1;
    sum = 0;
    for i in range(len(x)):
        sum+= (abs(x[i] - y[i]));
    return sum;

def matusita(x,y):
    if len(x) != len(y):
        return -1;
    sum = 0;
    for i in range(len(x)):
        if x[i] * y[i] < 0:
            return -1;
        sum += math.sqrt(x[i] * y[i]);

    result = 2 - 2 * sum;
    if result < 0:
        return -1;
    return math.sqrt(result);

def max_symmetric_chi(x,y):
    if len(x) != len(y):
        return -1;
    suma = 0;
    sumb = 0;
    xy = [];
    for i in range(len(x)):
        xy.append((x[i] - y[i]) ** 2);
    for i in range(len(x)):
        suma += (xy[i]/y[i]);
        sumb += (xy[i]/x[i]);
    return max((suma,sumb));

def min_symmetric_chi(x,y):
    if len(x) != len(y):
        return -1;
    suma = 0;
    sumb = 0;
    xy = [];
    for i in range(len(x)):
        xy.append((x[i] - y[i]) ** 2);
    for i in range(len(x)):
        suma += (xy[i]/y[i]);
        sumb += (xy[i]/x[i]);
    return min((suma,sumb));

"""p is the exponent of the minkowski formula"""
def minkowski(x,y,p):
    if len(x) != len(y):
        return -1;
    sum = 0;
    for i in range(len(x)):
        sum += abs(x[i] - y[i]) ** p;

    return sum ** (1/p);

def motyka(x,y):
    if len(x) != len(y):
        return -1;
    suma = 0;
    sumb = 0;
    for i in range(len(x)):
        suma += max((x[i],y[i]))
        sumb += x[i] + y[i];
    return suma/sumb;

def neyman(x,y):
    if len(x) != len(y):
        return -1;
    sum = 0;
    for i in range(len(x)):
        sum += (x[i] - y[i]) ** 2 / x[i];

    return sum;

def PairWiseScalingDistance(x,y):
    if len(x) != len(y):
        return -1;
    xy = [];
    for i in range(len(x)):
        xy.append(x[i] - y[i]);

    sumx = 0;
    sumxy = 0;
    for i in range(len(x)):
        sumx += x[i] ** 2;
        sumxy += xy[i] ** 2;

    return math.sqrt(sumxy)/math.sqrt(sumx);

def pearson(x,y):
    if len(x) != len(y):
        return -1;
    sum = 0;
    for i in range(len(x)):
        sum += ((x[i] - y[i]) ** 2) / y[i]

    return sum;

def prob_symmetric_chi(x,y):
    if len(x) != len(y):
        return -1;
    sum = 0;
    for i in range(len(x)):
        sum += ((x[i] - y[i]) ** 2) / (x[i] + y[i]);

    return 2 * sum;

def soergel(x,y):
    if len(x) != len(y):
        return -1;
    suma = 0;
    sumb = 0;
    for i in range(len(x)):
        suma += abs(x[i] - y[i])
        sumb += max((x[i],y[i]));
    return suma/sumb;

def sorensen(x,y):
    if len(x) != len(y):
        return -1;
    suma = 0;
    sumb = 0;
    for i in range(len(x)):
        suma += abs(x[i] - y[i]);
        sumb += (x[i] + y[i]);
    return suma/sumb;

def square_chord(x,y):
    if len(x) != len(y):
        return -1;
    sum = 0;
    for i in range(len(x)):
        sum += (math.sqrt(x[i]) - math.sqrt(y[i])) ** 2;

    return sum;

def squared_chi(x,y):
    if len(x) != len(y):
        return -1;
    sum = 0;
    for i in range(len(x)):
        sum += ((x[i] - y[i]) ** 2)/ (x[i] + y[i]);
    return sum;

def squared_euclidean(x,y):
    if len(x) != len(y):
        return -1;
    sum = 0;
    for i in range(len(x)):
        sum+= (x[i] - y[i]) ** 2;
    return sum;

def taneja(x,y):
    if len(x) != len(y):
        return -1;
    sum = 0;
    xy = [];
    for i in range(len(x)):
        xy.append((x[i] + y[i])/ 2);
    
    for i in range(len(x)):
        sum += xy[i] * math.log(xy[i] / math.sqrt(x[i] *y[i]))

    return sum;

def tanimoto(x,y): #Comeback to it
    if len(x) != len(y):
        return -1;
    minxy = np.minimum(x,y);
    sumxy = np.sum(x) + np.sum(y);
    a = (sumxy - 2 * minxy)
    b = np.linalg.pinv([sumxy - minxy])
    return np.sum(np.dot(a,b));

def topsoe(x,y):
    if len(x) != len(y):
        return -1;
    sum = 0;
    logxy = []
    for i in range(len(x)):
        logxy.append(math.log(x[i] + y[i]));
    for i in range(len(x)):
        sum += (x[i] * (math.log(2*x[i]) - logxy[i])) + (y[i] * (math.log(2*y[i]) - logxy[i]));
    return sum;

def vicis_wave_hedges(x,y):
    if len(x) != len(y):
        return -1;
    sum = 0;
    for i in range(len(x)):
        sum += abs(x[i] - y[i]) / min((x[i],y[i]))
        
    return sum;

def wave_hedges(x,y):
    if len(x) != len(y):
        return -1;
    sum = 0;
    for i in range(len(x)):
        sum += min((x[i],y[i])) / max((x[i],y[i]));
        
    return len(x) - sum

def NCC(x,y):

    length = len(x);
    fftlen = 2 ** math.ceil(math.log2(abs(2*length-1)));
    r = np.fft.ifft(np.multiply(np.fft.fft(x,fftlen),np.conj(np.fft.fft(y,fftlen))))
    
    lenr = len(r) - 1;

    result = np.append(r[lenr-length+2:lenr + 1],r[0:length])
        


    return result;

def NCCb(x,y):
    length = len(x);
    fftlen = 2 ** math.ceil(math.log2(abs(2*length-1)));
    r = np.fft.ifft(np.multiply(np.fft.fft(x,fftlen),np.conj(np.fft.fft(y,fftlen))))
    
    lenr = len(r) - 1;

    result = np.append(r[lenr-length+2:lenr + 1],r[0:length])

    return np.divide(result,length);

def NCCc(x,y):
    length = len(x);
    fftlen = 2 ** math.ceil(math.log2(abs(2*length-1)));
    r = np.fft.ifft(np.multiply(np.fft.fft(x,fftlen),np.conj(np.fft.fft(y,fftlen))))
    
    lenr = len(r) - 1;

    result = np.append(r[lenr-length+2:lenr + 1],r[0:length])

    return np.divide(result,np.linalg.norm(x) * np.linalg.norm(y))

def NCCu(x,y):

    result = np.correlate(x,y,'full');

    max = math.ceil(len(result)/2);

    a = []
    for i in range(result.size):
            if (i > max - 1):
                a.append(2*max-(i + 1));
            else:
                a.append(i + 1);

    return np.divide(result,a);
