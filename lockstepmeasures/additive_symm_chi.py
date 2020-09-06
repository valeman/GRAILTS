

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