def emanon3(x,y):
    if len(x) != len(y):
        return -1;
    sum = 0
    comp = 0;
    for i in range(len(x)):
        sum += ((x[i] - y[i]) ** 2) / min(x[i],y[i]);
    return sum;