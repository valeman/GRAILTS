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
