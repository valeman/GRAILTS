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