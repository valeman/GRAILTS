
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

    return 1 - (sum/ ((sumx ** (1/2))*(sumy ** (1/2))));
