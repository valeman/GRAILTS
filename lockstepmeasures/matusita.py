def matusita(x,y):
    if len(x) != len(y):
        return -1;
    sum = 0;
    for i in range(len(x)):
        if x[i] * y[i] < 0:
            return -1;
        sum += (x[i] * y[i]) ** (1/2);

    result = 2 - 2 * sum;
    if result < 0:
        return -1;
    return result ** (1/2);
