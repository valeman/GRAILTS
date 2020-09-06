def neyman(x,y):
    if len(x) != len(y):
        return -1;
    sum = 0;
    for i in range(len(x)):
        sum += (x[i] - y[i]) ** 2 / x[i];

    return sum;
