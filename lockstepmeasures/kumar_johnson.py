def kumar_johnson(x,y):
    if len(x) != len(y):
        return -1;
    sum = 0;
    for i in range(len(x)):
        sum += ((x[i] ** 2 - y[i] ** 2) ** 2) / (2 * (x[i] * y[i]) ** (3/2));
    return sum;
