def divergence(x,y):
    if len(x) != len(y):
        return -1;
    sum = 0;
    for i in range(len(x)):
        if (x[i] + y[i]) == 0:
            return -1;
        sum += ((x[i] - y[i]) ** 2)/ ((x[i] + y[i]) ** 2);

    

    return 2 * sum;