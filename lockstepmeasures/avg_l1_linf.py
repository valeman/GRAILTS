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