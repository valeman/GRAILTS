def dice(x,y):
    if len(x) != len(y):
        return -1;
    sum_dif = 0;
    sum_add = 0;
    for i in range(len(x)):
        sum_dif += (x[i] - y[i]) ** 2;
        sum_add += (x[i] ** 2 + y[i] ** 2);

    if (sum_add == 0):
        return -1;
    return sum_dif/sum_add;