
def czekanowski(x,y):
    if len(x) != len(y):
        return -1;
    sum_add = 0;
    sum_dif = 0;
    for i in range(len(x)):
        sum_add += (x[i] + y[i]);
        sum_dif += abs(x[i] - y[i]);

    if sum_add == 0:
        return -1;

    return sum_dif/sum_add;