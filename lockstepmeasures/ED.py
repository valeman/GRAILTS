
def ED(x,y):
    if len(x) != len(y):
        return -1;
    sum = 0
    for i in range(len(x)):
        sum += (x[i] - y[i]) ** 2;


    if sum < 0:
        return -1;

    return sum ** (1/2);