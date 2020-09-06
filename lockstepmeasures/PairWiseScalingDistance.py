def PairWiseScalingDistance(x,y):
    if len(x) != len(y):
        return -1;
    xy = [];
    for i in range(len(x)):
        xy.append(x[i] - y[i]);

    sumx = 0;
    sumxy = 0;
    for i in range(len(x)):
        sumx += x[i] ** 2;
        sumxy += xy[i] ** 2;

    return (sumxy ** (1/2))/(sumx ** (1/2));
