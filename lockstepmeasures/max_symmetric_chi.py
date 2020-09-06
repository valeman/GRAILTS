def max_symmetric_chi(x,y):
    if len(x) != len(y):
        return -1;
    suma = 0;
    sumb = 0;
    xy = [];
    for i in range(len(x)):
        xy.append((x[i] - y[i]) ** 2);
    for i in range(len(x)):
        suma += (xy[i]/y[i]);
        sumb += (xy[i]/x[i]);
    return max((suma,sumb));
