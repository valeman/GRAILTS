def kulczynski(x,y):
    if len(x) != len(y):
        return -1;
    suma = 0;
    sumb = 0;
    for i in range(len(x)):
        suma += abs(x[i] - y[i]);
        sumb += min(x[i], y[i]);
    if sumb == 0:
        return -1;
    return suma/sumb;
