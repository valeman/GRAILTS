def motyka(x,y):
    if len(x) != len(y):
        return -1;
    suma = 0;
    sumb = 0;
    for i in range(len(x)):
        suma += max((x[i],y[i]))
        sumb += x[i] + y[i];
    return suma/sumb;
