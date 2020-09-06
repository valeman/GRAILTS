def vicis_wave_hedges(x,y):
    if len(x) != len(y):
        return -1;
    sum = 0;
    for i in range(len(x)):
        sum += abs(x[i] - y[i]) / min((x[i],y[i]))
        
    return sum;
