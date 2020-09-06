def wave_hedges(x,y):
    if len(x) != len(y):
        return -1;
    sum = 0;
    for i in range(len(x)):
        sum += min((x[i],y[i])) / max((x[i],y[i]));
        
    return len(x) - sum
