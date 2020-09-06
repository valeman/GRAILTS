def square_chord(x,y):
    if len(x) != len(y):
        return -1;
    sum = 0;
    for i in range(len(x)):
        sum += ((x[i] ** (1/2)) - (y[i]) ** (1/2)) ** 2;

    return sum;
