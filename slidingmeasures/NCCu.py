def NCCu(x,y):

    result = np.correlate(x,y,'full');

    max = math.ceil(len(result)/2);

    a = []
    for i in range(result.size):
            if (i > max - 1):
                a.append(2*max-(i + 1));
            else:
                a.append(i + 1);

    return np.divide(result,a);
