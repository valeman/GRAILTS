def dtw_scb(x,y,w=5):

    cur = np.full(len(y),np.inf);
    prev = np.full(len(y),np.inf);
    for i in range(len(x)):
        minw = max(0,i-w);
        maxw = min(len(y),w);
        temp = prev;
        prev = cur;
        cur = temp;
        for j in range(int(minw),int(maxw)):

            if i + j == 0:
                cur[j] = abs(x[0] - y[0]) ** 2;
            elif i == 0:
                cur[j] = abs(x[0] - y[j]) ** 2 + cur[j-1];
            elif j == 0:
                cur[j] = abs(x[i] - y[0]) ** 2 + prev[j];
            else:
                cur[j] = abs(x[i] - y[j]) ** 2 + min(prev[j-1],prev[j],cur[j-1]);
    final_dtw = cur[len(y)-1];

    return final_dtw ** (1/2);
