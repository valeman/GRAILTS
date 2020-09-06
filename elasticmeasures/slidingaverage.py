import numpy as np;

def movingAverage(x,w):

    length = len(x) - w + 1;
    newx = np.zeros(int(length))

    for i in range(int(length)):
        sumd = 0;
        for j in range(w):
            sumd += x[i + j];
        print(sumd/w)
        newx[i] = sumd/w;

    return newx;




def movingAveragedistance(x,y,measure,w, **kwargs):

    y = movingAverage(y,w);
    x = movingAverage(x,w);
    try:
        m = kwargs["m"];
    except:
        m = 1;

    try:
        p = kwargs["p"];
        r = kwargs["r"];
        epsilon = ["epsilon"];
    except:
        p = 1;
        r = 1;
        epsilon = 1;

    try:
        lamb = kwargs["lamb"];
        nu = kwargs["nu"]
    except:
        lamb = 1;
        nu = 1;

    try:
        timesx = kwargs["timesx"];
        timesy = kwargs["timesy"];
    except:
        timesx = [];
        timesy = [];


    if measure == "DTW":
        result = dtw(x,y);
    elif measure == "EDR":
        result = edr(x,timesx,y,timesy,m);
    elif measure == "ERP":
        result = erp(x,y,m);
    elif measure == "LCSS":
        result = lcss(x,y,m);
    elif measure == "MSM":
        result = msm(x,y,m);
    elif measure == "TWED":
        result = twed(x,timesx,y,timesy,lamb,nu);
    elif measure == "Swale":
        result = swale(x,y,p,r,epsilon);
    elif measure == "NCC":
        result = NCC(x,y);

    return result/CID(x,y);