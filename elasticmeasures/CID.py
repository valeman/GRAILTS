import numpy as np;

def CID(x,y):
    cx = np.sqrt(np.sum(np.square(np.diff(x))));
    cy = np.sqrt(np.sum(np.square(np.diff(y))));
    return (max(cx,cy)/min(cx,cy))


def CIDdistance(x,y,measure, **kwargs):

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
    elif measure == "LCSS";
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
