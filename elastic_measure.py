
import math;
import numpy as np;
import warnings


""" constraint

    Creates a mask based on the Sakoe-Chiba Radius or the Itakura Max Slope.
    If constraint is None but either constraint value is set, a warning is sent and no constraint is used.
    If constraint is set but the only constraint value changed is the other one, the constraint is used with the default
    value of the constraint in the "constraint" parameter and a warning is sent. 
    (i.e. constraint(xlen,ylen,constraint = "Sakoe-Chiba",itakura_max_slope = 3)
    ------------------------------------------------------

    Inputs:

    xlen: length of time series x

    ylen: length of time series y

    constraint: the type of constraint either Sakoe-Chiba or Itakura or None
    If value is none of those then a runtime warning is sent and no constraint is used

    sakoe_chiba_radius = integer radius of Sakoe-Chiba Band
    If input is negative then a runtime warning is sent and no constraint is used.
    If input is a float, then the floor of the input is used and a warning is sent.

    itakura_max_slope
    If input is negative then a runtime warning is sent and no constraint is used.
    If input is a float, then the floor of the input is used and a warning is sent.
    ----------------------------------------------------------

    Outputs:
    A mask that depends on the constraint given.

"""

def constraints(xlen,ylen,constraint=None,sakoe_chiba_radius=1,itakura_max_slope=2):

    if constraint == None:
        if sakoe_chiba_radius != 1 or itakura_max_slope != 2:
            warnings.warn("""sakoe_chiba_radius or itakura_max_slope set but constraint does not match.
                             Since constraint is set to None, no constraint is being used.""",
                             RuntimeWarning)
        return np.zeros((xlen,ylen));
    elif constraint == "Sakoe-Chiba":
        if itakura_max_slope != 2:
           warnings.warn("""itakura_max_slope set but constraint does not match.
                             Since constraint is set to "Sakoe-Chiba", using the parameter sakoe_chiba_radius."""
                             , RuntimeWarning);
        return sakoe_chiba_mask(xlen,ylen,sakoe_chiba_radius);
    elif constraint == "Itakura":
        if sakoe_chiba_radius != 1:
            warnings.warn("""sakoe_chiba_radius set but constraint does not match.
                             Since constaint is set to "Itakura", using the parameter itakura_max_slope."""
                             , RuntimeWarning)

        return itakura_mask(xlen,ylen,itakura_max_slope);
    else:
        warnings.warn("""No permittable constraint was entered.\n
                         Defaulting to no constraint."""
                         ,RuntimeWarning)
        return np.zeros((xlen,ylen));

    


""" sakoe_chiba_mask

    Creates a mask for the dtw measure of width 2r. Sets values outside of the band to np.Inf and the rest to 0.
    All values inputted most be positive.

    Inputs:

    xlen: The length of the time series x.

    ylen: The length of time series y

    r: The radius of the sakoe-chiba band

    Output:
    An array of dimensions xlen X ylen corresponding to the dimensions of the Sakoe-Chiba Band

    Reference:
    https://github.com/tslearn-team/tslearn/blob/master/tslearn/metrics.py
"""
def sakoe_chiba_mask(xlen,ylen,r=1):

    mask = np.full((xlen,ylen),np.Inf);
    
    if (xlen > ylen):
        width = xlen - ylen + r
        for i in range(ylen):
            l = max(0, i - r);
            u = min(xlen,i + width) + 1;
            mask[l:u, i] = 0;
    else:
        width = ylen - xlen + r
        for i in range(xlen):
            l = max(0, i - r);
            u = min(ylen,i + width) + 1;
            mask[i,l:u] = 0;

    return mask;

""" itakura_mask

    Creates a mask using the itakura max slope constraint. Sets all values outside the constraint to np.Inf

    Inputs:

    xlen: length of the time series x

    ylen: length of the time series y

    m = max slope of the itakura parallelagram 

    Outputs:
    A mask of the array according to the itakura parallelagram 

    Reference:
    https://github.com/tslearn-team/tslearn/blob/master/tslearn/metrics.py

"""

def itakura_mask(xlen,ylen,m):

    #Mask to eventually return
    mask = np.full((xlen, ylen), np.Inf)

    
    min_slope = 1 / float(m)
    max_slope = m * (float(xlen) / float(ylen))
    min_slope *= (float(xlen) / float(ylen))

    lb = np.empty((2,ylen));

    lb[0] = min_slope * np.arange(ylen);
    lb[1] = ((xlen-1) - max_slope * (ylen - 1)
                        + max_slope * np.arange(ylen))

    ub = np.empty((2, ylen))
    ub[0] = max_slope * np.arange(ylen)
    ub[1] = ((xlen - 1) - min_slope * (ylen - 1)
                      + min_slope * np.arange(ylen))

    upper_bound = np.empty(ylen);
    lower_bound = np.empty(ylen);
    for i in range(ylen):
        upper_bound[i] = min(ub[0,i], ub[1,i]);
        lower_bound[i] = max(lb[0,i], lb[1,i]);

    upper_bound = np.floor(upper_bound + 1);
    lower_bound = np.ceil(lower_bound);

    for i in range(ylen):
        mask[int(lower_bound[i]): int(upper_bound[i]),i] = 0;


    check = False;

    for i in range(xlen):
        if np.all(np.isinf(mask[i])):
            check = True;
            break;
    if not check:
        for i in range(ylen):
            if np.all(np.isinf(mask[:,i])):
                check = True;
                break;

    if check:
        warnings.warn("""itakura_max_slope leads to fully infinite row or column.
                         No path possible for the given time series.
                        """, RuntimeWarning)


    return mask;


def dtw(x,y,constraint=None,sakoe_chiba_radius=1,itakura_max_slope = 2):
    
    N = constraints(len(x),len(y),constraint = constraint,
                   sakoe_chiba_radius = sakoe_chiba_radius,
                   itakura_max_slope = itakura_max_slope);


    N[0][0] = abs(x[0] - y[0]);

    for i in range(1,len(x)):
        N[i][0] = max(abs(x[i] - y[0]), N[i][0]);

    for i in range(1,len(y)):
        N[0][i] = max(abs(x[0] - y[i]),N[0][i]);


    for i in range(1,len(x)):
        for j in range(1,len(y)):
            if N[i][j] != np.Inf:
                N[i][j] = abs(x[i] - y[j]) + min(min(N[i-1][j], N[i][j-1]), N[i-1][j-1]);

    final_dtw = N[xlen-1][ylen-1];

    #Print Path

    n = xlen - 1;
    m = ylen - 1;

    path = [];

    path.append({"Coordinate": (n,m), "Value": N[n][m]})


    while m != 0 or y != 0:
        up = N[n-1][m];
        left  = N[n][m-1];
        diagonal = N[n-1][m-1];
        next = min(diagonal,up,left)

        if next == diagonal:
            n -= 1;
            m -= 1;
        elif next == up:
            m -= 1;
        else:
            n -= 1;
        path.append({"Coordinate": (n,m), "Value": N[n][m]})

    print(path);
    print(N);

    return  N[n][m];



def msm_dist(new, x, y, c):
    if ((x <= new) and (new <= y)) or ((y <= new) and (new <= x)):
        dist = c;
    else:
        dist = c + min(abs(new - x), abs(new - y))

    return dist;

def msm(x,y,c):

    cost[xlen][ylen];

    cost[0][0] = abs(x[i] - y[i]);

    for i in range(1,len(x)):
        cost[i][0] = cost[i-1][0] + msm_dist(x[i],x[i-1],y[0],c);

    for i in range(1,len(y)):
        cost[0][i] = cost[0][i-1] + msm_dist(y[i], x[0],y[i-1],c);

    for i in range(2,xlen):
        for j in range(2,ylen):
            cost[i][j] = min(cost[i-1][j-1] + abs(x[i] - y[j]),
                            cost[i-1][j] + msm_dist(x[i], x[i -1],y[j],c),
                            cost[i][j-1] + msm_dist(y[j], x[i], y[j-1],c));

    return cost[xlen-1][ylen-1];


def twed(x,timesx,y,timesy,lamb,nu):

    x.insert(0,0);
    y.insert(0,0);
    timesx.insert(0,0);
    timesy.insert(0,0);

    dp[len(x)][len(y)];

    dp[:][0] = INF;
    dp[0][:] = INF;
    dp[0][0] = 0;
    for i in range(0,xlen):
        for j in range(0,ylen):
             c1 = dp[i - 1][j] + math.sqrt(dist(x[i - 1], x[i])) + nu * (timesx[i] - timesx[i - 1]) + lamb;
             c2 = dp[i][j - 1] + math.sqrt(dist(y[j - 1], y[j])) + nu * (timesy[j] - timesy[j - 1]) + lamb;
             c3 = dp[i - 1][j - 1] + math.sqrt(dist(x[i], y[j])) + math.sqrt(dist(x[i - 1], y[j - 1])) + nu * (fabs(timesx[i] - timesy[j]) + fabs(timesx[i - 1] - timesy[j - 1]));

             dp[i][j] = min(c1,c2,c3);

    return dp[xlen-1][ylen-1];


def erp(x,y,m):
    df[len(x) + 1][len(y) + 1];

    df[0][0] = 0;

    for i in range(1,len(y)+1):
        df[0][i] = df[0][i-1] - pow(y[i] - m, 2);

    for i in range(1,len(x)+1):
        df[i][0] = df[i-1][0] - pow(y[i] - m, 2);

    for i in range(1,len(x) + 1):
        for j in range(1,len(y) + 1):
            df[i][j] = max(df[i - 1][j-1] - pow(x[i] - y[j],2), df[i][j-1] - pow(x[i] - m,2), df[i-1][j] - pow(y[j] - m, 2))

    return math.sqrt(0 - df[xlen][ylen]);



def edr(x,y,m):

    df[len(x[0])][len(y[0])];

    for i in range(len(x[0])):
        df[i][0] = -i;

    for j in range(len(y[0])):
        df[0][i] = -1;

    for i in range(len(x[0])):
        for j in range(len(y[0])):
            if ((abs(x[0][i] - y[0][i]) <= m) and (abs(x[1][i] - y[1][i]) <= m)):
                s1 = 0;
            else:
                s1 = -1;


            df[i][j] = max(df[i-1][j-1] + s1,df[i][j-1]-1,df[i-1][j]-1);


    return 0 - df[len(x[0]) - 1][len(y[0])-1];


def lcss(x,y,epsilon):
    lcss[len(x)][len(y)];

    for i in range(len(x)):
        for j in range(len(y)):
            if (i + j == 0):
                cost = 0;
            else:
                if (i == 0):
                    cost = lcss[i][j-1]
                elif j == 0:
                    cost = lcss[i-1][j];
                elif (x[i] - epsilon <= y[j] and x[i] + epsilon >= y[j]):
                    cost = lcss[i-1][j-1] + 1;
                elif (lcss[i - 1][j] > lcss[i][j - 1]):
                    cost = lcss[i-1][j];
                else:
                    cost = lcss[i][j-1];
            lcss[i][j] = cost;

    return 1 - result/len(x);


def swale(x,y,p,r,epsilon):
    df[len(x)][len(y)];

    for i in range(len(y)):
        df[0][i] = i * p;
    for i in range(len(x)):
        df[0][i] = i * p;

    for i in range(1,len(x)):
        for j in range(1,len(y)):
            if (abs(x[i] - y[i]) <= epsilon):
                df[i][j] = df[i-1][j-1] + r;
            else:
               df[i][j] = max(df[i][j-1], df[i-1][j]) + p;

    df[len(x)-1][len(y)-1];




x = [1,2,3,4];
y = [2,4,6,8];
xt = [1,2,3,4];
yt = [1,2,3,4];

print(dtw(x,y));
print(msm(x,y,.5));
print(twed(x,xt,y,yt,.5,.5));
print(erp(x,y,.5));
print(edr(x,y,.5));
print(lcss(x,y,.5));
print(swale(x,y,.11,-.5,2));

    
