import numpy as np;

def lower_b(t,w):

  b = np.zeros(len(t));
  for i in range(len(t)):
    b[i] = min(t[max(0,i-w):min(len(t),i+w)]);
  
  return b;

def upper_b(t,w):

  b = np.zeros(len(t));
  for i in range(len(t)):
    b[i] = max(t[max(0,i-w):min(len(t),i+w)]);
  
  return b;
    

def lb_keogh(x,u,l):
  sumd = 0
  for i in range(len(x)):
    if x[i] > u[i]:
      sumd += (x[i] - u[i]) ** 2;
    if x[i] < l[i]:
      sumd += (x[i] - l[i]) ** 2;
    
  return sumd ** (1/2);


def lbk_improved(x,y,w):
    h = []
    l = lower_b(y,w);
    u = upper_b(y,w)
    for i in range(len(y)):
        if x[i] <= l[i]:
            h.append(l[i]);
        elif x[i] >= u[i]:
            h.append(u[i]);
        else:
            h.append(x[i]);

    upper_h = upper_b(h,w);
    lower_h = lower_b(h,w);

    return lb_keogh(x,u,l) + lb_keogh(y,upper_h,lower_h);