def lower_b(t,w,g):

  b = np.zeros(len(t));
  for i in range(len(t)):
    b[i] = min(t[max(0,i-w):min(len(t),i+w)]) - g;
  
  return b;

def upper_b(t,w,g):

  b = np.zeros(len(t));
  for i in range(len(t)):
    b[i] = max(t[max(0,i-w):min(len(t),i+w)]) + g;
  
  return b;


def lb_keogh(x,u,l):
  sumd = 0
  for i in range(len(x)):
    if x[i] > u[i]:
      sumd += (x[i] - u[i]) ** 2;
    if x[i] < l[i]:
      sumd += (x[i] - l[i]) ** 2;
    
  return sumd ** (1/2);