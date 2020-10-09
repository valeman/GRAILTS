import numpy as np

def SimpleSimulationVLtimeseries(n=200,lag=5,YstFixInx=110,YfnFixInx=170, XpointFixInx=100,arimaFlag=True,seedVal=-1,expflag=False,causalFlag=True):
  X = np.zeros(n + lag)
  Y = np.zeros(n + lag)

  if seedVal !=-1:
    np.random.seed(seedVal)

  rmat = np.random.normal(0, 1, n*2)
  rmat = np.reshape(rmat,(n,2))

  for i in range(n):
    # X bussiness
    if arimaFlag == False: # using normal generator
        X[i + 1] = rmat[i,0]

    else:
      X[i + 1] = 0.2 * X[i] + rmat[i,0]

    # Y bussiness
    if causalFlag == True:
      Y[i + lag] = X[i]*(not expflag) + 0.1*rmat[i,1] +np.exp(X[i])*(expflag)
    elif arimaFlag == False:
      Y[i + 1] = rmat[i,1]
    else:
      Y[i + 1] = 0.2 * Y[i] + rmat[i,1]


  Y[YstFixInx:YfnFixInx] = X[XpointFixInx]

  X = X[lag:]
  Y = Y[lag:]
  return X, Y


X,Y = SimpleSimulationVLtimeseries()
