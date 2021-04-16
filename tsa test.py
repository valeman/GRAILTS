from statsmodels.tsa.stattools import grangercausalitytests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_process import arma_generate_sample



#Create Time series
t1 = [0.1*np.random.normal()]
for _ in range(100):
    t1.append(0.5*t1[-1] + 0.1*np.random.normal())

t2 = [item + 0.1*np.random.normal() for item in t1]


#Make a lag
t1 = t1[3:]
t2 = t2[:-3]

# plt.figure()
# plt.plot(t1, color = "b")
# plt.plot(t2, color = "r")
# plt.legend(["t1", "t2"])

df = pd.DataFrame(columns = ["t2", "t1"], data = zip(t2,t1))

res = grangercausalitytests(df, 3)
print(res[3][0]['ssr_ftest'][1])



