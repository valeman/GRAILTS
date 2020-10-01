import rpy2.robjects as robjects
from rpy2.robjects.packages import importr

importr("VLTimeCausality")
simplesim = robjects.r("VLTimeCausality::SimpleSimulationVLtimeseries")
T = simplesim()
print(T.r_repr())

