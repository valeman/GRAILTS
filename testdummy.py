import numpy as np
import Correlation
import SINK


# matrix = TimeSeries.load("csvtest.csv", "UCR")
# correlation = Correlation.Correlation(matrix[0,:], matrix[1,:], correlation_protocol_name="Pearson")
# print(correlation.correlate())


x = np.array([[1,2,3,4],[1,2,3,4]])
print(x[:, np.array([0,2])])