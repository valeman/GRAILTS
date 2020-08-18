import numpy as np
import Correlation
import SINK


# matrix = TimeSeries.load("csvtest.csv", "UCR")
# correlation = Correlation.Correlation(matrix[0,:], matrix[1,:], correlation_protocol_name="Pearson")
# print(correlation.correlate())

x = np.array([1,2,3,4,5])
y = np.array([3,4,5,6,7])

print(SINK.NCC(x,y,e=.5))


correlation = Correlation.Correlation(x,y, correlation_protocol_name="NCC_compressed", e = .5)
print(correlation.correlate())