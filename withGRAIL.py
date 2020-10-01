from MultipleTSsimulation import MultipleSimulationVLtimeseries
import Representation
import rpy2

grail = Representation.GRAIL(kernel="kdtw", d = 10)

TS = MultipleSimulationVLtimeseries()
TS_GRAIL = grail.get_representation(TS)
