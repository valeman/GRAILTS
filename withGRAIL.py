from MultipleTSsimulation import MultipleSimulationVLtimeseries
import Representation

grail = Representation.GRAIL(kernel="kdtw")

TS = MultipleSimulationVLtimeseries()
TS_GRAIL = grail.get_representation(TS)
