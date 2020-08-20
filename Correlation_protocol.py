from scipy import stats
import SINK

class CorrelationProtocol:
    def __init__(self, x, y, **kwargs):
        self.x = x
        self.y = y
        self.kwargs = kwargs

    def execute(self):
        raise NotImplemented

class Pearson(CorrelationProtocol):
    def execute(self):
        return stats.pearsonr(self.x, self.y)[0]


class GRAIL_ED(CorrelationProtocol):
    pass

class NCC(CorrelationProtocol):
    def execute(self):
        return SINK.NCC(self.x, self.y)

class NCC_Compressed(CorrelationProtocol):
    def execute(self):
        return SINK.NCC(self.x, self.y, **self.kwargs)

class SINK_protocol(CorrelationProtocol):
    def execute(self):
        #check if compressed here
        return SINK.SINK(self.x, self.y, **self.kwargs)

class SINK_compressed(CorrelationProtocol):
    def execute(self):
        return SINK.SINK(self.x, self.y, **self.kwargs)

correlation_protocols = {
    "Pearson": Pearson,
    "GRAIL_ED": GRAIL_ED,
    "NCC": NCC,
    "NCC_compressed": NCC_Compressed,
    "SINK": SINK_protocol,
    "SINK_compressed": SINK_compressed
}
