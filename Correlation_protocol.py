from scipy import stats
import SINK
import numpy as np


class CorrelationProtocol:
    def __init__(self, x, y, **kwargs):
        self.x = x
        self.y = y
        self.kwargs = kwargs

    def execute(self):
        raise NotImplemented


class Pearson(CorrelationProtocol):
    def __init__(self, x, y, **kwargs):
        super().__init__(x, y, **kwargs)
        self.similarity = True

    def execute(self):
        return stats.pearsonr(self.x, self.y)[0]


class ED(CorrelationProtocol):

    def __init__(self, x, y, **kwargs):
        super().__init__(x, y, **kwargs)
        self.similarity = False

    def execute(self):
        return np.sum(np.power(self.x - self.y, 2))


class NCC(CorrelationProtocol):
    def __init__(self, x, y, **kwargs):
        super().__init__(x, y, **kwargs)
        self.similarity = True

    def execute(self):
        return SINK.NCC(self.x, self.y)


class NCC_Compressed(CorrelationProtocol):
    def __init__(self, x, y, **kwargs):
        super().__init__(x, y, **kwargs)
        self.similarity = True

    def execute(self):
        return SINK.NCC(self.x, self.y, **self.kwargs)


class SINK_protocol(CorrelationProtocol):
    def __init__(self, x, y, **kwargs):
        super().__init__(x, y, **kwargs)
        self.similarity = True

    def execute(self):
        # check if compressed here
        return SINK.SINK(self.x, self.y, **self.kwargs)


class SINK_compressed(CorrelationProtocol):
    def __init__(self, x, y, **kwargs):
        super().__init__(x, y, **kwargs)
        self.similarity = True

    def execute(self):
        return SINK.SINK(self.x, self.y, **self.kwargs)


correlation_protocols = {
    "Pearson": Pearson,
    "ED": ED,
    "NCC": NCC,
    "NCC_compressed": NCC_Compressed,
    "SINK": SINK_protocol,
    "SINK_compressed": SINK_compressed
}
