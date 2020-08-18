from scipy import stats
import SINK

class CorrelationProtocol:
    def __init__(self, x, y, *args):
        self.x = x
        self.y = y
        self.args = args

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
        return SINK.NCC(self.x, self.y, self.args)

correlation_protocols = {
    "Pearson": Pearson,
    "GRAIL_ED": GRAIL_ED,
    "NCC": NCC,
    "NCC_compressed": NCC_Compressed
}
