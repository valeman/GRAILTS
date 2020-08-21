import GRAIL

#not sure I will use these classes but I might

class Representation:
    def __init__(self):
        pass

    def get_representation(self):
        raise NotImplemented


class GRAIL(Representation):

    def __init__(self, d, f, r, GV, fourier_coeff = -1, e = -1, eigenvecMatrix = None, inVa = None):
        self.d = d
        self.f = f
        self.r = r
        self.GV = GV
        self.fourier_coeff = fourier_coeff
        self.e = e
        self.eigenvecMatrix = eigenvecMatrix
        self.inVa = inVa


    @classmethod
    def get_representation(self, X):
        """
        Get the representation of matrix X
        :param X:
        :return:
        """
        return GRAIL.GRAIL(X,self.d,self.f,self.r,self.GV,self.fourier_coeff,self.e, self.eigenvecMatrix, self.inVA)