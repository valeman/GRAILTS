from GRAIL import GRAIL_rep

#not sure I will use these classes but I might

class Representation:
    def __init__(self):
        pass

    def get_representation(self):
        raise NotImplemented


class GRAIL(Representation):

    def __init__(self, d = 100, f = 0.99, r = 20, GV = [*range(1,21)], fourier_coeff = -1, e = -1, eigenvecMatrix = None, inVa = None, gamma = None, initialization_method = "partition"):
        self.d = d
        self.f = f
        self.r = r
        self.GV = GV
        self.fourier_coeff = fourier_coeff
        self.e = e
        self.eigenvecMatrix = eigenvecMatrix
        self.inVa = inVa
        self.initialization_method = initialization_method
        self.gamma = None


    def get_representation(self, X):
        """
        Get the representation of matrix X
        :param X:
        :return:
        """
        if self.d > X.shape[0]:
            raise ValueError("The number of landmark series should be smaller than the number of time series.")
        return GRAIL_rep(X,self.d,self.f,self.r,self.GV,self.fourier_coeff,self.e, self.eigenvecMatrix, self.inVa, self.gamma, self.initialization_method)