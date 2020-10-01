from GRAIL import GRAIL_rep
from grail_kdtw import GRAIL_rep_kdtw

#not sure I will use these classes but I might

class Representation:
    def __init__(self):
        pass

    def get_representation(self):
        raise NotImplemented


class GRAIL(Representation):

    def __init__(self, kernel = "SINK", d = 100, f = 0.99, r = 20, GV = [*range(1,21)],
                 fourier_coeff = -1, e = -1, eigenvecMatrix = None, inVa = None, gamma = None, sigma = None, initialization_method = "partition"):
        self.kernel = kernel
        self.d = d
        self.f = f
        self.r = r
        self.GV = GV
        self.fourier_coeff = fourier_coeff
        self.e = e
        self.eigenvecMatrix = eigenvecMatrix
        self.inVa = inVa
        self.initialization_method = initialization_method
        self.gamma = gamma
        self.sigma = sigma


    def get_representation(self, X):
        """
        Get the representation of matrix X
        :param X:
        :return:
        """
        if self.d > X.shape[0]:
            raise ValueError("The number of landmark series should be smaller than the number of time series.")
        if self.kernel == "SINK":
            Z_k, Zexact = GRAIL_rep(X, self.d, self.f, self.r, self.GV, self.fourier_coeff, self.e, self.eigenvecMatrix, self.inVa,
                  self.gamma, self.initialization_method)
        elif self.kernel == "kdtw":
            Z_k, Zexact = GRAIL_rep_kdtw(X, self.d, self.f, self.r, self.GV, self.sigma, self.eigenvecMatrix, self.inVa)
        return Z_k

    def get_exact_representation(self, X):
        """
        Get the representation of matrix X
        :param X:
        :return:
        """
        if self.d > X.shape[0]:
            raise ValueError("The number of landmark series should be smaller than the number of time series.")
        if self.kernel == "SINK":
            Z_k, Zexact = GRAIL_rep(X, self.d, self.f, self.r, self.GV, self.fourier_coeff, self.e, self.eigenvecMatrix, self.inVa,
                  self.gamma, self.initialization_method)
        elif self.kernel == "kdtw":
            Z_k, Zexact = GRAIL_rep_kdtw(X, self.d, self.f, self.r, self.GV, self.sigma, self.eigenvecMatrix, self.inVa)
        return Zexact