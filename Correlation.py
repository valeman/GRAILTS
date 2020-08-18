import numpy as np
import csv
import Correlation_protocol
import exceptions


class Correlation:

    def __init__(self, x, y, correlation_protocol_name="NCC", time_window=None, *args):
        """

        :param x: time series. Can be either a path to a csv file or list
        :param y: time series. Can be either a path to a csv file or list
        :param correlation_algorithm:
        :param time_window: the time window in which the series are being correlated.
        :param args: arguments for the correlation algorithm.
        """

        self.x = self.load(x)
        self.y = self.load(y)
        self.correlation_protocol_name = correlation_protocol_name
        try:
            self.protocol_class = Correlation_protocol.correlation_protocols[self.correlation_protocol_name]
        except KeyError:
            raise exceptions.ProtocolNotFound("Correlation Protocol" + correlation_protocol_name + "not found")

        if args:
            self.protocol = self.protocol_class(x, y, args)
        else:
            self.protocol = self.protocol_class(x,y)

        self.check_entries()
        if time_window:
            self.time_window = time_window
            x = self.crop(x, time_window)
            y = self.crop(y, time_window)



    def check_entries(self):
        if hasattr(self, "time_window"):
            if (not isinstance(self.time_window, list)) or (len(self.time_window) != 2):
                raise TypeError("time window should be a list of 2 elements.")

    def load(self, x):
        if isinstance(x, list):
            return np.array(x)
        elif isinstance(x, np.generic) or isinstance(x, np.ndarray):
            pass
        else:
            raise TypeError("Time series should be a list, or a numpy array")

    def crop(self, x, window):
        return x[window[0]:window[1]]

    def correlate(self):
        return self.protocol.execute()


