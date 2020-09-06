import numpy as np
cimport numpy as np


cpdef lcss(np.ndarray x,np.ndarray y, int delta, double epsilon):

	return lcssc(x,y,delta,epsilon);

cdef lcssc(np.ndarray x,np.ndarray y, int delta, double epsilon):

	arr = np.zeros((len(x),len(y)));
	cdef int i = 0;
	cdef int j = 0;
	cdef double cost = 0;
	cdef int wmin = 0;
	cdef int wmax = 0;
	for i in range(len(x)):
		wmin = max(0,i-delta);
		wmax = min(len(y),i+delta);
		for j in range(int(wmin),int(wmax)):
			if (i + j == 0):
				cost = 0;
			else:
				if (i == 0):
					cost = arr[i][j-1]
				elif j == 0:
					cost = arr[i-1][j];
				elif (x[i] - epsilon <= y[j] and x[i] + epsilon >= y[j]):
					cost = arr[i-1][j-1] + 1;
				elif (arr[i - 1][j] > arr[i][j - 1]):
					cost = arr[i-1][j];
				else:
					cost = arr[i][j-1];
		arr[i][j] = cost;

	result = arr[len(x)-1][len(y)-1];

	return 1 - result/min(len(x),len(y));