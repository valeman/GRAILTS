from cpython cimport array
import array

cdef extern from "headers.h":
	void hello(double *v, int size)
	
cdef array.array a = array.array('d', [1, 2, 3])
cdef double[:] ca = a

hello(&ca[0], 3)