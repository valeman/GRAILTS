from cpython cimport array
import array

cdef extern from "headers.h":
	double kdtw(double* x, int xlen, double* y, int ylen, double sigma)
	
cdef array.array x = array.array('d', [1, 2, 3])
cdef array.array y = array.array('d', [2,3,4])

cdef double[:] xa = x
cdef double[:] ya  = y

print(kdtw(&xa[0], 3, &ya[0], 3, 1))