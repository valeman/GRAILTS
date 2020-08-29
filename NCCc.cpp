#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <complex.h>
#include <fftw3.h>

#define max(x, y) ((x)>(y)?(x):(y))

double* scmultiply(double* x, double* y)
{
	if (y[1] < -10000000000000000000.0)
	{
		y[1] = 0;
	}
	if (x[1] < -10000000000000000.0)
	{
		x[1] = 0;
	}
	double* result = (double*)malloc(sizeof(double) * 2);
	result[0] = x[0] * y[0] - x[1] * y[1];
	result[1] = x[1] * y[0] + x[0] * y[1];

	return  result;
}


double sdmultiply(double* x, double* y)
{
	double* result = scmultiply(x, y);
	double c = sqrt((result[0] * result[0]) + (result[1] * result[1]));
	free(result);

	return c;
}

fftw_complex* NCCc(fftw_complex* x, int xlen, fftw_complex* y, int ylen)
{
	int i;
	int len = (int)pow(2, ceil(log2(abs(2 * max(xlen, ylen) - 1))));

	fftw_complex* xi = (fftw_complex*)malloc(sizeof(fftw_complex) * len);
	fftw_complex* yi = (fftw_complex*)malloc(sizeof(fftw_complex) * len);

	fftw_complex* xo = (fftw_complex*)malloc(sizeof(fftw_complex) * len);
	fftw_complex* yo = (fftw_complex*)malloc(sizeof(fftw_complex) * len);

	for (i = 0; i < len; i++)
	{
		if (i < xlen)
		{
			xi[i][0] = x[i][0];
		}
		else
		{
			xi[i][0] = 0;
		}
		xi[i][1] = 0;
	}
	for (i = 0; i < len; i++)
	{
		if (i < ylen)
		{
			yi[i][0] = y[i][0];
		}
		else
		{
			yi[i][0] = 0;
		}
		yi[i][1] = 0;
	}

	fftw_plan p = fftw_plan_dft_1d(len, xi, xo, FFTW_FORWARD, FFTW_ESTIMATE);
	fftw_plan p2 = fftw_plan_dft_1d(len, yi, yo, FFTW_FORWARD, FFTW_ESTIMATE);

	fftw_execute(p);
	fftw_execute(p2);

	fftw_destroy_plan(p);
	fftw_destroy_plan(p2);

	fftw_complex* arr = (fftw_complex*)malloc(sizeof(fftw_complex) * len);
	fftw_complex* arro = (fftw_complex*)malloc(sizeof(fftw_complex) * len);
	for (int i = 0; i < len; i++)
	{

		arr[i][0] = sdmultiply(xo[i], yo[i]);
		arr[i][1] = 0;
	}

	fftw_plan p3 = fftw_plan_dft_1d(len, arr, arro, FFTW_BACKWARD, FFTW_ESTIMATE);
	fftw_execute(p3);
	fftw_destroy_plan(p3);
	fftw_free(xo);
	fftw_free(yo);

	for (i = 0; i < len; i++)
	{
		arro[i][0] = arro[i][0] / len;
	}
	fftw_free(arr);
	fftw_free(xi);
	fftw_free(yi);

	return arro;
}