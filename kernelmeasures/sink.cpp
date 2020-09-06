#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <complex.h>
#include <fftw3.h>

#define min(x, y) ((x)<(y)?(x):(y))
#define max(x, y) ((x)>(y)?(x):(y))
#define dist(x, y)((x-y)*(x-y))

double* cmultiply(double* x, double* y)
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


double dmultiply(double* x, double* y)
{
	double* result = cmultiply(x, y);
	double c = sqrt((result[0] * result[0]) + (result[1] * result[1]));
	free(result);

	return c;
}

double dfnorm(fftw_complex* x, int xlen)
{
	int i;
	double sum = 0;
	for (i = 0; i < xlen; i++)
	{
		sum += pow(x[i][0], 2);
	}
	return sqrt(sum);
}


double sink(fftw_complex* x, int xlen, fftw_complex* y, int ylen, double gamma)
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

	double normp = dfnorm(x, xlen);
	normp *= dfnorm(y, ylen);

	fftw_complex* arr = (fftw_complex*)malloc(sizeof(fftw_complex) * len);
	fftw_complex* arro = (fftw_complex*)malloc(sizeof(fftw_complex) * len);
	for (int i = 0; i < len; i++)
	{

		arr[i][0] = dmultiply(xo[i], yo[i]);
		arr[i][1] = 0;
	}

	fftw_plan p3 = fftw_plan_dft_1d(len, arr, arro, FFTW_BACKWARD, FFTW_ESTIMATE);
	fftw_execute(p3);
	fftw_destroy_plan(p3);
	fftw_free(xo);
	fftw_free(yo);

	double result = 0;

	for (i = 0; i < len; i++)
	{
		arro[i][0] = arro[i][0] / len;
	}

	for (i = len - max(xlen, ylen) + 1; i < len; i++)
	{
		result += exp(gamma * arro[i][0] / (normp));
	}

	for (i = 0; i < max(xlen, ylen); i++)
	{
		result += exp(gamma * arro[i][0] / normp);
	}


	fftw_free(xi);
	fftw_free(yi);
	fftw_free(arr);
	fftw_free(arro);

	return result;

}