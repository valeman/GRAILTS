#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>


#define min(x, y) ((x)<(y)?(x):(y))
#define max(x, y) ((x)>(y)?(x):(y))
#define dist(x, y)((x-y)*(x-y))

#define INF 1e20



/* dtw

	Computes the dtw distance between two time series

	Inputs:

	double * x: the first pointer to a double array corresponding to a time series

	int xlen: length of the first time series;

	 double * y: the first pointer to a double array corresponding to a time series

	int ylen: length of the second time series;

	char* constraint: the type of constraint to be used
	Valid options are "itakura", "sakoe-chiba", and "None"

	int sakoe_chiba_radius: radius of the band to be used, must be positive and an integer

	int itakura_max_slope:

	Output:

	double ** mask: a pointer corresponding to a double* array

*/

double dtw(double* x, int xlen, double* y, int ylen, const char* constraint, int w)
{
	if (strcmp(constraint,"None"))
	{
		return dtw_n(x, xlen, y, ylen);
	}
	else if (strcmp(constraint, "Sakoe-Chiba"))
	{
		return dtw_scb(x, xlen, y, ylen,w);
	}
	else if (strcmp(constraint, "Sakoe-Chiba"))
	{
		return dtw_ip(x, xlen, y, ylen, w);
	}
	else
	{
		fprintf(stderr, "constraint did not match any of the acceptable values ('None', 'Sakoe-Chiba', 'Itakura'), defaulting to 'None'");
		return dtw_n(x, xlen, y, ylen);
	}
	

}

double dtw_n(double* x, int xlen, double* y, int ylen)
{
	double** arr;
	int i, j;
	arr = (double**)malloc(sizeof(double*) * xlen);
	for (i = 0; i < xlen; i++)
	{
		arr[i] = (double*)malloc(ylen * sizeof(double));
		for (j = 0; j < ylen; j++)
		{
			arr[i][j] = 0.0;
		}
	}
	arr[0][0] = pow(x[0] - y[0],2);


	for (i = 1; i < xlen; i++)
	{
		arr[i][0] = pow(fabs(x[i] - y[0]), 2) + arr[i - 1][0];
	}

	for (i = 1; i < ylen; i++)
	{
		arr[0][i] = pow(x[0] - y[i], 2) + arr[0][i - 1];
	}

	for (i = 1; i < xlen; i++)
	{
		for (j = 1; j < ylen; j++)
		{
			if (arr[i][j] != INF)
			{
				arr[i][j] = pow(x[i] - y[j], 2) + min(min(arr[i - 1][j], arr[i][j - 1]), arr[i - 1][j - 1]);
			}
		}
	}

	double final_dtw = arr[xlen - 1][ylen - 1];

	for (i = 0; i < xlen; i++)
	{
		free(arr[i]);
	}
	free(arr);

	return pow(final_dtw, 1 / 2);
}


double dtw_scb(double* x, int xlen, double* y, int ylen, int w)
{

	double* prev = (double*)malloc(sizeof(double*) * ylen);
	double* cur = (double*)malloc(sizeof(double*) * ylen);
	double* temp;
	int i, j;
	for (i = 0; i < xlen; i++)
	{
		temp = prev;
		prev = cur;
		cur = temp;
		for (j = max(0, i - w); j < min(ylen, i + w); j++)
		{
			if (i + j == 0)
			{
				cur[j] = dist(x[i], y[j]);
			}
			else if (i == 0)
			{
				cur[j] = dist(x[0], y[j]) + cur[j - 1];
			}
			else if (j == 0)
			{
				cur[j] = dist(x[i], y[0]) + prev[j];
			}
			else
			{
				cur[j] = dist(x[i], y[j]) + min(prev[j - 1], min(prev[j], cur[j - 1]));
			}
		}


	}

	double result = cur[ylen - 1];
	free(cur);
	free(prev);

	return result;
}

double dtw_ip(double* x, int xlen, double* y, int ylen, int s)
{

	double* prev, * cur, * temp;
	cur = (double*)malloc(ylen * sizeof(double));
	prev = (double*)malloc(ylen * sizeof(double));
	int i, j;
	double s1, s2, s3;

	double min_slope = 1 / s * (double)xlen / (double)ylen;
	double max_slope = s * (double)xlen / (double)ylen;

	min_slope* i;
	(double)(((double)ylen - 1) - max_slope * ((double)xlen - 1))
		+ max_slope * i;

	max_slope* i;
	(((double)ylen - 1) - min_slope * ((double)xlen - 1))
		+ min_slope * i;

	for (i = 0; i < xlen; i++)
	{
		temp = prev;
		prev = cur;
		cur = temp;
		int minw = max(min_slope * i,
			(double)(((double)ylen - 1) - max_slope * ((double)xlen - 1))
			+ max_slope * i);
		int maxw = min(max_slope * i,
			(((double)ylen - 1) - min_slope * ((double)xlen - 1))
			+ min_slope * i);
		for (j = minw; j < maxw; j++)
		{
			if (i + j == 0)
			{
				cur[j] = dist(x[i], y[j]);
			}
			else if (i == 0)
			{
				cur[j] = dist(x[0], y[j]) + cur[j - 1];
			}
			else if (j == 0)
			{
				cur[j] = dist(x[i], y[0]) + prev[j];
			}
			else
			{
				cur[j] = dist(x[i], y[j]) + min(prev[j - 1], min(prev[j], cur[j - 1]));
			}
		}


	}

	double result = cur[ylen - 1];
	free(cur);
	free(prev);

	return result;
}