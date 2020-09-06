#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define min(x, y) ((x)<(y)?(x):(y))
#define max(x, y) ((x)>(y)?(x):(y))
#define dist(x, y)((x-y)*(x-y))

double dtw_ip(double* x, int xlen, double* y, int ylen, int w)
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