#include <stdlib.h>
#include <stdio.h>
#include <math.h>


#define min(x, y) ((x)<(y)?(x):(y))
#define max(x, y) ((x)>(y)?(x):(y))
#define dist(x, y)((x-y)*(x-y))



double dtw_scb(double* x, int xlen, double* y, int ylen, int w)
{

	double* prev = (double*) malloc(sizeof(double*) * ylen);
	double* cur = (double*) malloc(sizeof(double*) * ylen);
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

