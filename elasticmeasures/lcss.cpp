#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>


#define min(x, y) ((x)<(y)?(x):(y))
#define max(x, y) ((x)>(y)?(x):(y))
#define dist(x, y)((x-y)*(x-y))

#define INF 1e20

double lcss(double* x, int xlen, double* y, int ylen, const char* constraint,double epsilon, int w)
{
	if (strcmp(constraint, "None"))
	{
		return lcss_n(x, xlen, y, ylen,epsilon);
	}
	else if (strcmp(constraint, "Sakoe-Chiba"))
	{
		return lcss_scb(x, xlen, y, ylen,epsilon,w);
	}
	else if (strcmp(constraint, "Sakoe-Chiba"))
	{
		return lcss_ip(x, xlen, y, ylen,epsilon, w);
	}
	else
	{
		fprintf(stderr, "constraint did not match any of the acceptable values ('None', 'Sakoe-Chiba', 'Itakura'), defaulting to 'None'");
		return lcss_n(x, xlen, y, ylen,epsilon);
	}


}

double lcss_n(double* x, int xlen, double* y, int ylen, double epsilon)
{
	double** lcss = (double**)malloc(xlen * sizeof(double*));
	double cost;
	int i, j;

	for (i = 0; i < xlen; i++)
	{
		lcss[i] = (double*)calloc(ylen, sizeof(double));
	}

	for (i = 0; i < xlen; i++)
	{
		for (j = 0; j < ylen; j++)
		{
			if (i + j == 0)
			{
				if ((x[i] - epsilon <= y[j]) && (x[i] + epsilon >= y[j]))
				{
					cost = 1;
				}
				else
				{
					cost = 0;
				}
			}
			else
			{
				if (i == 0)
				{
					cost = lcss[i][j - 1];
				}
				else if (j == 0)
				{
					cost = lcss[i - 1][j];
				}
				else if ((x[i] - epsilon <= y[j]) && (x[i] + epsilon >= y[j]))
				{
					cost = lcss[i - 1][j - 1] + 1;
				}
				else if (lcss[i - 1][j] > lcss[i][j - 1])
				{
					cost = lcss[i - 1][j];
				}
				else if (lcss[i][j - 1] > lcss[i - 1][j])
				{
					cost = lcss[i][j - 1];
				}
			}
			lcss[i][j] = cost;
		}
	}

	double result = lcss[xlen - 1][ylen - 1];
	for (i = 0; i < xlen; i++)
	{
		free(lcss[i]);
	}
	free(lcss);
	return 1 - result / xlen;
}

double lcss_scb(double* x, int xlen, double* y, int ylen, double epsilon,int w)
{

	double* prev, * cur, * temp;
	cur = (double*)malloc(ylen * sizeof(double));
	prev = (double*)malloc(ylen * sizeof(double));
	double cost;
	int i, j;

	for (i = 0; i < xlen; i++)
	{
		temp = prev;
		prev = cur;
		cur = temp;
		int minw = max(0, i - w);
		int maxw = min(ylen, i + w);
		for (j = minw; j < maxw; j++)
		{
			if (i + j == 0)
			{
				if ((x[i] - epsilon <= y[j]) && (x[i] + epsilon >= y[j]))
				{
					cost = 1;
				}
				else
				{
					cost = 0;
				}
			}
			else
			{
				if (i == 0)
				{
					cost = cur[j - 1];
				}
				else if (j == 0)
				{
					cost = prev[j];
				}
				else if ((x[i] - epsilon <= y[j]) && (x[i] + epsilon >= y[j]))
				{
					cost = prev[j-1] + 1;
				}
				else if (prev[j] > cur[j - 1])
				{
					cost = prev[j];
				}
				else if (cur[j - 1] > prev[j])
				{
					cost = cur[j - 1];
				}
			}
			cur[j] = cost;
		}
	}

	double result = cur[ylen - 1];

	free(cur);
	free(prev);
	return 1 - result / xlen;
}

double lcss_ip(double* x, int xlen, double* y, int ylen, double epsilon, int s)
{
	double cost;
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
				if ((x[i] - epsilon <= y[j]) && (x[i] + epsilon >= y[j]))
				{
					cost = 1;
				}
				else
				{
					cost = 0;
				}
			}
			else
			{
				if (i == 0)
				{
					cost = cur[j - 1];
				}
				else if (j == 0)
				{
					cost = prev[j];
				}
				else if ((x[i] - epsilon <= y[j]) && (x[i] + epsilon >= y[j]))
				{
					cost = prev[j - 1] + 1;
				}
				else if (prev[j] > cur[j - 1])
				{
					cost = prev[j];
				}
				else if (cur[j - 1] > prev[j])
				{
					cost = cur[j - 1];
				}
			}
			cur[j] = cost;
		}
	}

	double result = cur[ylen - 1];

	free(cur);
	free(prev);
	return 1 - result / xlen;
}