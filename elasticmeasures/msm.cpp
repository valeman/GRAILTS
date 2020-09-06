#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>


#define min(x, y) ((x)<(y)?(x):(y))
#define max(x, y) ((x)>(y)?(x):(y))
#define dist(x, y)((x-y)*(x-y))

#define INF 1e20

double msm(double* x, int xlen, double* y, int ylen, const char* constraint, double c,int w)
{
	if (strcmp(constraint, "None"))
	{
		return msm_n(x, xlen, y, ylen,c);
	}
	else if (strcmp(constraint, "Sakoe-Chiba"))
	{
		return msm_scb(x, xlen, y, ylen,c, w);
	}
	else if (strcmp(constraint, "Sakoe-Chiba"))
	{
		return msm_ip(x, xlen, y, ylen,c, w);
	}
	else
	{
		fprintf(stderr, "constraint did not match any of the acceptable values ('None', 'Sakoe-Chiba', 'Itakura'), defaulting to 'None'");
		return msm_n(x, xlen, y, ylen,c);
	}


}

double msm_dist(double new_point, double x, double y, double c)
{
	double dist = 0;
	if (((x <= new_point) && (new_point <= y)) || ((y <= new_point) && (new_point <= x)))
	{
		dist = c;
	}
	else
	{
		dist = c + min(fabs(new_point - x), fabs(new_point - y));
	}
	return dist;
}

/* msm

	Computes the msm distance between two time series

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
double msm_n(double* x, int xlen, double* y, int ylen, double c)
{
	double** cost = (double**)malloc(sizeof(double*) * xlen);
	int i, j;

	for (i = 0; i < xlen; i++)
	{
		cost[i] = (double*)calloc(ylen, sizeof(double));
	}


	cost[0][0] = fabs(x[0] - y[0]);

	for (i = 1; i < xlen; i++)
	{
		cost[i][0] = cost[i - 1][0] + msm_dist(x[i], x[i - 1], y[0], c);
	}
	for (i = 1; i < ylen; i++)
	{
		cost[0][i] = cost[0][i - 1] + msm_dist(y[i], x[0], y[i - 1], c);
	}

	for (i = 1; i < xlen; i++)
	{
		for (j = 1; j < ylen; j++)
		{
			double d1 = cost[i - 1][j - 1] + fabs(x[i] - y[j]);
			double d2 = cost[i - 1][j] + msm_dist(x[i], x[i - 1], y[j], c);
			double d3 = cost[i][j - 1] + msm_dist(y[j], x[i], y[j - 1], c);
			cost[i][j] = min(min(d1, d2), d3);
		}
	}
	double result = cost[xlen - 2][ylen - 2];
	for (i = 0; i < xlen; i++)
	{
		free(cost[i]);
	}
	free(cost);
	return result;
}

double msm_scb(double* x, int xlen, double* y, int ylen, double c,int w)
{
	double* prev, * cur, * temp;
	cur = (double*)malloc(ylen * sizeof(double));
	prev = (double*)malloc(ylen * sizeof(double));
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
				cur[0] = fabs(x[0] - y[0]);
			}
			else if (j == 0)
			{
				cur[j] = prev[0] + msm_dist(x[i], x[i - 1], y[0], c);
			}
			else if (i == 0)
			{
				cur[j] = cur[j - 1] + msm_dist(y[j], x[0], y[j - 1], c);
			}
			else
			{

				double d1 = prev[j - 1] + fabs(x[i] - y[j]);
				double d2 = prev[j] + msm_dist(x[i], x[i - 1], y[j], c);
				double d3 = cur[j - 1] + msm_dist(y[j], x[i], y[j - 1], c);
				cur[j] = min(min(d1, d2), d3);
			}
		}
	}
	double result = cur[ylen - 1];
	free(cur);
	free(prev);
	return result;
}

double msm_ip(double* x, int xlen, double* y, int ylen, double c, int s)
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
				cur[0] = fabs(x[0] - y[0]);
			}
			else if (j == 0)
			{
				cur[j] = prev[0] + msm_dist(x[i], x[i - 1], y[0], c);
			}
			else if (i == 0)
			{
				cur[j] = cur[j - 1] + msm_dist(y[j], x[0], y[j - 1], c);
			}
			else
			{

				double d1 = prev[j - 1] + fabs(x[i] - y[j]);
				double d2 = prev[j] + msm_dist(x[i], x[i - 1], y[j], c);
				double d3 = cur[j - 1] + msm_dist(y[j], x[i], y[j - 1], c);
				cur[j] = min(min(d1, d2), d3);
			}
		}
	}
	double result = cur[ylen - 1];
	free(cur);
	free(prev);
	return result;
}