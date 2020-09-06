#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>


#define min(x, y) ((x)<(y)?(x):(y))
#define max(x, y) ((x)>(y)?(x):(y))
#define dist(x, y)((x-y)*(x-y))

#define INF 1e20

double twed(double* x, double* timesx, int xlen, 
	double* y, double * timesy, int ylen,
	double nu, double lambda,
	const char* constraint, int w)
{
	if (strcmp(constraint,"None"))
	{
		return twed_n(x,timesx, xlen, y,timesy, ylen,lambda,nu);
	}
	else if (strcmp(constraint, "Sakoe-Chiba"))
	{
		return twed_scb(x,timesx, xlen, y,timesy, ylen,lambda,nu,w);
	}
	else if (strcmp(constraint, "Sakoe-Chiba"))
	{
		return twed_ip(x,timesx, xlen, y,timesy, ylen,lambda,nu,w);
	}
	else
	{
		fprintf(stderr, "constraint did not match any of the acceptable values ('None', 'Sakoe-Chiba', 'Itakura'), defaulting to 'None'");
		return twed_n(x,timesx, xlen, y,timesy, ylen,lambda,nu);
	}
	

}

double twed_n(double x[], double timesx[], int xlen,
	double* y, double* timesy, int ylen,
	double lambda, double nu)
{

	double* xp = (double*)malloc(sizeof(double) * (xlen + 1));
	double* yp = (double*)malloc(sizeof(double) * (xlen + 1));
	double* timesxp = (double*)malloc(sizeof(double) * (xlen + 1));
	double* timesyp = (double*)malloc(sizeof(double) * (xlen + 1));

	int i, j;

	xp[0] = 0;
	yp[0] = 0;
	timesxp[0] = 0;
	timesyp[0] = 0;

	for (i = 1; i < xlen + 1; i++)
	{
		xp[i] = x[i - 1];
		timesxp[i] = timesx[i - 1];
	}

	for (i = 1; i < ylen + 1; i++)
	{
		yp[i] = y[i - 1];
		timesyp[i] = timesy[i - 1];
	}

	x = xp;
	y = yp;
	timesx = timesxp;
	timesy = timesyp;

	xlen = xlen + 1;
	ylen = ylen + 1;

	double** dp = (double**)malloc(sizeof(double*) * xlen);

	for (i = 0; i < xlen; i++)
	{
		dp[i] = (double*)calloc(ylen, sizeof(double));
		dp[i][0] = INF;
	}
	for (i = 0; i < ylen; i++)
	{
		dp[0][i] = INF;
	}
	dp[0][0] = 0;

	for (i = 1; i < xlen; i++)
	{
		for (j = 1; j < ylen; j++)
		{
			double c1 = dp[i - 1][j] + sqrt(dist(x[i - 1], x[i])) + nu * (timesx[i] - timesx[i - 1]) + lambda;
			double c2 = dp[i][j - 1] + sqrt(dist(y[j - 1], y[j])) + nu * (timesy[j] - timesy[j - 1]) + lambda;
			double c3 = dp[i - 1][j - 1] + sqrt(dist(x[i], y[j])) +
				sqrt(dist(x[i - 1], y[j - 1])) + nu * (fabs(timesx[i] - timesy[j]) + fabs(timesx[i - 1] - timesy[j - 1]));



			dp[i][j] = min(min(c1, c2), c3);
		}
	}


	free(xp);
	free(yp);
	free(timesxp);
	free(timesyp);

	return dp[xlen - 1][ylen - 1];

}

double twed_scb(double x[], double timesx[], 
	int xlen, double* y, double* timesy, 
	int ylen, double lambda, double nu, int w)
{

	int i, j;
	double c1, c2, c3;

	double* cur, * prev, * temp;
	cur = (double*)malloc(ylen * sizeof(double));
	prev = (double*)malloc(ylen * sizeof(double));

	for (i = 0; i < xlen; i++)
	{
		temp = prev;
		prev = cur;
		cur = temp;
		int minw = max(0, i - w);
		int maxw = min(ylen, i + w);
		for (j = 0; j < ylen; j++)
		{
			if (i + j == 0)
			{
				cur[j] = sqrt(dist(x[i], y[j]));
			}
			else if (i == 0)
			{

				cur[j] = cur[j-1] + sqrt(dist(y[j - 1], y[j])) + nu * (timesy[j] - timesy[j - 1]) + lambda;
			}
			else if (j == 0)
			{
				cur[j] = prev[j] + sqrt(dist(x[i - 1], x[i])) + nu * (timesx[i] - timesx[i - 1]) + lambda;
			}
			else
			{

				c1 = prev[j] + sqrt(dist(x[i - 1], x[i])) + nu * (timesx[i] - timesx[i - 1]) + lambda;
				c2 = cur[j - 1] + sqrt(dist(y[j - 1], y[j])) + nu * (timesy[j] - timesy[j - 1]) + lambda;
				c3 = prev[j - 1] + sqrt(dist(x[i], y[j])) +
					sqrt(dist(x[i - 1], y[j - 1])) + nu * (fabs(timesx[i] - timesy[j]) + fabs(timesx[i - 1] - timesy[j - 1]));

				cur[j] = min(c1, min(c2, c3));
			}

		}

	}

	double result = cur[j];
	free(cur);
	free(prev);

	return result;

}

double twed_ip(double x[], double timesx[], int xlen, double* y, double* timesy, int ylen, double lambda, double nu, int s)
{

	double* prev, * cur, * temp;
	cur = (double*)malloc(ylen * sizeof(double));
	prev = (double*)malloc(ylen * sizeof(double));
	int i, j;
	double c1, c2, c3;

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
				cur[j] = sqrt(dist(x[i], y[j]));
			}
			else if (i == 0)
			{

				cur[j] = cur[j - 1] + sqrt(dist(y[j - 1], y[j])) + nu * (timesy[j] - timesy[j - 1]) + lambda;
			}
			else if (j == 0)
			{
				cur[j] = prev[j] + sqrt(dist(x[i - 1], x[i])) + nu * (timesx[i] - timesx[i - 1]) + lambda;
			}
			else
			{

				c1 = prev[j] + sqrt(dist(x[i - 1], x[i])) + nu * (timesx[i] - timesx[i - 1]) + lambda;
				c2 = cur[j - 1] + sqrt(dist(y[j - 1], y[j])) + nu * (timesy[j] - timesy[j - 1]) + lambda;
				c3 = prev[j - 1] + sqrt(dist(x[i], y[j])) +
					sqrt(dist(x[i - 1], y[j - 1])) + nu * (fabs(timesx[i] - timesy[j]) + fabs(timesx[i - 1] - timesy[j - 1]));

				cur[j] = min(c1, min(c2, c3));
			}

		}

	}

	double result = cur[j];
	free(cur);
	free(prev);

	return result;

}