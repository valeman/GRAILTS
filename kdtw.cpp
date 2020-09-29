#include "headers.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>


#define min(x, y) ((x)<(y)?(x):(y))
#define max(x, y) ((x)>(y)?(x):(y))
#define dist(x, y)((x-y)*(x-y))

#define INF 1e20

double kdtw_distance(double x, double y, double sigma)
{
	double factor = 1.0 / 3;
	double minprob = pow(10, -20);
	return factor * (exp(-sigma * pow(x - y, 2)) + minprob);
}

double kdtw(double* x, int xlen, double* y, int ylen, double sigma)
{
	double* xp = (double*)malloc(sizeof(double) * (xlen + 1));
	double* yp = (double*)malloc(sizeof(double) * (xlen + 1));
	int i, j;

	xp[0] = 0;
	yp[0] = 0;

	for (i = 1; i < xlen + 1; i++)
	{
		xp[i] = x[i - 1];
	}

	for (i = 1; i < ylen + 1; i++)
	{
		yp[i] = y[i - 1];
	}

	xlen = xlen + 1;
	ylen = ylen + 1;

	x = xp;
	y = yp;



	double** dp = (double**)malloc(sizeof(double*) * xlen);
	double** dp1 = (double**)malloc(sizeof(double*) * xlen);
	double* dp2 = (double*)calloc(max(xlen, ylen), sizeof(double));

	dp2[0] = 1;

	for (i = 1; i < min(xlen, ylen); i++)
	{
		dp2[i] = kdtw_distance(x[i], y[i], sigma);
	}

	for (i = 0; i < xlen; i++)
	{
		dp[i] = (double*)calloc(ylen, sizeof(double));
		dp1[i] = (double*)calloc(ylen, sizeof(double));
	}
	int len = min(xlen, ylen);

	dp[0][0] = 1;
	dp1[0][0] = 1;
	for (i = 1; i < xlen; i++)
	{
		dp[i][0] = dp[i - 1][0] * kdtw_distance(x[i], y[1], sigma);
		dp1[i][0] = dp1[i - 1][0] * dp2[i];
	}
	for (i = 1; i < ylen; i++)
	{
		dp[0][i] = dp[0][i - 1] * kdtw_distance(x[1], y[i], sigma);
		dp1[0][i] = dp1[0][i - 1] * dp2[i];
	}
	for (i = 1; i < xlen; i++)
	{
		for (j = 1; j < ylen; j++)
		{
			double lcost = kdtw_distance(x[i], y[j], sigma);
			dp[i][j] = (dp[i - 1][j] + dp[i][j - 1] + dp[i - 1][j - 1]) * lcost;
			if (i == j)
			{
				dp1[i][j] = dp1[i - 1][j - 1] * lcost + dp1[i - 1][j] * dp2[i] + dp1[i][j - 1] * dp2[j];
			}
			else
			{
				dp1[i][j] = dp1[i - 1][j] * dp2[i] + dp1[i][j - 1] * dp2[j];
			}
		}
	}
	for (i = 0; i < xlen; i++)
	{
		for (j = 0; j < ylen; j++)
		{
			dp[i][j] += dp1[i][j];
		}
	}
	double ans = dp[xlen - 1][ylen - 1];
	for (i = 0; i < xlen; i++)
	{
		free(dp[i]);
		free(dp1[i]);
	}
	free(xp);
	free(yp);
	free(dp);
	free(dp1);
	
	return ans;
}