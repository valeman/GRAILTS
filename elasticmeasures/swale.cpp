#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>


#define min(x, y) ((x)<(y)?(x):(y))
#define max(x, y) ((x)>(y)?(x):(y))
#define dist(x, y)((x-y)*(x-y))

#define INF 1e20



double swale(double* x, int xlen, double* y, int ylen,double p, double r,double epsilon, const char* constraint, int w)
{
	if (strcmp(constraint, "None"))
	{
		return swale_n(x, xlen, y, ylen,p,r,epsilon);
	}
	else if (strcmp(constraint, "Sakoe-Chiba"))
	{
		return swale_scb(x, xlen, y, ylen,p,r,epsilon,w);
	}
	else if (strcmp(constraint, "Sakoe-Chiba"))
	{
		return swale_ip(x, xlen, y, ylen,p,r,epsilon,w);
	}
	else
	{
		fprintf(stderr, "constraint did not match any of the acceptable values ('None', 'Sakoe-Chiba', 'Itakura'), defaulting to 'None'");
		return swale_n(x, xlen, y, ylen,p,r,epsilon);
	}


}
double swale_n(double* x, int xlen, double* y, int ylen, double p, double r, double epsilon)
{
	double** df = (double**)malloc(sizeof(double*) * (xlen));
	int i, j;

	for (i = 0; i < ylen; i++)
	{
		df[i] = (double*)malloc(ylen * sizeof(double));
		df[0][i] = i * p;
	}
	for (i = 0; i < xlen; i++)
	{
		df[i][0] = i * p;
	}

	for (i = 1; i < xlen; i++)
	{
		for (j = 1; j < ylen; j++)
		{
			if (fabs(x[i] - y[j]) <= epsilon)
			{
				df[i][j] = df[i - 1][j - 1] + r;
			}
			else
			{
				double d1 = df[i][j - 1] + p;
				double d2 = df[i - 1][j] + p;
				df[i][j] = max(d1, d2);
			}
		}
	}
	double result = df[xlen - 1][ylen - 1];
	for (i = 0; i < xlen; i++)
	{
		free(df[i]);
	}
	free(df);

	return result;
}



double swale_ip(double* x, int xlen, double* y, int ylen, double p, double r, double epsilon, int s)
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
				cur[j] = 0;
			}
			else if (i == 0)
			{
				cur[j] = j * p;
			}
			else if (j == 0)
			{
				cur[j] = i * p;
			}
			else
			{
				if (fabs(x[i] - y[j]) <= epsilon)
				{
					cur[j] = prev[j - 1] + r;
				}
				else
				{
					double d1 = cur[j - 1] + p;
					double d2 = prev[j] + p;
					cur[j] = max(d1, d2);
				}
			}
		}
	}
		double result = cur[ylen - 1];
		free(cur);
		free(prev);

		return result;
}

double swale_scb(double* x, int xlen, double* y, int ylen, double p, double r, double epsilon, int w)
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
				cur[j] = 0;
			}
			else if (i == 0)
			{
				cur[j] = j * p;
			}
			else if (j == 0)
			{
				cur[j] = i * p;
			}
			else
			{
				if (fabs(x[i] - y[j]) <= epsilon)
				{
					cur[j] = prev[j - 1] + r;
				}
				else
				{
					double d1 = cur[j - 1] + p;
					double d2 = prev[j] + p;
					cur[j] = max(d1, d2);
				}
			}
		}
	}
	double result = cur[ylen - 1];
	free(cur);
	free(prev);

	return result;
}

