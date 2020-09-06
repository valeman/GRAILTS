#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>


#define min(x, y) ((x)<(y)?(x):(y))
#define max(x, y) ((x)>(y)?(x):(y))
#define dist(x, y)((x-y)*(x-y))

#define INF 1e20


double edr(double** x, int xlen, double** y, int ylen,double m, const char* constraint, int w)
{
	if (strcmp(constraint, "None"))
	{
		return edr_n(x, xlen, y, ylen,m);
	}
	else if (strcmp(constraint, "Sakoe-Chiba"))
	{
		return edr_scb(x, xlen, y, ylen, w,m);
	}
	else if (strcmp(constraint, "Sakoe-Chiba"))
	{
		return edr_ip(x, xlen, y, ylen, w,m);
	}
	else
	{
		fprintf(stderr, "constraint did not match any of the acceptable values ('None', 'Sakoe-Chiba', 'Itakura'), defaulting to 'None'");
		return edr_n(x, xlen, y, ylen,m);
	}


}


double edr_n(double** x, int xlen, double** y, int ylen, double m)
{
	double** df = (double**)malloc(sizeof(double*) * (xlen));
	int i, j;
	double s1, s2, s3;

	for (i = 0; i < xlen; i++)
	{
		df[i] = (double*)malloc(sizeof(double) * ylen);
		df[i][0] = -i;
	}
	for (i = 0; i < ylen; i++)
	{
		df[0][i] = -i;
	}

	for (i = 1; i < xlen; i++)
	{
		for (j = 1; j < ylen; j++)
		{
			if ((fabs(x[0][i] - y[0][j]) <= m) &&
				(fabs(x[1][i] - y[1][j]) <= m))
			{
				s1 = 0;
			}
			else
			{
				s1 = -1;
			}

			s1 = df[i - 1][j - 1] + s1;
			s2 = df[i][j - 1] - 1;
			s3 = df[i - 1][j - 1] - 1;

			df[i][j] = max(max(s1, s2), s3);

		}
	}
	double result = df[xlen - 1][ylen - 1];
	for (i = 0; i < xlen; i++)
	{
		free(df[i]);
	}
	free(df);
	return  0 - result;
}

double edr_scb(double** x, int xlen, double** y, int ylen, double m, int w)
{
	double* prev, * cur, * temp;
	cur = (double*)malloc(ylen * sizeof(double));
	prev = (double*)malloc(ylen * sizeof(double));
	int i, j;
	double s1, s2, s3;

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
				cur[j] = -j;
			}
			else if (j == 0)
			{
				cur[j] = -i;
			}
			else
			{
				if ((fabs(x[0][i] - y[0][j]) <= m) &&
					(fabs(x[1][i] - y[1][j]) <= m))
				{
					s1 = 0;
				}
				else
				{
					s1 = -1;
				}

				s1 = prev[j - 1] + s1;
				s2 = cur[j - 1] - 1;
				s3 = prev[j] - 1;

				cur[j] = max(max(s1, s2), s3);
			}
		}
	}
	double result = cur[ylen - 1];
	free(cur);
	free(prev);
	return 0 - result;
}

double edr_ip(double** x, int xlen, double** y, int ylen, double m, int s)
{
	double* prev, * cur, * temp;
	cur = (double*)malloc(ylen * sizeof(double));
	prev = (double*)malloc(ylen * sizeof(double));
	int i, j;
	double s1, s2, s3;

	double min_slope = 1 / s * (double)xlen / (double)ylen;
	double max_slope = s * (double)xlen / (double)ylen;

	min_slope * i;
	(double)(((double)ylen - 1) - max_slope * ((double)xlen - 1))
		+ max_slope * i;

	max_slope * i;
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
				cur[j] = -j;
			}
			else if (j == 0)
			{
				cur[j] = -i;
			}
			else
			{
				if ((fabs(x[0][i] - y[0][j]) <= m) &&
					(fabs(x[1][i] - y[1][j]) <= m))
				{
					s1 = 0;
				}
				else
				{
					s1 = -1;
				}

				s1 = prev[j - 1] + s1;
				s2 = cur[j - 1] - 1;
				s3 = prev[j] - 1;

				cur[j] = max(max(s1, s2), s3);
			}
		}
	}
	double result = cur[ylen - 1];
	free(cur);
	free(prev);
	return 0 - result;
}
