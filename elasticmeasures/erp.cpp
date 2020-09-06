#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>


#define min(x, y) ((x)<(y)?(x):(y))
#define max(x, y) ((x)>(y)?(x):(y))
#define dist(x, y)((x-y)*(x-y))

#define INF 1e20

double erp(double* x, int xlen, double* y, int ylen, double m, const char* constraint, int w)
{
	if (strcmp(constraint, "None"))
	{
		return erp_n(x, xlen, y, ylen,m);
	}
	else if (strcmp(constraint, "Sakoe-Chiba"))
	{
		return erp_scb(x, xlen, y, ylen,m, w);
	}
	else if (strcmp(constraint, "Sakoe-Chiba"))
	{
		return erp_ip(x, xlen, y, ylen,m, w);
	}
	else
	{
		fprintf(stderr, "constraint did not match any of the acceptable values ('None', 'Sakoe-Chiba', 'Itakura'), defaulting to 'None'");
		return erp_n(x, xlen, y, ylen,m);
	}


}

double erp_n(double x[], int xlen, double y[], int ylen, double m)
{
	double** df = (double**)malloc(sizeof(double*) * (xlen + 1));
	int i, j;
	double s1, s2, s3;

	for (i = 0; i < xlen + 1; i++)
	{
		df[i] = (double*)malloc(sizeof(double) * (ylen + 1));
		if (i == 0)
		{
			df[i][0] = 0;
		}
		else
		{
			df[i][0] = df[i - 1][0] - pow(x[i] - m, 2);
		}
	}

	df[0][0] = 0;


	for (i = 1; i < ylen + 1; i++)
	{
		df[0][i] = df[0][i - 1] - pow(y[i] - m, 2);
	}

	df[1][1] = 0;

	for (i = 1; i < xlen + 1; i++)
	{
		for (j = 1; j < ylen + 1; j++)
		{
			double diff_d = pow(x[i] - y[j], 2);
			double diff_h = pow(x[i] - m, 2);
			double diff_v = pow(y[j] - m, 2);


			s1 = df[i - 1][j - 1] - diff_d;
			s2 = df[i][j - 1] - diff_v;
			s3 = df[i - 1][j] - diff_h;

			df[i][j] = max(max(s1, s2), s3);
		}
	}
	double result = df[xlen - 1][ylen - 1];
	for (i = 0; i < xlen + 1; i++)
	{
		free(df[i]);
	}
	free(df);
	return  sqrt(0 - result);
}

double erp_scb(double x[], int xlen, double y[], int ylen, double m,int w)
{
	double* prev, * cur, * temp;
	cur = (double*)malloc(ylen * sizeof(double));
	prev = (double*)malloc(ylen * sizeof(double));
	int i, j;
	double s1, s2, s3;

	for (i = 0; i < xlen + 1; i++)
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
				cur[0] = 0;
			}
			else if (i == 0)
			{
				cur[j] = cur[j-1] - pow(y[j] - m, 2);
			}
			else if (j == 0)
			{
				cur[j] = prev[j] - pow(x[i] - m, 2);
			}
			else
			{

				double diff_d = pow(x[i] - y[j], 2);
				double diff_h = pow(x[i] - m, 2);
				double diff_v = pow(y[j] - m, 2);


				s1 = prev[j - 1] - diff_d;
				s2 = cur[j - 1] - diff_v;
				s3 = prev[j] - diff_h;

				cur[j] = max(max(s1, s2), s3);
			}
		}
	}
	double result = cur[ylen - 1];
	
	free(cur);
	free(prev);
	return  sqrt(0 - result);
}

double erp_ip(double x[], int xlen, double y[], int ylen, double m, int s)
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
				cur[0] = 0;
			}
			else if (i == 0)
			{
				cur[j] = cur[j - 1] - pow(y[j] - m, 2);
			}
			else if (j == 0)
			{
				cur[j] = prev[j] - pow(x[i] - m, 2);
			}
			else
			{

				double diff_d = pow(x[i] - y[j], 2);
				double diff_h = pow(x[i] - m, 2);
				double diff_v = pow(y[j] - m, 2);


				s1 = prev[j - 1] - diff_d;
				s2 = cur[j - 1] - diff_v;
				s3 = prev[j] - diff_h;

				cur[j] = max(max(s1, s2), s3);
			}
		}
	}
	double result = cur[ylen - 1];

	free(cur);
	free(prev);
	return  sqrt(0 - result);
}
