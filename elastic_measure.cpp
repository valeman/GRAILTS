#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>


#define min(x, y) ((x)<(y)?(x):(y))
#define max(x, y) ((x)>(y)?(x):(y))
#define dist(x, y)((x-y)*(x-y))

#define INF 1e20


/* sakoe_chiba

	Sets a matrix of size xlen, ylen according to a particular sakoe_chiba_radius


	Inputs:

	int xlen: length of the first time series;

	int ylen: length of the second time series;

	int sakoe_chiba_radius: radius of the band to be used,, must be positive and an integer

	Output:

	double ** mask: a pointer corresponding to a double* array

*/


double** sakoe_chiba(int xlen, int ylen, int sakoe_chiba_radius)
{
	double** mask = (double**)malloc(sizeof(double*) * xlen);
	int i, j;
	int width= 0;


	for (i = 0; i < xlen; i++)
	{
		mask[i] = (double*)malloc(sizeof(double*) * ylen);
	}

	for (i = 0; i < xlen; i++)
	{
		for (j = 0; j < ylen; j++)
		{
			mask[i][j] = INF;
		}
	}

	if (xlen > ylen)
	{
		width = xlen - ylen + sakoe_chiba_radius;
		for (i = 0; i < ylen; i++)
		{
			int l = max(0, i - sakoe_chiba_radius);
			int u = min(xlen, i + width) + 1;
			for (j = l; j < u; j++)
			{
				mask[j][i] = 0;
			}
		}
	}
	else
	{
		width = ylen - xlen + sakoe_chiba_radius;
		for (i = 0; i < xlen; i++)
		{
			int l = max(0, i - sakoe_chiba_radius);
			int u = min(ylen, i + width) + 1;
			for (j = l; j < u; j++)
			{
				mask[i][j] = 0;
			}
		}
	}

	return mask;
}

/* itakura

	Sets a matrix of size xlen, ylen according to a particular itakura_max_slope

	Inputs:

	int xlen: length of the first time series;

	int ylen: length of the second time series;

	int itakura_max_slope: radius of the band to be used,, must be positive and an integer

	Output:

	double ** mask: a pointer corresponding to a double* array

*/

double** itakura(int xlen, int ylen, int itakura_max_slope)
{
	double** mask = (double**)malloc(sizeof(double*) * xlen);
	int i, j;
	int width = 0;
	double min_slope, max_slope;


	for (i = 0; i < xlen; i++)
	{
		mask[i] = (double*)malloc(sizeof(double*) * ylen);
	}

	for (i = 0; i < xlen; i++)
	{
		for (j = 0; j < ylen; j++)
		{
			mask[i][j] = INF;
		}
	}

	min_slope = 1 / itakura_max_slope * (double)xlen * (double)ylen;
	max_slope = itakura_max_slope * (double)xlen * (double)ylen;

	double* lb1 = (double*)malloc(sizeof(double) * ylen);
	double* lb2 = (double*)malloc(sizeof(double) * ylen);
	double* ub1 = (double*)malloc(sizeof(double) * ylen);
	double* ub2 = (double*)malloc(sizeof(double) * ylen);

	for (i = 0; i < ylen; i++)
	{
		lb1[i] = min_slope * i;
		lb2[i] = (double)(((double)xlen - 1) - max_slope * ((double)ylen - 1))
			+ max_slope * i;

		ub1[i] = max_slope * i;
		ub2[i] = (((double)xlen - 1) - min_slope * ((double)ylen - 1))
			+ min_slope * i;
	}

	int* upper_bound = (int*)malloc(sizeof(int) * ylen);
	int* lower_bound = (int*)malloc(sizeof(int) * ylen);

	for (i = 0; i < ylen; i++)
	{
		upper_bound[i] = (int)min(ub1[i], ub2[i]) + 1;
		lower_bound[i] = (int)max(lb1[i], lb2[i]);
	}

	for (i = 0; i < ylen; i++)
	{
		for (j = lower_bound[i]; j < upper_bound[i]; j++)
		{
			mask[j][i] = 0;
		}
	}

	int check = 0;
	int row;

	for (i = 0; i < xlen; i++)
	{
		row = 1;
		for (j = 0; j < ylen; j++)
		{
			if (mask[i][j] == 0)
			{
				row = 0;
			}
		}
		if (row == 1)
		{
			check = 1;
		}
	}
	if (!check)
	{
		for (i = 0; i < ylen; i++)
		{
			row = 1;
			for (j = 0; j < xlen; j++)
			{
				if (mask[i][j] == 0)
				{
					row = 0;
				}
			}
			if (row == 1)
			{
				check = 1;
			}
		}

	}

	if (check)
	{
		fprintf(stderr, "No permissable path, all values set to infinity. \n Itakura max slope parameters not possible");
		fprintf(stderr, "xlen: %d, ylen: %d, itakura_max_slope: %d", xlen, ylen, itakura_max_slope);
		exit(1);
	}

	free(lb1);
	free(lb2);
	free(ub1);
	free(ub2);
	free(upper_bound);
	free(lower_bound);

	return mask;

}

/* constraints

	Sets a matrix of size xlen, ylen according to a the constraint parameter set

	Inputs:

	int xlen: length of the first time series;

	int ylen: length of the second time series;

	char* constraint: the type of constraint to be used
	Valid options are "itakura", "sakoe-chiba", and "None"

	int sakoe_chiba_radius: radius of the band to be used, must be positive and an integer

	int itakura_max_slope:

	Output:

	double ** mask: a pointer corresponding to a double* array

*/

double** constraints(int xlen, int ylen, const char* constraint, int sakoe_chiba_radius, int itakura_max_slope)
{
	double** arr;
	int i, j;

	if (strcmp("None", constraint) == 0)
	{
		arr = (double**)malloc(sizeof(double*) * xlen);
		for (i = 0; i < xlen; i++)
		{
			arr[i] = (double*)malloc(ylen * sizeof(double));
			for (j = 0; j < ylen; j++)
			{
				arr[i][j] = 0.0;
			}
		}
		if ((sakoe_chiba_radius != -1) || (itakura_max_slope != -1))
		{
			fprintf(stderr, "Constraint parameter set but constraint set to None");
		}
	}
	else if (strcmp("Sakoe-Chiba", constraint) == 0)
	{
		arr = sakoe_chiba(xlen, ylen, sakoe_chiba_radius);
		if (itakura_max_slope != -1)
		{
			fprintf(stderr, "itakura_max_slope set but constraint set to Sakoe-Chiba");
		}
	}
	else if (strcmp("Itakura", constraint) == 0)
	{
		arr = itakura(xlen, ylen, itakura_max_slope);
		if (sakoe_chiba_radius != -1)
		{
			fprintf(stderr, "sakoe_chiba_radius set but constraint set to Itakura");
		}
	}
	else
	{
		fprintf(stderr, "Constraint does not match any of the options, defaulting to no constraint");
		arr = (double**)malloc(sizeof(double*) * xlen);
		for (i = 0; i < xlen; i++)
		{
			arr[i] = (double*)malloc(ylen * sizeof(double));
			for (j = 0; j < ylen; j++)
			{
				arr[i][j] = 0.0;
			}
		}
	}

	return arr;
}

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

double dtw(double* x, int xlen, double* y, int ylen, const char* constraint, int sakoe_chiba_radius, int itakura_max_slope)
{
	double** arr;
	int i, j;


	arr = constraints(xlen, ylen, constraint, sakoe_chiba_radius, itakura_max_slope);



	arr[0][0] = fabs(x[0] - y[0]);


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

	return final_dtw;
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

double msm(double* x, int xlen, double* y, int ylen, double c)
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

double twed(double x[], double timesx[], int xlen, double* y, double* timesy, int ylen, double lambda, double nu)
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

double erp(double x[], int xlen, double y[], int ylen, double m)
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
	double result = df[xlen-1][ylen-1];
	for (i = 0; i < xlen + 1; i++)
	{
		free(df[i]);
	}
	free(df);
	return  sqrt(0 - result);
}



double edr(double** x, int xlen, double** y, int ylen, double m)
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

double lcss(double* x, int xlen, double* y, int ylen, double epsilon)
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


double swale(double* x, int xlen, double* y, int ylen, double p, double r, double epsilon)
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

/***
double dtw_itakura_max_slope(double* x, double* y, int xlen, int ylen, int itakura_max_slope) :
{
	double min_slope, max_slope;

	min_slope = 1 / itakura_max_slope * (double)xlen * (double)ylen;
	max_slope = itakura_max_slope * (double)xlen * (double)ylen;

	double lb[2][ylen];
	double ub[2][ylen];

	for (i = 0; i < ylen; i++)
	{
		lb[0][i] = min_slope * i;
		lb[1][i] = ((xlen - 1) - max_slope * (ylen - 1))
			+ max_slope * i;

		ub[0][i] = max_slope * i;
		ub[1][i] = ((xlen - 1) - min_slope * (ylen - 1))
			+ min_slope * i;
	}

	int upper_bound[ylen];
	int lower_bound[ylen];

	for (i = 0; i < ylen; i++)
	{
		upper_bound[i] = (int)min(ub[0][i], ub[1][i]) + 1;
		lower_bound[i] = (int)max(lb[0][i], lb[1][i]);
	}

	for (i = 0; i < ylen; i++)
	{
		for (j = lower_bound[i]; j < upper_bound[i]; j++)
		{
			mask[j][i] = 0;
		}
	}




}
***/

/*
double erp2(double x[],int xlen, double y[], int ylen, double m, int bandsize)
{
  double * curr = (double*) malloc(xlen * sizeof(double));
  double* prev = (double*) malloc(xlen * sizeof(double));
  int band = ceil(bandsize * xlen);
  int i,j;
  double * temp;
  int p,r;

  for(i = 0; i < xlen; i ++)
  {
	temp = prev;
	prev = curr;
	curr = temp;
	p = i - (band + 1);
	if (p < 0)
	{
	  p = 0;
	}
	r = i + (band + 1);
	if (r > m -1)
	{
	  r = m-1;
	}
	for(j = p; j < r; j ++)
	{
	  if (abs(i - j) < band)
	  {
		double diff_h = pow(x[i] - m,2);
		double diff_v = pow(y[i] - m,2);
		double diff_d = pow(x[i] - y[j],2);
	  }
	}
  }
  return 0;
}*/

