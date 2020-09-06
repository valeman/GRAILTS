#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <complex.h>
#include <fftw3.h>
#include <gsl/gsl_blas.h>
#include <gsl\gsl_eigen.h>

#define min(x, y) ((x)<(y)?(x):(y))
#define max(x, y) ((x)>(y)?(x):(y))
#define dist(x, y)((x-y)*(x-y))

double* cmultiply(double* x, double* y)
{
	if (y[1] < -10000000000000000000.0)
	{
		y[1] = 0;
	}
	if (x[1] < -10000000000000000.0)
	{
		x[1] = 0;
	}
	double* result = (double*)malloc(sizeof(double) * 2);
	result[0] = x[0] * y[0] - x[1] * y[1];
	result[1] = x[1] * y[0] + x[0] * y[1];

	return  result;
}


double dmultiply(double* x, double* y)
{
	double* result = cmultiply(x, y);
	double c = sqrt((result[0] * result[0]) + (result[1] * result[1]));
	free(result);

	return c;
}

double dfnorm(fftw_complex* x, int xlen)
{
	int i;
	double sum = 0;
	for (i = 0; i < xlen; i++)
	{
		sum += pow(x[i][0], 2);
	}
	return sqrt(sum);
}


double sink(fftw_complex* x, int xlen, fftw_complex* y, int ylen, double gamma)
{
	int i;
	int len = (int)pow(2, ceil(log2(abs(2 * max(xlen, ylen) - 1))));
	fftw_complex* xi = (fftw_complex*)malloc(sizeof(fftw_complex) * len);
	fftw_complex* yi = (fftw_complex*)malloc(sizeof(fftw_complex) * len);
	fftw_complex* xo = (fftw_complex*)malloc(sizeof(fftw_complex) * len);
	fftw_complex* yo = (fftw_complex*)malloc(sizeof(fftw_complex) * len);
	for (i = 0; i < len; i++)
	{
		if (i < xlen)
		{
			xi[i][0] = x[i][0];
		}
		else
		{
			xi[i][0] = 0;
		}
		xi[i][1] = 0;
	}
	for (i = 0; i < len; i++)
	{
		if (i < ylen)
		{
			yi[i][0] = y[i][0];
		}
		else
		{
			yi[i][0] = 0;
		}
		yi[i][1] = 0;
	}

	fftw_plan p = fftw_plan_dft_1d(len, xi, xo, FFTW_FORWARD, FFTW_ESTIMATE);
	fftw_plan p2 = fftw_plan_dft_1d(len, yi, yo, FFTW_FORWARD, FFTW_ESTIMATE);

	fftw_execute(p);
	fftw_execute(p2);

	fftw_destroy_plan(p);
	fftw_destroy_plan(p2);

	double normp = dfnorm(x, xlen);
	normp *= dfnorm(y, ylen);

	fftw_complex* arr = (fftw_complex*)malloc(sizeof(fftw_complex) * len);
	fftw_complex* arro = (fftw_complex*)malloc(sizeof(fftw_complex) * len);
	for (int i = 0; i < len; i++)
	{

		arr[i][0] = dmultiply(xo[i], yo[i]);
		arr[i][1] = 0;
	}

	fftw_plan p3 = fftw_plan_dft_1d(len, arr, arro, FFTW_BACKWARD, FFTW_ESTIMATE);
	fftw_execute(p3);
	fftw_destroy_plan(p3);
	fftw_free(xo);
	fftw_free(yo);

	double result = 0;

	for (i = 0; i < len; i++)
	{
		arro[i][0] = arro[i][0] / len;
	}

	for (i = len - max(xlen, ylen) + 1; i < len; i++)
	{
		result += exp(gamma * arro[i][0] / (normp));
	}

	for (i = 0; i < max(xlen, ylen); i++)
	{
		result += exp(gamma * arro[i][0] / normp);
	}


	fftw_free(xi);
	fftw_free(yi);
	fftw_free(arr);
	fftw_free(arro);

	return result;

}

int comp(const void* x, const void* y)
{
	int xp = (int)x;
	int yp = (int)y;
	if (xp > yp)
	{
		return -1;
	}
	else if (yp > xp)
	{
		return 1;
	}
	return 0;
}

gsl_matrix* GRAIL(fftw_complex** x, int* xdim, fftw_complex** sample, int* samdim, double gamma, double r)
{
	int i, j;
	double* W = (double*)calloc(samdim[0] * samdim[0], sizeof(double));
	double* E = (double*)calloc(samdim[0] * samdim[0], sizeof(double));
	if (r > samdim[0])
	{
		fprintf(stderr, "Error:The number of return values wanted is greater than the number of return values (i.e. r > samdim[0]) \n");
		exit(-1);
	}
	if (r < 0)
	{
		fprintf(stderr, "Error: r is less than 0 and negative r values are not supported. Defaulting to entire array. (i.e. r < 0) \n");
		r = samdim[0];
	}

	for (i = 0; i < samdim[0]; i++)
	{
		for (j = 0; j < samdim[0]; j++)
		{
			W[i * samdim[1] + j] = sink(sample[i], samdim[1], sample[j], samdim[1], gamma);
			printf("%d %d\n", i, j);
		}
	}
	printf("\n\n");
	for (i = 0; i < xdim[0]; i++)
	{
		for (j = 0; j < samdim[0]; j++)
		{
			E[i * xdim[1] + j] = sink(x[i], xdim[1], sample[j], samdim[1], gamma);
			printf("%d %d\n", i, j);
		}
	}


	gsl_matrix_view mW = gsl_matrix_view_array(W, samdim[0], samdim[0]);


	gsl_vector* eval = gsl_vector_alloc(samdim[0]);
	gsl_matrix* evec = gsl_matrix_alloc(samdim[0], samdim[0]);

	gsl_eigen_symmv_workspace* w = gsl_eigen_symmv_alloc(samdim[0]);

	gsl_eigen_symmv(&mW.matrix, eval, evec, w);

	double* inVa_array = (double*)calloc(samdim[0] * samdim[0], sizeof(double));


	for (i = 0; i < samdim[0]; i++)
	{
		double value = gsl_vector_get(eval, i);
		if (value <= 0)
			inVa_array[i * samdim[0] + i] = 0;
		else
		{
			inVa_array[i * samdim[0] + i] = pow(value, -.5);
		}
	}

	gsl_matrix_view inVa = gsl_matrix_view_array(inVa_array, samdim[0], samdim[0]);
	gsl_matrix_view mE = gsl_matrix_view_array(E, xdim[0], samdim[0]);


	//E * evec * inVa
	//(X x Sam) * (Sam x Sam) * (Sam * Sam) = (X x Sam)
	gsl_matrix* intermediate = gsl_matrix_alloc(samdim[0], samdim[0]);
	gsl_matrix* Zexact = gsl_matrix_alloc(xdim[0], samdim[0]);



	gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1, evec, &inVa.matrix, 0, intermediate);
	gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1, &mE.matrix, intermediate, 0, Zexact);

	//Intermediate free
	free(W);
	free(E);
	gsl_vector_free(eval);
	gsl_matrix_free(evec);
	gsl_matrix_free(intermediate);
	free(inVa_array);

	gsl_eigen_symmv_free(w);




	gsl_matrix* Zexact_sqr = gsl_matrix_alloc(samdim[0], samdim[0]);

	gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1, Zexact, Zexact, 0, Zexact_sqr);

	gsl_vector* evalz = gsl_vector_alloc(samdim[0]);
	gsl_matrix* evecz = gsl_matrix_alloc(samdim[0], samdim[0]);

	gsl_eigen_symmv_workspace* wz = gsl_eigen_symmv_alloc(samdim[0]);

	gsl_eigen_symmv(Zexact_sqr, evalz, evecz, wz);
	qsort(evalz->data, evalz->size, sizeof(double), comp);


	double* varexplained = (double*)malloc(sizeof(double) * evalz->size);

	double sum = 0;
	for (i = 0; i < evalz->size; i++)
	{
		sum += evalz->data[i];
	}

	varexplained[0] = evalz->data[0] / sum;
	for (i = 1; i < evalz->size; i++)
	{
		varexplained[i] = evalz->data[i - 1] + evalz->data[i] / sum;
	}

	gsl_matrix* result = NULL;
	gsl_matrix* v2 = NULL;
	if (r >= 1)
	{
		result = gsl_matrix_alloc(xdim[0], r);
		v2 = gsl_matrix_alloc(samdim[0], r);
		gsl_vector* temp = gsl_vector_alloc(samdim[0]);
		for (i = 0; i < r; i++)
		{
			gsl_matrix_get_col(temp, evecz, i);
			gsl_matrix_set_col(v2, i, temp);
		}
		gsl_vector_free(temp);
		gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1, Zexact, v2, 0, result);

	}
	else if (r == 0)
	{
		r = evalz->size;
		result = gsl_matrix_alloc(xdim[0], r);
		v2 = gsl_matrix_alloc(samdim[0], r);
		gsl_vector* temp = gsl_vector_alloc(samdim[0]);
		for (i = 0; i < r; i++)
		{
			gsl_matrix_get_col(temp, evecz, i);
			gsl_matrix_set_col(v2, i, temp);
		}
		gsl_vector_free(temp);
		gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1, Zexact, v2, 0, result);
	}/* Need to add percentage support.
	else if ((r < 1) && (r > 0))
	{
		r =  floor(samdim[0] * r);
		result = (double*)malloc(sizeof(double) * ((int) r));
	}*/

	gsl_matrix_free(v2);
	gsl_matrix_free(Zexact_sqr);
	gsl_vector_free(evalz);
	gsl_matrix_free(evecz);
	gsl_eigen_symmv_free(wz);

	return result;
}
