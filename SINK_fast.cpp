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
#define LOGP(x, y) (((x)>(y))?(x)+log1p(exp((y)-(x))):(y)+log1p(exp((x)-(y))))

#define LOG0 -10000          /* log(0) */
#define INF 1e20


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
	double* result =(double*) malloc(sizeof(double) * 2);
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
	return dp[xlen - 1][ylen - 1];
}

double sink(double* x_arr, int xlen, double* y_arr, int ylen, double gamma)
{
	fftw_complex x[xlen];
    fftw_complex y[ylen];
	for (int i = 0; i < xlen; i++)
	{
		x[i] = x_arr[i]
	}
	for (int i = 0; i < ylen; i++)
	{
		y[i] = y_arr[i]
	}
	int i;
	int len = (int)pow(2, ceil(log2(abs(2 * max(xlen,ylen) - 1))));
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

	for (i = 0; i < max(xlen,ylen); i++)
	{
		result += exp(gamma * arro[i][0] / normp);
	}


	fftw_free(xi);
	fftw_free(yi);
	fftw_free(arr);
	fftw_free(arro);

	return result;

}

//Reference https://marcocuturi.net/GA.html
/*
double logGAK2(double* x, int xlen, double* y, int ylen, int dimvect,  double sigma, int triangular)
{
	int i, j,cur,old,curpos,frompos1,frompos2,frompos3;
	int cl = ylen + 1;


	double sum = 0;
	double gram, Sig;
	
	double* arr = (double*) malloc(2 * cl * sizeof(double));


	int trimax = (xlen > ylen) ? xlen - 1 : ylen - 1;

	double* logTC = (double*)malloc((trimax + 1) * sizeof(double));

	if (triangular > 0)
	{
		for (i = 0; i <= trimax; i++)
		{
			logTC[i] = LOG0;
		}

		for (i = 0; i <= ((trimax < triangular) ? trimax + 1 : triangular); i++)
		{
			logTC[i] = log(1 - ((double) i) / triangular);
		}
	}
	else
	{
		for (i = 0; i <= trimax; i++)
		{
			logTC[i] = 0;
		}
	}

	Sig = -1 / (2 * sigma * sigma);


	for (i = 1; i < cl; i++)
	{
		arr[i] = LOG0;
	}

	arr[0] = 0;

	cur = 1;
	old = 0;
	double aux;

	for (i = 1; i <= xlen; i++)
	{
		curpos = cur * cl;
		arr[curpos] = LOG0;
		for (j = 1; j <= ylen; j++)
		{
			curpos = cur * cl + j;
			if (logTC[abs(i - j)] > LOG0)
			{
				frompos1 = old * cl + j;
				frompos2 = cur * cl + j - 1;
				frompos3 = old * cl + j - 1;

				sum = 0;

				for (int k = 0; k < dimvect; k++)
				{
					sum += (x[i - 1 + k * xlen] - y[j - 1 + k * ylen]) * (x[i - 1 + k * xlen] * y[j - 1 + k * ylen]);
				}
				gram = logTC[abs(i - j)] + sum * Sig;
				gram -= log(2 - exp(gram));

				aux = LOGP(arr[frompos1], arr[frompos2]);
				arr[curpos] = LOGP(aux,arr[frompos3]) + gram;
				
			}
			else
			{
				arr[curpos] = LOG0;
			}
		}
		cur = 1 - cur;
		old = 1 - old;
	}
	aux = arr[curpos];

	free(arr);
	free(logTC);

	return aux;
}
*/


//Code Source https://marcocuturi.net/GA.html
//Author: Marco Cuturi
double logGAK(double *seq1 , double *seq2, int nX, int nY, int dimvect, double sigma, int triangular)
/* Implementation of the (Triangular) global alignment kernel.
 *
 * See details about the matlab wrapper mexFunction below for more information on the inputs that need to be called from Matlab
 *
 * /* seq1 is a first sequence represented as a matrix of real elements. Each line i corresponds to the vector of observations at time i.
 * seq2 is the second sequence formatted in the same way.
 * nX, nY and dimvect provide the number of lines of seq1 and seq2.
 * sigma stands for the bandwidth of the \phi_\sigma distance used kernel
 * lambda is an additional factor that can be used with the Geometrically divisible Gaussian Kernel
 * triangular is a parameter which parameterizes the triangular kernel
 * kerneltype selects either the Gaussian Kernel or its geometrically divisible equivalent */

{
    int i, j, ii, cur, old, curpos, frompos1, frompos2, frompos3;    
	double aux;
    int cl = nY+1;                /* length of a column for the dynamic programming */
    
    
    double sum=0;
    double gram, Sig;    
    /* logM is the array that will stores two successive columns of the (nX+1) x (nY+1) table used to compute the final kernel value*/
    double * logM = (double*) malloc(2*cl * sizeof(double));        
    
    int trimax = (nX>nY) ? nX-1 : nY-1; /* Maximum of abs(i-j) when 1<=i<=nX and 1<=j<=nY */
    
    double *logTriangularCoefficients =(double*) malloc((trimax+1) * sizeof(double)); 
    if (triangular>0) {
        /* initialize */
        for (i=0;i<=trimax;i++){
            logTriangularCoefficients[i]=LOG0; /* Set all to zero */
        }
        
        for (i=0;i<((trimax<triangular) ? trimax+1 : triangular);i++) {
            logTriangularCoefficients[i]=log(1-i/triangular);
        }
    }
    else
        for (i=0;i<=trimax;i++){
        logTriangularCoefficients[i]=0; /* 1 for all if triangular==0, that is a log value of 0 */
        }
    Sig=-1/(2*sigma*sigma);
    
    
    
    /****************************************************/
    /* First iteration : initialization of columns to 0 */
    /****************************************************/
    /* The left most column is all zeros... */
    for (j=1;j<cl;j++) {
        logM[j]=LOG0;
    }
    /* ... except for the lower-left cell which is initialized with a value of 1, i.e. a log value of 0. */
    logM[0]=0;
    
    /* Cur and Old keep track of which column is the current one and which one is the already computed one.*/
    cur = 1;      /* Indexes [0..cl-1] are used to process the next column */
    old = 0;      /* Indexes [cl..2*cl-1] were used for column 0 */
    
    /************************************************/
    /* Next iterations : processing columns 1 .. nX */
    /************************************************/
    
    /* Main loop to vary the position for i=1..nX */
    for (i=1;i<=nX;i++) {
        /* Special update for positions (i=1..nX,j=0) */
        curpos = cur*cl;                  /* index of the state (i,0) */
        logM[curpos] = LOG0;
        /* Secondary loop to vary the position for j=1..nY */
        for (j=1;j<=nY;j++) {
            curpos = cur*cl + j;            /* index of the state (i,j) */
            if (logTriangularCoefficients[abs(i-j)]>LOG0) {
                frompos1 = old*cl + j;            /* index of the state (i-1,j) */
                frompos2 = cur*cl + j-1;          /* index of the state (i,j-1) */
                frompos3 = old*cl + j-1;          /* index of the state (i-1,j-1) */
                
                /* We first compute the kernel value */
                sum=0;
                for (ii=0;ii<dimvect;ii++) {
                    sum+=(seq1[i-1+ii*nX]-seq2[j-1+ii*nY])*(seq1[i-1+ii*nX]-seq2[j-1+ii*nY]);
                }
                gram= logTriangularCoefficients[abs(i-j)] + sum*Sig ;
                gram -=log(2-exp(gram));
                
                /* Doing the updates now, in two steps. */
                aux= LOGP(logM[frompos1], logM[frompos2] );
                logM[curpos] = LOGP( aux , logM[frompos3] ) + gram;
            }
            else {
                logM[curpos]=LOG0;
            }
        }
        /* Update the culumn order */
        cur = 1-cur;
        old = 1-old;
    }
    aux = logM[curpos];
    free(logM);
	free(logTriangularCoefficients);
    /* Return the logarithm of the Global Alignment Kernel */    
    return aux;  
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

gsl_matrix* GRAIL(fftw_complex** x,unsigned int* xdim, fftw_complex** sample,unsigned int* samdim, double gamma, size_t r)
{
	unsigned int i, j;
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
			W[i*samdim[1] + j] = sink(sample[i], samdim[1], sample[j], samdim[1], gamma);
			printf("%d %d\n",i, j);
		}
	}
	printf("\n\n");
	for (i = 0; i < xdim[0]; i++)
	{
		for (j = 0; j < samdim[0]; j++)
		{
			E[i*xdim[1] + j] = sink(x[i], xdim[1], sample[j], samdim[1], gamma);
			printf("%d %d\n", i, j);
		}
	}


	gsl_matrix_view mW = gsl_matrix_view_array(W, samdim[0], samdim[0]);


	gsl_vector* eval = gsl_vector_alloc(samdim[0]);
	gsl_matrix* evec = gsl_matrix_alloc(samdim[0], samdim[0]);

	gsl_eigen_symmv_workspace* w = gsl_eigen_symmv_alloc(samdim[0]);

	gsl_eigen_symmv(&mW.matrix, eval, evec, w);

	double* inVa_array = (double*)calloc(samdim[0] * samdim[0],sizeof(double));


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
	gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1, &mE.matrix, intermediate,0, Zexact);

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
	qsort(evalz->data, evalz->size, sizeof(double),comp);


	double* varexplained = (double*)malloc(sizeof(double) * evalz->size);
	
	double sum = 0;
	for (i = 0; i < evalz->size; i++)
	{
		sum += evalz->data[i];
	}

	varexplained[0] = evalz->data[0]/sum;
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



int main()
{
		unsigned int h = 4;
		unsigned int v = 4;
		size_t r = 2;
		fftw_complex** x = (fftw_complex**)malloc(sizeof(fftw_complex*) * h);
		unsigned int xdim[2] = { h,v };
		for (unsigned int i = 0; i < h; i++)
		{
			x[i] = (fftw_complex*)malloc(sizeof(fftw_complex) * v);
			for (unsigned int j = 0; j < v; j++)
			{
				x[i][j][0] = i * v + j;
				x[i][j][1] = 0;
			}
		}

	fftw_complex* y[2] = { x[1],x[2] };
	unsigned int ydim[2] = { 2,v };
	gsl_matrix* result = GRAIL(x, xdim, y, ydim, .5, 2);
	printf("\n\n");
	for (unsigned int i = 0; i < h; i++)
	{
		for (unsigned int j = 0; j < r; j++)
		{
			printf("%.2f  ", gsl_matrix_get(result, i, j));
		}
		printf("\n");
	}

	gsl_matrix_free(result);
}