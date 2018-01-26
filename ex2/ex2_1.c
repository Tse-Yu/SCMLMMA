#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>

int main(int argc, char **argv){
    // initialize time
    clock_t start_time, end_time;
    float total_time = 0;

    // initialize vector and matrix
    int n = atoi(argv[1]);
    double *A = malloc(n*n*sizeof(double));
    
    // random seed based on clock time
    srand(time(NULL));
    
    // initialize matrix using tri-diagonal type matrix
    for(int i=0; i<n; i++){
        for(int j=0; j<n; j++){
            A[i*n+j] = rand();
            A[i*n+j] = 0;
            if (i==j) A[i*n+j] = -2;
            if (i-j==1) A[i*n+j] = 1;
            if (j-i==1) A[i*n+j] = 1;
        }
    }

    /*// print out this matrix
    for(int i=0; i<n; i++){
        for(int j=0; j<n; j++){
             printf("%2f ", A[i*n+j]);
             if (j%n==n-1) printf("\n");
        }
    }*/
    
    // tic
    start_time = clock(); /* mircosecond */
  
   // LU-factorization
    for(int i=0; i<n; i++) {
	for(int j=i+1; j<n; j++) {
		A[j*n+i] /= A[i*n+i];
		for(int k=i+1; k<n; k++) {
			A[j*n+k] -= A[j*n+i] * A[i*n+k];
		}
        }
    }
    // toc
    end_time = clock(); /* mircosecond */
    total_time = (float)(end_time - start_time)/CLOCKS_PER_SEC;


//void dgetrf( const MKL_INT* m, const MKL_INT* n, double* a, const MKL_INT* lda,
//             MKL_INT* ipiv, MKL_INT* info );

    /*int *ipiv = malloc(n*sizeof(int));
    int flag;
    dgetrf(&n, &n, C, &n, ipiv, &flag);
    */
 
    // print computational time
    printf("Time : %f sec \n", total_time);
    /*// print its LU-decomposition in-place
    for(int i=0; i<n; i++){
        for(int j=0; j<n; j++){
             printf("%02f ", A[i*n+j]);
             if (j%n==n-1) printf("\n");
        }
    }*/


    return 0;
}
