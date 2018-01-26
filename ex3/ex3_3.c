#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>
#include<omp.h>

void diag_fact(int blockSize, double *A) {
    for(int i=0; i<blockSize; i++) {
        for(int j=i+1; j<blockSize; j++) {
                A[j*blockSize+i] /= A[i*blockSize+i];
                for(int k=i+1; k<blockSize; k++) {
                        A[j*blockSize+k] -= A[j*blockSize+i] * A[i*blockSize+k];
                }
        }
    }
}

void col_update(int n, double *Aij, double *Aii ) {
    double inv_;
    for(int j=0; j<n; j++){ 
        inv_ = 1./Aii[j*n+j];
        for(int i=0; i<n; i++){
            Aij[i*n+j] *= inv_;
        }
        for(int i=0; i<n; i++){
            for(int k=j+1; k<n; k++){
                Aij[i*n+k] -= Aii[i*n+j]*Aij[i*n+j];
            }
        }
    }
}

void row_update(int n, double *Aij, double *Aii ) {
    for(int i=0; i<n; i++){
        for(int k=i+1; k<n; k++){
            for(int j=0; j<k; j++){
                Aij[k*n+j] -= Aii[i*n+j]*Aij[i*n+j];
            }
        }
    }
}

void trail_update(int n, double *Ajk, double *Aik, double *Aji ) {
    for (int i = 0; i<n; i++) {
        for (int j = 0; j<n; j++) {
            for (int k = 0; k<n; k++) {
                Ajk[i*n+j] -= Aji[i*n+k]*Aik[k*n+j];
            }
        }
    }
}



int main(int argc, char **argv){
    // initialize time
    clock_t start_time, end_time;
    float total_time = 0;

    // initialize vector and matrix
    int n = atoi(argv[1]);
    int blockSize = 2;
    int num_blocks = (n+1) / blockSize;
    int blockSize_last = n - blockSize*num_blocks;
    printf("There are %d blocks with size %d\n", num_blocks, blockSize);
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

    // blockwise A in to B[i][j]
    double ***B;
    B = (double ***)malloc(num_blocks*sizeof(double **));

    for(int i=0; i<num_blocks; i++){
        B[i] = (double **) malloc(num_blocks*sizeof(double *));
        for(int j=0; j<num_blocks; j++){
            B[i][j] = (double *) malloc(blockSize*blockSize*sizeof(double *));
            for(int p=0; p<blockSize; p++){
                for(int q=0; q<blockSize; q++){
                    int idx = (i*blockSize+p)*n + (j*blockSize+q);
                    B[i][j][p*blockSize+q] = A[idx];
                }
            }
        }
    }

    // tic
    start_time = clock(); /* mircosecond */

    // LU-factorization
    diag_fact(n, A);
    // block-LU
    #pragma omp parallel
    for(int i=0; i<num_blocks; i++) {
        #pragma omp single
        diag_fact( blockSize, B[i][i] );        
        #pragma omp for
        for(int j=i+1; j<num_blocks; j++){
            col_update(blockSize, B[j][i], B[i][i] );
            row_update(blockSize, B[i][j], B[i][i] );
        }
        #pragma omp for
        for(int j=i+1; j<num_blocks; j++) {
            for(int k=i+1; k<num_blocks; k++)
                trail_update(blockSize, B[j][k], B[i][k], B[j][i] );
            }
    }

    // toc
    end_time = omp_get_wtime();
    total_time = end_time - start_time;

    // print computational time
    printf("Time : %f sec \n", total_time);
    // print its LU-decomposition in-place
    printf("usual LU:\n");
    for(int i=0; i<n; i++){
        for(int j=0; j<n; j++){
             printf("%02f ", A[i*n+j]);
             if (j%n==n-1) printf("\n");
        }
    }

    printf("\n");
    for(int i=0; i<blockSize; i++){
        for(int j=0; j<blockSize; j++){
             printf("%02f ", B[0][1][i*n+j]);
             if (j%blockSize==blockSize-1) printf("\n");
        }
    }

    
    return 0;
}
