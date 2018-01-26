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
    double *x = malloc(n*sizeof(double));
    double *y = malloc(n*sizeof(double));
    double *A = malloc(n*n*sizeof(double));
    
    // random seed based on clock time
    srand(time(NULL));
    
    // initialize random vectors and matrix
    for(int i=0; i<n; i++){
        x[i] = rand();
        y[i] = rand();
        for(int j=0; j<n; j++){
            A[i*n+j] = rand();
        }
    }
    
    // tic
    start_time = clock(); /* mircosecond */

    // matrix-vector multiplication with addition
    for(int i=0; i<n; i++){
        for(int j=0; j<n; j++){
            y[i] += A[i*n+j]*x[j];
        }
    }
      
    // toc
    end_time = clock(); /* mircosecond */
    total_time = (float)(end_time - start_time)/CLOCKS_PER_SEC;

    // print square root of temp to display norm of x
    printf("Time : %f sec \n", total_time);


    return 0;
}
