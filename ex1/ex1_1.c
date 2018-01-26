#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>

int main(int argc, char **argv){
    // initial time, vector size and allocate vector
    clock_t start_time, end_time;
    float total_time = 0;

    int n = atoi(argv[1]);
    double *x = malloc(n*sizeof(double));
    
    // random seed based on clock time
    srand(time(NULL));
    
    // initialize random vector
    for(int i=0; i<n; i++){
        x[i] = rand();
    }

    struct timespec tp;
    clockid_t clk_id;
    clk_id = CLOCK_MONOTONIC;
 
    // tic
    start_time = clock();

    // initialize cumulative sum
    double temp = 0.0;
    // sum over vector x
    for(int i=0; i<n; i++){
        temp += x[i]*x[i];
    }
    
    // toc
    end_time = clock();
    total_time = (float)(end_time - start_time)/CLOCKS_PER_SEC;

    // print square root of temp to display norm of x
    //printf("%lf\n", sqrt(temp));
    // print computational time
    printf("Time : %f sec \n", total_time);


    return 0;
}
