#include <stdlib.h>
#include <stdio.h>
#include<math.h>
#include<time.h>

// row-wise nonzero counting
  __global__ void
count_nonzero(int n, float *A, int *nz){

    int row = blockIdx.x*blockDim.x + threadIdx.x;
    int count = 0;
    for (int i=0; i<n; i++)
        if (A[row*n+i] != 0) 
            count += 1;
    nz[row] = count;
}

// partial sum calculation
  __global__ void
partial_sum(int n, int *nz, int *x){
    int row = blockIdx.x*blockDim.x + threadIdx.x;
    int sum = 0;
    for (int i=0; i<row+1; i++){
            sum += nz[i];
    }
    x[row+1] = sum;
}

// collection of nonzero value and its column index
  __global__ void
value_colidx(int n, int *nz, float *A, int *psum, int *x, float *y){

    int row = blockIdx.x*blockDim.x + threadIdx.x;
    int count = 0;
    int thread_start = psum[row];
    for (int i=0; i<n; i++){
        // stop if all nz are found
        if (count <nz[row]){
            if (A[row*n+i] != 0){
                x[thread_start+count] = i;
                y[thread_start+count] = A[row*n+i];
                count += 1;
            }
        }
    }
}


int main(int argc, char **argv) {

    //int n=2000;
    int n = atoi(argv[1]);
    int memSize = n*sizeof(int);
    cudaEvent_t start, stop, start_all, stop_all;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&start_all);
    cudaEventCreate(&stop_all);

    int *nz;
    int*d_nz;
    nz = (int*) malloc (n*sizeof(*nz));
    cudaMalloc( (void**) &d_nz, memSize);

    float *A, *d_A;
    A = (float*) malloc (n*n*sizeof(*A));
    cudaMalloc( (void**) &d_A, n*n*sizeof(float));
   
    // initialize matrix
    for(int j=0; j<n; j++){
        nz[j] = 0;
        for(int k=0; k<n; k++){
            A[j*n+k] = 0;
            if ((j+k) %2 != 0)    A[j*n+k] = (float) j+k;
        }
    }

    cudaMemcpy( d_A, A, n*memSize, cudaMemcpyHostToDevice);
    cudaMemcpy( d_nz, nz, n*sizeof(int), cudaMemcpyHostToDevice);

    dim3 block(1);
    dim3 grid(n);

    // counting number of nonzero elements from each row
    cudaEventRecord(start);
    cudaEventRecord(start_all);
    count_nonzero<<<grid,block>>>(n, d_A, d_nz);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("runtime   nz[s]: %f\n", milliseconds/1000.0);

    // calculating partial sum
    int *psum, *d_psum;
    psum = (int*) malloc ((n+1)*sizeof(*psum));
    cudaMalloc( (void**) &d_psum, n*memSize);
    
    // no need nz in CPU in fact
    //cudaMemcpy( d_nz, nz, memSize, cudaMemcpyHostToDevice);

    cudaEventRecord(start);
    partial_sum<<<grid,block>>>(n, d_nz, d_psum);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("runtime psum[s]: %f\n", milliseconds/1000.0);

    // copy psum (rowptr) from GPU to CPU
    cudaMemcpy( psum, d_psum, (n+1)*sizeof(*psum), cudaMemcpyDeviceToHost);
    psum[0] = 0;

    // size of CSR
    int m = psum[n];

    int *colidx, *d_colidx;
    colidx = (int*) calloc(m, sizeof(int));
    cudaMalloc( (void**) &d_colidx, m*sizeof(int));

    float *value, *d_value;
    value = (float*) malloc (m*sizeof(float));
    cudaMalloc( (void**) &d_value, m*sizeof(float));

    cudaMemcpy( d_colidx, colidx, m*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy( d_value, value, m*sizeof(float), cudaMemcpyHostToDevice);

    // collect value and column index based on partial sum
    cudaEventRecord(start);
    value_colidx<<<grid,block>>>(n, d_nz, d_A, d_psum, d_colidx, d_value);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventRecord(stop_all);

    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("runtime  CSR[s]: %f\n", milliseconds/1000.0);

    cudaEventElapsedTime(&milliseconds, start_all, stop_all);
    printf("runtime  ALL[s]: %f\n", milliseconds/1000.0);

    // copy colidx, value from GPU to CPU
    cudaMemcpy( colidx, d_colidx, m*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy( value, d_value, m*sizeof(float), cudaMemcpyDeviceToHost);
}
