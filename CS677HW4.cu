/*Unrolled version of kernel that computes everything working with shared and constant memory*/
/*Worked in taking arguments*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

__constant__ float INP1[128];

bool relativelyEqual(float a, float b, float maxreldiff = FLT_EPSILON){
  const float difference = fabs(a-b);
  a = fabs(a);
  b = fabs(b);
  const float scaledepsilon = maxreldiff*max(a,b);
  return difference <=scaledepsilon;
}

__global__ void kernel(float ip[], float op[]){
  __shared__ float local[128];
  int tx = threadIdx.x;
  int t = tx +blockIdx.x*blockDim.x;
  float temp_res=0;
  int i;
  local[tx] = ip[t];
  __syncthreads();
  for(i=0;i<=tx;i++){
    temp_res+=local[i];
  }
  for(i=0;i<128;i++){
    temp_res += (INP1[tx]*INP1[i]);
  }
  op[t] = temp_res;
}

__global__ void kerneltwo(float ip[], float op[]){
  __shared__ float local[128];
  int tx = threadIdx.x;
  int t = tx +blockIdx.x*blockDim.x;
  float temp_res=0;
  int i;
  local[tx] = ip[t];
  __syncthreads();
  for(i=0;i<=tx;i++){
    temp_res+=local[i];
  }
  for(i=0;i<64;i++){
    temp_res += (INP1[tx]*INP1[i]);
    temp_res += (INP1[tx]*INP1[i+64]);
  }
  op[t] = temp_res;
}

__global__ void kernelfour(float ip[], float op[]){
  __shared__ float local[128];
  int tx = threadIdx.x;
  int t = tx +blockIdx.x*blockDim.x;
  float temp_res=0;
  int i;
  local[tx] = ip[t];
  __syncthreads();
  for(i=0;i<=tx;i++){
    temp_res+=local[i];
  }
  for(i=0;i<32;i++){
    temp_res += (INP1[tx]*INP1[i]);
    temp_res += (INP1[tx]*INP1[i+32]);
    temp_res += (INP1[tx]*INP1[i+64]);
    temp_res += (INP1[tx]*INP1[i+96]);
  }
  op[t] = temp_res;
}

__global__ void kerneleight(float ip[], float op[]){
  __shared__ float local[128];
  int tx = threadIdx.x;
  int t = tx +blockIdx.x*blockDim.x;
  float temp_res=0;
  int i;
  local[tx] = ip[t];
  __syncthreads();
  for(i=0;i<=tx;i++){
    temp_res+=local[i];
  }
  for(i=0;i<16;i++){
    temp_res += (INP1[tx]*INP1[i]);
    temp_res += (INP1[tx]*INP1[i+16]);
    temp_res += (INP1[tx]*INP1[i+32]);
    temp_res += (INP1[tx]*INP1[i+48]);
    temp_res += (INP1[tx]*INP1[i+64]);
    temp_res += (INP1[tx]*INP1[i+80]);
    temp_res += (INP1[tx]*INP1[i+96]);
    temp_res += (INP1[tx]*INP1[i+112]);
  }
  op[t] = temp_res;
}

int main( int argc, char* argv[] )
{
  int factor = atoi(argv[1]);
  float time;
  //Host input vectors
  float *input1;
  float *input2;

  int i, j;
  int n;
  n = 128;

  //Size in bytes for each vector
  size_t bytes1 = n*sizeof(float);
  size_t bytes2 = n*n*sizeof(float);

  //Allocating memory for host vectors
  input1 = (float*)malloc(bytes1);
  input2 = (float*)malloc(bytes2);

  //Device input vectors
  float *d_ip1;
  float *d_ip2;

  //Allocate memory for vectors on GPU
  cudaMalloc(&d_ip1, bytes1);
  cudaMalloc(&d_ip2, bytes2);

  //Initializing input vectors
  for(i=0;i<n;i++){
    for(j=0;j<n;j++){
      input2[n*i+j] = (float)(rand()/1000000000);
    }
    input1[i]=(float)(rand()/1000000000);
  }
/***********************---CPU code---*******************************/

  float cresult[128][128];
  float temp[128];
  int k;
  clock_t Cstart, Cstop;
  double ctime;
  Cstart = clock();
  for(i=0;i<128;i++){
    temp[i]=0;
    for(j=0;j<128;j++){
      temp[i]+=input2[i*128+j];
      cresult[i][j]=temp[i];
      for(k=0;k<128;k++){
        cresult[i][j] += input1[j]*input1[k];
      }
      //printf("cresult[%d][%d] : %f",i,j,cresult[i][j]);
    }
  }
  Cstop = clock();
  ctime = (((double)(Cstop-Cstart))/CLOCKS_PER_SEC)*1000 ;
  printf("Time taken on CPU: %fms\n", ctime);

/********************---end of CPU code---***************************/
  //Copying input1 to constant memory for better use of memory hierarchy
  cudaMemcpyToSymbol(INP1, input1, bytes1);

  //Host output vector
  float *result;

  //Allocating memory for host output vectors
  result = (float*)malloc(bytes2);

  //Device output vector
  float *d_result;

  //Allocating memory for device output vectors
  cudaMalloc(&d_result, bytes2);

  // Copy host vectors to device
  cudaMemcpy(d_ip2, input2, bytes2, cudaMemcpyHostToDevice);

  // Number of threads in each thread block
  dim3 dimBlock(128,1);
  // Number of thread blocks in grid
  dim3 dimGrid(128,1);

  cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start,0);

  if (factor == 1) {
    printf("Unrolled Kernel version \n");
		kernel<<<dimGrid, dimBlock>>>(d_ip2, d_result);
	}

	else if (factor == 2) {
    printf("Unrolling factor of two Kernel version \n");
		kernel<<<dimGrid, dimBlock>>>(d_ip2, d_result);

	}
	else if (factor == 4) {
    printf("Unrolling factor of four Kernel version  \n");
		kernel<<<dimGrid, dimBlock>>>(d_ip2, d_result);
	}

	else if (factor == 8) {
    printf("Unrolling factor of eight Kernel version \n");
		kernel<<<dimGrid, dimBlock>>>(d_ip2, d_result);
	}

  cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	time = 0;
	cudaEventElapsedTime(&time, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
	printf("Time taken on GPU: %lf ms\n", time);


  cudaMemcpy(result, d_result, bytes2, cudaMemcpyDeviceToHost);
  int count = 0;
  for(i=0;i<128;i++){
    for(j=0;j<128;j++){
      if(floor(cresult[i][j])!=floor(result[i*128+j])){
        //(ceil)(result[i*128+j])==(ceil)(cresult[i][j])
        count+=1;
      }
    }
  }
  printf("count: %d\n",count);
  if(count==0){
    printf("Verified correctness of CPU and GPU results\n");
  }

  // Release device memory
  cudaFree(d_ip1);
  cudaFree(d_ip2);
  cudaFree(d_result);

  // Release host memory
  free(input1);
  free(input2);
  free(result);

  return 0;
}
