#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <fti.h>

#define CUDA_ERROR_CHECK(fun)                                                                   \
do{                                                                                             \
    cudaError_t err = fun;                                                                      \
    if(err != cudaSuccess)                                                                      \
    {                                                                                           \
      fprintf(stderr, "Cuda error %d %s:: %s\n", __LINE__, __func__, cudaGetErrorString(err));  \
      exit(EXIT_FAILURE);                                                                       \
    }                                                                                           \
}while(0);

#define KERNEL_ERROR_CHECK()                                                                    \
do{                                                                                             \
  cudaError_t err = cudaGetLastError();                                                         \
  if (err != cudaSuccess)                                                                       \
  {                                                                                             \
    fprintf(stderr, "Kernel error %d %s:: %s\n", __LINE__, __func__, cudaGetErrorString(err));  \
    exit(EXIT_FAILURE);                                                                         \
  }                                                                                             \
}while(0);

FTIT_type U_LL;

__global__ void vector_add(const unsigned short int *a, const unsigned short int *b, unsigned short int *c, unsigned long long n)
{
  /* Get our global thread ID */
  unsigned long long id = blockIdx.x*blockDim.x+threadIdx.x;

  if(id > n)
  {
    return;
  }

  c[id] = a[id] + b[id];
}

int main(int argc, char *argv[])
{
  if(argc != 3)
  {
    fprintf(stderr, "Usage: %s <vector-size> <iterations>\n", argv[0]);
    exit(EXIT_FAILURE);
  }
  
  int rank_id = 0;
  int processes = 0;

  char config_path[] = "config.fti";

  MPI_Init(&argc, &argv);
  FTI_Init(config_path, MPI_COMM_WORLD);
  MPI_Comm_size(FTI_COMM_WORLD, &processes);
  MPI_Comm_rank(FTI_COMM_WORLD, &rank_id);

  FTI_InitType(&U_LL, sizeof(unsigned long long));

  unsigned long long vector_size = strtoull(argv[1], NULL, 10);
  unsigned long long iterations = strtoull(argv[2], NULL, 10);
  size_t size = vector_size * sizeof(unsigned short int);

  unsigned short int *h_a = (unsigned short int *)malloc(size);
  unsigned short int *h_b = (unsigned short int *)malloc(size);
  unsigned short int *h_c = (unsigned short int *)malloc(size);

  if(h_a == NULL || h_b == NULL || h_c == NULL)
  {
    fprintf(stderr, "Failed to allocate %zu bytes!\n", size);
    exit(EXIT_FAILURE);
  }

  unsigned short int *d_a = NULL;
  unsigned short int *d_b = NULL;
  unsigned short int *d_c = NULL;

  unsigned long long i = 0;
  unsigned long long j = 0;
  unsigned long long local_sum = 0;

  CUDA_ERROR_CHECK(cudaMalloc((void **)&d_a, size));
  CUDA_ERROR_CHECK(cudaMalloc((void **)&d_b, size));
  CUDA_ERROR_CHECK(cudaMalloc((void **)&d_c, size));

  /* Initialize vectors */
  for(i = 0; i < vector_size; i++)
  {
    h_a[i] = 1;
    h_b[i] = 1; 
  } 

  CUDA_ERROR_CHECK(cudaMemcpy((void *)d_a, (const void*)h_a, size, cudaMemcpyHostToDevice));
  CUDA_ERROR_CHECK(cudaMemcpy((void *)d_b, (const void*)h_b, size, cudaMemcpyHostToDevice));
 
  unsigned long long block_size = 1024; 
  unsigned long long grid_size = (unsigned long long)(ceill((long double)vector_size/(long double)block_size));

  FTI_Protect(0, &i, 1, U_LL);
  FTI_Protect(1, &local_sum, 1, U_LL);

  for(i = 0; i < iterations; i++)
  {
    FTI_Snapshot();
    vector_add<<<grid_size, block_size>>>(d_a, d_b, d_c, vector_size);
    KERNEL_ERROR_CHECK();
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());
  
    CUDA_ERROR_CHECK(cudaMemcpy((void *)h_c, (const void *)d_c, size, cudaMemcpyDeviceToHost));
 
    for(j = 0; j < vector_size; j++)
    {
      local_sum = local_sum + h_c[j];
    }
  }

  if(rank_id == 0)
  {
    unsigned long long global_sum = local_sum;
    int i = 0;
    for(i = 1; i < processes; i++)
    {
      MPI_Recv(&local_sum, 1, MPI_UNSIGNED_LONG_LONG, i, 0, FTI_COMM_WORLD, MPI_STATUS_IGNORE);
      global_sum = global_sum + local_sum;
    }

    if((2 * vector_size * processes * iterations) == global_sum)
    {
      fprintf(stdout, "Result: Pass\n");
    }
    else
    {
      fprintf(stderr, "Result: Failed\n");
    }

    fprintf(stdout, "Sum: %llu\n", global_sum);
  }
  else
  {
    MPI_Send(&local_sum, 1, MPI_UNSIGNED_LONG_LONG, 0, 0, FTI_COMM_WORLD);
  }

  /* Housekeeping... */
  free(h_a);
  free(h_b);
  free(h_c);

  CUDA_ERROR_CHECK(cudaFree((void *)d_a));
  CUDA_ERROR_CHECK(cudaFree((void *)d_b));
  CUDA_ERROR_CHECK(cudaFree((void *)d_c));

  CUDA_ERROR_CHECK(cudaDeviceReset());
  FTI_Finalize();
  MPI_Finalize();

  return EXIT_SUCCESS;
}
