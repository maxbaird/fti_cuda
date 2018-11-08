#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <unistd.h>
#include <fti.h>

#define MASTER 0
#define DO_IF_MASTER()          \
  if(rank_id == MASTER)         

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

#define BUFFER 128

/*
   Holds the chunk information for each MPI process.
*/
typedef struct {
  unsigned long long lower; /* Index in vector from which the current MPI process should start */
  unsigned long long upper; /* Index in vector from which current MPI process should end */
  unsigned long long n_items; /* Number of items from lower to upper */
}Chunk_Info_t;

Chunk_Info_t calculate_chunk(int processes, int rank_id, unsigned long long *vector_size)
{
  Chunk_Info_t chunk_info;

  //The array size to be created from this value is zero based
  //So this needs to be increased so that we compute up to the
  //value provided at runtime.
  (*vector_size)++;

  //Next get the size of the chunk the current process needs 
  unsigned long long local_vector_size = (*vector_size) / processes;

  //Calculates the start position in the vector for the current process 
  chunk_info.lower = local_vector_size * rank_id;

  //Calculate the end of the chunk
  chunk_info.upper = chunk_info.lower + (local_vector_size - 1);

  //The last process needs to handle any extra vector elements
  if(rank_id == (processes - 1))
  {
    chunk_info.upper = chunk_info.upper + (*vector_size % processes);
  }

  //Finally determine how many elements are in the chunk
  chunk_info.n_items = (chunk_info.upper - chunk_info.lower) + 1;
  
  return chunk_info;
}

FTIT_type U_LL;

__global__ void 
//vector_add(const unsigned long long *a, const unsigned long long *b, unsigned long long *c, unsigned long long n)
FTI_KERNEL_DEF(vector_add, const unsigned long long *a, const unsigned long long *b, unsigned long long *c, unsigned long long n, int rank_id)
{
  FTI_CONTINUE();
  /* Get our global thread ID */
  unsigned long long id = blockIdx.x*blockDim.x+threadIdx.x;

  if(id > n)
  {
    return;
  }

  c[id] = a[id] + b[id];
}

__global__ void increment(unsigned long long *c, unsigned long long n)
{

  unsigned long long id = blockIdx.x*blockDim.x+threadIdx.x;

  if(id > n)
  {
    return;
  }

  c[id] = c[id] + 1;
}

void print(FILE *stream, const char *str)
{
  fprintf(stream, str);
  fflush(stream);
}

int main(int argc, char *argv[])
{
  if(argc != 2)
  {
    fprintf(stderr, "Usage: %s <vector-size>\n", argv[0]);
    exit(EXIT_FAILURE);
  }
  
  int rank_id = 0;
  int processes = 0;

  char config_path[] = "config.fti";
  char str[BUFFER];

  MPI_Init(&argc, &argv);
  FTI_Init(config_path, MPI_COMM_WORLD);
  MPI_Comm_size(FTI_COMM_WORLD, &processes);
  MPI_Comm_rank(FTI_COMM_WORLD, &rank_id);

  double start = MPI_Wtime();

  FTI_InitType(&U_LL, sizeof(unsigned long long));

  unsigned long long vector_size = strtoull(argv[1], NULL, 10);
  //unsigned long long iterations = strtoull(argv[2], NULL, 10);
  //int execute_first_kernel_loop = strtol(argv[3], NULL, 10);

  Chunk_Info_t chunk_info = calculate_chunk(processes, rank_id, &vector_size);

  size_t size = chunk_info.n_items * sizeof(unsigned long long);

  unsigned long long *h_a = (unsigned long long *)malloc(size);
  unsigned long long *h_b = (unsigned long long *)malloc(size);
  unsigned long long *h_c = (unsigned long long *)malloc(size);

  if(h_a == NULL || h_b == NULL || h_c == NULL)
  {
    fprintf(stderr, "Failed to allocate %zu bytes!\n", size);
    exit(EXIT_FAILURE);
  }

  unsigned long long *d_a = NULL;
  unsigned long long *d_b = NULL;
  unsigned long long *d_c = NULL;

  unsigned long long i = 0;
  unsigned long long j = 0;
  unsigned long long local_sum = 0;

  CUDA_ERROR_CHECK(cudaMalloc((void **)&d_a, size));
  CUDA_ERROR_CHECK(cudaMalloc((void **)&d_b, size));
  CUDA_ERROR_CHECK(cudaMalloc((void **)&d_c, size));

  unsigned long long idx = 0;
  /* Initialize vectors */
  for(i = chunk_info.lower; i <= chunk_info.upper; i++)
  {
    h_a[idx] = i;
    h_b[idx] = i; 
    idx++;
  } 

  CUDA_ERROR_CHECK(cudaMemcpy((void *)d_a, (const void*)h_a, size, cudaMemcpyHostToDevice));
  CUDA_ERROR_CHECK(cudaMemcpy((void *)d_b, (const void*)h_b, size, cudaMemcpyHostToDevice));
 
  unsigned long long block_size = 1024; 
  unsigned long long grid_size = (unsigned long long)(ceill((long double)chunk_info.n_items/(long double)block_size));

  FTI_Protect(0, &i, 1, U_LL);
  FTI_Protect(1, &local_sum, 1, U_LL);
  FTI_Protect(2, d_c, chunk_info.n_items, U_LL);

  local_sum = 0;

  if(FTI_Status() != 0)
  {
    FTI_Recover();
  }

  //vector_add<<<grid_size, block_size>>>(d_a, d_b, d_c, chunk_info.n_items);
  FTI_Protect_Kernel(42, 0.00001, vector_add, grid_size, block_size,0,0,d_a, d_b, d_c, chunk_info.n_items, rank_id);
  KERNEL_ERROR_CHECK();
  CUDA_ERROR_CHECK(cudaDeviceSynchronize());
  
  CUDA_ERROR_CHECK(cudaMemcpy((void *)h_c, (const void *)d_c, size, cudaMemcpyDeviceToHost));
 
  for(j = 0; j < chunk_info.n_items; j++)
  {
    local_sum = local_sum + h_c[j];
  }
  
  unsigned long long tmp = local_sum;

  local_sum = tmp;

  increment<<<grid_size, block_size>>>(d_c, chunk_info.n_items);
  KERNEL_ERROR_CHECK();
  CUDA_ERROR_CHECK(cudaDeviceSynchronize());
  CUDA_ERROR_CHECK(cudaMemcpy((void *)h_c, (const void *)d_c, size, cudaMemcpyDeviceToHost));

  for(j = 0; j < chunk_info.n_items; j++)
  {
    local_sum = local_sum + h_c[j];
  }

  if(rank_id == MASTER)
  {
    unsigned long long global_sum = local_sum;
    int i = 0;
    for(i = 1; i < processes; i++)
    {
      MPI_Recv(&local_sum, 1, MPI_UNSIGNED_LONG_LONG, i, 0, FTI_COMM_WORLD, MPI_STATUS_IGNORE);
      global_sum = global_sum + local_sum;
    }

    unsigned long long expected_global_sum = 0;
    unsigned long long j = 0;
    for(j = 0; j < vector_size; j++)
    {
      expected_global_sum = expected_global_sum + (j + j);
    }
    expected_global_sum = (expected_global_sum * 2) + vector_size;
    if(expected_global_sum == global_sum)
    {
      print(stdout, "Result: Pass\n");
    }
    else
    {
      sprintf(str, "%llu != %llu\n", expected_global_sum, global_sum);
      print(stderr, str);
      print(stderr, "Result: Failed\n");
    }

    sprintf(str, "Sum: %llu\n", global_sum);
    print(stdout, str);
    double end = MPI_Wtime();
    sprintf(str, "Time: %f seconds\n", end - start);
    fprintf(stdout, str);
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

  //CUDA_ERROR_CHECK(cudaDeviceReset());
  FTI_Finalize();
  MPI_Finalize();

  return EXIT_SUCCESS;
}
