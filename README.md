# fti\_cuda
This project demonstrates how FTI can be used with CUDA. It performs a simple
vector addition of C = A + B by dividing the vector size evenly among the number
of MPI processes. Each process then launches a CUDA kernel to compute their
partition of the vector.

## Compiling
To compile the following environment variables need to be set:
+ MPI\_HOME
+ CUDA\_HOME
+ FTI\_HOME

These variables should point to the home directory of MPI, CUDA and FTI
respectively. To compile run `make`.

## Running
Execute the binary with the following two arguments
1. vector-size
2. iterations

+ _vector-size_ Specifies the length of the vector
+ _iterations_ Specifies how many times each MPI process should launch its kernel 

You will need to have FTI built and configured for a successful run. For more
information on FTI see their [github](https://github.com/leobago/fti) repository.

## Example Run
The following will spawn 8 MPI processes and each process will execute their
kernel 10 times.

`mpirun -np 8 ./fti_cuda.out 10000 10`
