#! /bin/bash
srun --mpi=pmi2 --gres=gpu --nodes=1 --partition=amd-shortq --ntasks=32 ./fti_cuda.out 100000000





#srun --mpi=mvapich --gres=gpu --partition=amd-longq --nodes=2 --ntasks=64 ./fti_cuda.out 10000000 10000
