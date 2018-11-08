#! /usr/bin/bash
srun --mpi=pmi2 --gres=gpu --nodes=1 --partition=amd-shortq --ntasks=32 ./fti_cuda.out 100
#srun --mpi=pmi2 --gres=gpu --nodes=1 --partition=amd-shortq --ntasks=32 cuda-memcheck ./fti_cuda.out 100
