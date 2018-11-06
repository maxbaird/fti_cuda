#! /usr/bin/bash
srun --mpi=pmi2 --gres=gpu --nodes=1 --partition=amd-shortq --ntasks=32 ./fti_cuda.out 1000
