#! /bin/bash
#export MV2_ENABLE_AFFINITY=0
#export MV2_MPI_PMI_LIBRARY=/cm/shared/apps/slurm/17.02.2/lib64/libpmi2.so
#export MV2_MPI_PMI2=yes
#srun --mpi=mvapich --gres=gpu --partition=amd-longq --nodes=2 --ntasks=64 ./fti_cuda.out 10000000 10000
srun --mpi=pmi2 --gres=gpu --nodes=1 --partition=amd-shortq --ntasks=32 ./fti_cuda.out 100000000
#srun --mpi=mvapich --gres=gpu --partition=amd-longq --nodes=1 --ntasks=32 ./fti_cuda.out 11948575 1

#Let me know when this finishes
#mail -s "Experiment Done" maxbaird.gy@gmail.com << EOT
#The experiment has concluded!
#EOT
