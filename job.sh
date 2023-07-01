#!/bin/sh
#PBS -q calc
#PBS -l select=5:ncpus=80:mpiprocs=80
#PBS -N smc_inv_mpi

cd /home/nakao/smc_inversion
make main
mpiexec -n 400 ./main 20 20 > mpi80000.log

