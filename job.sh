#!/bin/sh
#PBS -q calc
#PBS -l select=5:ncpus=80:mpiprocs=80
#PBS -N smc_inv_mpi

cd /home/nakao/smc_inversion
make main
mpiexec -bind-to core -n 400 ./main 20 20 > mpi20000.log

