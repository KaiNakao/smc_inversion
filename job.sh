#!/bin/sh
#PBS -q calc
#PBS -l select=1:ncpus=1:mpiprocs=1
#PBS -N smc_inv

cd /home/nakao/smc_inversion
make main

export OMP_NUM_THREADS=1
mpiexec -n 1 bash -c "ulimit -s unlimited"
mpiexec -n 1 ./main > tmp.log

