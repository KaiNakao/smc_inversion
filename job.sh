#!/bin/sh
#PBS -q calc
#PBS -l select=5:ncpus=80:mpiprocs=4
#PBS -N smc_inv

cd /home/nakao/smc_inversion
make main

export OMP_NUM_THREADS=20
mpiexec -n 20 bash -c "ulimit -s unlimited"
mpiexec -n 20 ./main > toy.log

