#!/bin/sh
#PBS -q calc
#PBS -l select=5:ncpus=80:mpiprocs=4
#PBS -N smc_inv

cd /home/nakao/smc_inversion
make main

mpiexec -n 20 bash -c "ulimit -s unlimited"
mpiexec -n 20 ./main > tmp.log

