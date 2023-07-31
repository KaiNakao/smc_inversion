#!/bin/sh
#PBS -q calc
#PBS -l select=5:ncpus=80:mpiprocs=20
#PBS -N smc_inv

cd /home/nakao/smc_inversion
make main

mpiexec -n 100 bash -c "ulimit -s unlimited"
mpiexec -n 100 ./main > var_faultsize_max3.log

