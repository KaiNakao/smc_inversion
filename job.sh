#!/bin/sh
#PBS -q calc
#PBS -l select=4:ncpus=80:mpiprocs=20
#PBS -N smc_inv

cd /home/nakao/smc_inversion
make main

mpiexec -n 80 bash -c "ulimit -s unlimited"
mpiexec -n 80 ./main 25 15 > tmp.log

