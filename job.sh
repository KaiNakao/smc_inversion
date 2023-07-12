#!/bin/sh
#PBS -q calc-lm
#PBS -l select=1:ncpus=160:mpiprocs=160
#PBS -N smc_inv

cd /home/nakao/smc_inversion
make main
ulimit -s unlimited

mpiexec -n 160 bash -c "ulimit -s unlimited"
mpiexec -n 160 ./main 15 15 > cvtest.log

