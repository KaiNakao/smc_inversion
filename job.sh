#!/bin/sh
#PBS -q calc-lm
#PBS -l ncpus=192
#PBS -N smc_inv_omp

cd /home/nakao/smc_inversion
make main
ulimit -s unlimited
./main 20 20 > omp100000.log

