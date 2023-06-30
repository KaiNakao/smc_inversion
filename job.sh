#!/bin/sh
#PBS -q calc-lm
#PBS -l ncpus=192
#PBS -N smc_inv_omp

cd /home/nakao/smc_inversion
make main
./main 20 20 > omp40000.log

