#!/bin/sh
#PBS -q calc-lm
#PBS -l ncpus=192
#PBS -N smc_inv

cd /home/nakao/smc_inversion_omp
make main
./main 20 20 > omp2.log

