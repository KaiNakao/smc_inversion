#!/bin/sh
#PBS -q calc-lm
#PBS -l ncpus=192
#PBS -N smc_inv

cd /home/nakao/smc_inversion
make main
./main 15 15 > 15_15_all_slip.log

