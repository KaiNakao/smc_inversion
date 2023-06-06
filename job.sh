#!/bin/sh
#PBS -q calc
#PBS -l ncpus=80
#PBS -N smc_slip
cd /home/nakao/smc_inversion
make main
./main > output.log
