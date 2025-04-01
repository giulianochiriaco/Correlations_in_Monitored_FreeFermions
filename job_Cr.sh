#!/bin/bash
#
#SBATCH --job-name=Cr
#SBATCH --account=IscrC_EMEND-Q
#SBATCH --partition=g100_usr_prod
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-core=1
#
#SBATCH --mem-per-cpu=4000mb
#
#SBATCH --array=0-20
#SBATCH --time=23:59:00
#SBATCH --output=./Outputs/Cr.o%A-%a
#SBATCH --error=./Errors/Cr.e%A-%a
#

pR=${SLURM_ARRAY_TASK_ID}
multiplier=0.05

t1=1
t2=1.0
t12=1.570796
L=32
Nmax=100
NRseries=100

python3 FF_TransientCr.py $L $t1 $t2 $t12 $pR $Nmax $NRseries $multiplier
