#!/bin/bash
#SBATCH --mail-user=lavery@uwaterloo.ca
#SBATCH --mail-type=ALL
#SBATCH --job-name=options84s16
#SBATCH --output=options84s16-%j.out
#SBATCH --time=23:59:00
#SBATCH --ntasks=16
#SBATCH --nodes=4
##SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=3G
##SBATCH --mem=96G
export OPTS=options84
export KSFDDEBUG=ALL
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export KSFD=$HOME/KSFD
source $KSFD/bin/activate
cd $HOME/KSFD
mkdir -p $KSFD/tmp
export TMPDIR=$KSFD/tmp
cd $HOME/KSFD
uname -a
date
gtime -v srun --label --slurmd-debug=info python ksfdsolver2.py @$OPTS
date
