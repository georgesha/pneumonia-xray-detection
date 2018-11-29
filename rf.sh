#!/bin/bash
#SBATCH --job-name=rf
#SBATCH --partition=cpu_medium
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=Zijing.Sha@nyulangone.org
#SBATCH --ntasks=1
#SBATCH --mem=40gb
#SBATCH --time=3-00:00:00
#SBATCH --output=rf.log

module load python/gcc/3.6.5
source ./env/bin/activate
python rf.py
