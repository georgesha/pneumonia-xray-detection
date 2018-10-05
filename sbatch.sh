#!/bin/bash 
#SBATCH --job-name=pneumonia # Job name 
#SBATCH --mail-type=END,FAIL # Mail events (NONE, BEGIN, END, FAIL, ALL) 
#SBATCH --mail-user=Moosun.kim@nyumc.org # Where to send mail 
#SBATCH --ntasks=10 # Run on a single CPU 
#SBATCH --mem=1-20gb # Job memory request 
#SBATCH --time=03:00:00 # Time limit hrs:min:sec 
#SBATCH --output=serial_test_%j.log # Standard output and error log pwd; hostname; date sleep 300 date 

