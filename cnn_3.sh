#!/bin/bash
#SBATCH --job-name=cnn1
#SBATCH --partition=gpu8_medium
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=Zijing.Sha@nyulangone.org
#SBATCH --ntasks=1
#SBATCH --mem=40gb
#SBATCH --time=3-00:00:00
#SBATCH --output=cnn3.log

module load python/gcc/3.6.5
python cnn_keras_3.py
