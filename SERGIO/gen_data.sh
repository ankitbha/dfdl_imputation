#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=18:00:00
#SBATCH --mem=32GB
#SBATCH --job-name=gen_data
#SBATCH --output=gendata_%j.out

conda activate /scratch/ab9738/cctv_pollution/env/;
export PATH=/scratch/ab9738/cctv_pollution/env/bin:$PATH;
cd /scratch/ab9738/dfdl_imputation/SERGIO/
python generate_data.py
