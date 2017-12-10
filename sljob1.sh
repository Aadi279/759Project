#!/bin/bash
#SBATCH --partition=slurm_shortgpu
#SBATCH --time=0-00:05:00 # run time in days-hh:mm:ss
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --error=sbatch.err
#SBATCH --output=sbatch.out
#SBATCH --gres=gpu:1
cd /srv/home/gelkind/HPC_project # go to job submission directory
./findIntersections

