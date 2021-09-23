#!/bin/bash
# ALWAYS specify CPU and RAM resources needed, as well as walltime
#SBATCH --partition=long
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=24G
#SBATCH --gres=gpu:ampere:1
# email user with progress
#SBATCH --mail-user=keith.azzopardi.16@um.edu.mt
#SBATCH --mail-type=all
#SBATCH --job-name=dmtrain


# This script should always be run on radagast, and 
# schedules a job to run on the other nodes

# DO NOT FORGET TO GIT PULL BEFORE RUNNING THIS
srun ~/msc_dissertation/dm_training/train_uni_worker.sh