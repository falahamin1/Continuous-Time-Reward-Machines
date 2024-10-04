#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=25:00:00                # Adjust the time as needed (e.g., 2 hours)
#SBATCH --partition=aa100              # Set the partition to aa100
#SBATCH --ntasks=1                     # Number of tasks (usually 1 for Python scripts)
#SBATCH --cpus-per-task=1           # Number of CPU cores for the task
#SBATCH --gres=gpu:1                 # Request 1 GPU
#SBATCH --mem=20G                      # Request 20GB of CPU memory
#SBATCH --job-name=python-comparison   # Job name
#SBATCH --output=python-comparison.%j.out  # Output file (%j will be replaced by job ID)

# Load Anaconda module
module purge
module load anaconda

# Activate the Conda environment
conda activate amfa-custom-env

# Run the Python script with the given arguments
python3 Comparison.py --env treasure-map --specify_dimension yes --rows 7 --columns 7 --runs 30 --threshold 0.9 --max_episodes 100000 --save_file treasurehunt-deeprl-server --buffer_size 50000 --save_data treasurehunt-deeprl-server-data

