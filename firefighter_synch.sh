#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=15:00:00                # Adjust the time as needed (e.g., 2 hours)
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
python3 Comparison.py --env firefighter-synch --specify_dimension yes --rows 7 --columns 7 --runs 20 --threshold 0.85 --max_episodes 20000 --episode_len 500 --discount_factor 0.035  --save_file firefightersynch-deeprl-server --buffer_size 75000 --save_data firefightersynch-deeprl-server-data

