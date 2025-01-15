#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=10:00:00                # Adjust the time as needed (e.g., 2 hours)
#SBATCH --partition=amilan              # Set the partition to aa100
#SBATCH --ntasks=1                     # Number of tasks (usually 1 for Python scripts)
#SBATCH --cpus-per-task=1           # Number of CPU cores for the task
#SBATCH --mem=20G                      # Request 20GB of CPU memory
#SBATCH --job-name=python-comparison   # Job name
#SBATCH --output=python-comparison.%j.out  # Output file (%j will be replaced by job ID)

# Load Anaconda module
module purge
module load anaconda

# Activate the Conda environment
conda activate amfa-custom-env

# Run the Python script with the given arguments
python3 Comparison.py --env treasure-map --specify_dimension yes --rows 7 --columns 7 --runs 5 --reward_shaping no --threshold 0.9 --max_episodes 20000 --episode_len 1000 --update_frequency 10 --discount_factor 0.01  --save_file treasurehunt-tabular-server --buffer_size 50000 --save_data treasurehunt-server-tabular-data --deep_rl no

