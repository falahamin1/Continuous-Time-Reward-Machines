#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=13:00:00                # Adjust the time as needed (e.g., 2 hours)
#SBATCH --partition=amilan              # Set the partition to amilan
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
python3 Comparison.py --env firefighter-car --specify_dimension yes --reward_shaping no --rows 7 --columns 7 --runs 5 --threshold 0.8 --max_episodes 25000 --update_frequency 10 --discount_factor 0.105 --episode_len 1000 --learning_rate 0.2 --save_file firefighter-tabular-server --buffer_size 75000  --save_data firefighter-tabular-server-data --deep_rl no

