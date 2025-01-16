#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=10:00:00                # Adjust the time as needed (e.g., 2 hours)
#SBATCH --partition=amilan            # Set the partition to aa100
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
python3 Comparison.py --env cop-car --specify_dimension yes --rows 9 --columns 9 --runs 10 --threshold 0.9 --max_episodes 1000 --episode_len 500 --discount_factor 0.01 --save_file copcar-server-tabular-rs --buffer_size 50000 --save_data copcar-server-tabular-data-rs --deep_rl no --update_frequency 5 --reward_shaping yes

