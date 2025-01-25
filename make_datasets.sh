#!/bin/bash
#SBATCH --job-name=testing_make)dataset        # Job name
#SBATCH --output=%j.log
#SBATCH --nodes=1                     # Number of nodes (use 1 node)
#SBATCH --ntasks=1                    # One task
#SBATCH --cpus-per-task=1            # 1 CPU cores for parallelism
#SBATCH --mem=5GB                    # Total memory for the job (adjust based on need)
#SBATCH --time=00:10:00               # Time limit for the job (e.g., 2 hours)

# remove all previously loaded modules
module purge

# load python 3.8.16
module load Python/3.8.16-GCCcore-11.2.0   
 
# activate virtual environment
source $HOME/venvs/updog/bin/activate

# Run training
python3 make_all_datasets.py