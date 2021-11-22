#!/bin/bash
#SBATCH -J find-max_n               # job name
#SBATCH --mail-user=hagenfritz@utexas.edu
#SBATCH --mail-type=ALL
#SBATCH -o launcher.o%j             # output and error file name (%j expands to SLURM jobID)
#SBATCH -N 4                        # number of nodes requested
#SBATCH -n 32                       # total number of tasks to run in parallel
#SBATCH -p skx-dev                  # queue (partition) 
#SBATCH -t 2:00:00                 # run time (hh:mm:ss) 
#SBATCH -A Energy-Profile-Clust     # Allocation name to charge job against

module load launcher

export LAUNCHER_PLUGIN_DIR=$LAUNCHER_DIR/plugins
export LAUNCHER_WORKDIR=/scratch/01234/joe/tests
export LAUNCHER_JOB_FILE=run_max_n 

${LAUNCHER_DIR}/paramrun