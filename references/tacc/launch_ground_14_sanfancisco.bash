#!/bin/bash
#SBATCH -J launcher-sanfrancisco
#SBATCH --mail-user=demir@utexas.edu
#SBATCH --mail-type=ALL
#SBATCH -o launcher.o%j
#SBATCH -p skx-dev
#SBATCH -N 4
#SBATCH -n 192
#SBATCH -t 2:00:00
#SBATCH -A DemandAnalysis

module load launcher

export LAUNCHER_PLUGIN_DIR=$LAUNCHER_DIR/plugins
export LAUNCHER_RMI=SLURM
export LAUNCHER_JOB_FILE=launcher_job_ground_sanfrancisco

$LAUNCHER_DIR/paramrun