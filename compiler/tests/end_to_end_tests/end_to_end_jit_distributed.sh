#!/bin/bash
#SBATCH --job-name=end_to_end_jit_distributed
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=antoniu.pop@zama.ai
#SBATCH --nodes=4
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --output=end_to_end_jit_distributed_%j.log

echo "Date              = $(date)"
echo "Hostname          = $(hostname -s)"
echo "Working Directory = $(pwd)"
echo ""
echo "Number of Nodes Allocated      = $SLURM_JOB_NUM_NODES"
echo "Number of Tasks Allocated      = $SLURM_NTASKS"
echo "Number of Cores/Task Allocated = $SLURM_CPUS_PER_TASK"

export OMP_NUM_THREADS=8
export DFR_NUM_THREADS=2

srun ./build/bin/end_to_end_jit_distributed

date
