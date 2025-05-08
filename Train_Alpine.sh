#!/bin/bash
#SBATCH --partition=ami100
#SBATCH --nodes=4
#SBATCH --ntasks=16
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:3                    # Request 3 GPUs
#SBATCH --job-name=Arabidopsis_Train    # Job name
#SBATCH --output=job_output_%j.log      # Redirect output to a file with job ID
#SBATCH --error=job_error_%j.log        # Redirect errors to a file with job ID
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ted.monyak@colostate.edu

module purge
module load anaconda
module load cuda/11.8
module load gnu_parallel/20210322

echo "Starting my job..."

# Start timing the process
start_time=$(date +%s)

fileLocation=$1
args=$2

# ########### PYTHON PROGRAM #############################
# Ensure output directory exists
mkdir -p "${PWD}/output"

# Correct output file name
output_names="${PWD}/output/output_${SLURM_JOB_ID}_$(basename ${fileLocation})"

conda activate gp
python DNACNN.py

# End timing the process
end_time=$(date +%s)
total_time=$(( (end_time - start_time) / 60 ))

# Print the time to complete the process
echo "Total time to complete the job: $total_time minutes"

# ########### TO EXECUTE RUN IN TERMINAL #########################
# run as: sbatch runner_pipeline.sh /dev/null 2>&1 & disown

exit 0

# ########### TO REMOVE SOME FILES #########################

# To remove files
# ls *.tif
# rm -r temp_*
# rm -r analysis_*
# rm -r slurm* out* temp_* masks_*

# ########### SLURM COMMANDS #########################
# scancel [jobid]
# squeue -u [username]
# squeue
