#!/bin/bash
#SBATCH --nodes=1			# Request 1 node
#SBATCH --ntasks=4 			# Request 4 core
#SBATCH --time=1:00:00			# Request 1 hour
#SBATCH --job-name=Arabidopsis_Data	# Job name
#SBATCH --partition=amilan		# Parition
#SBATCH --output=job_output_%j.log	# Redirect output to a file with job ID
#SBATCH --error=job_error_%j.log	# Redirect errors to a file with job ID
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ted.monyak@colostate.edu

module purge
module load anaconda
# module load gnu_parallel/20210322
# module load cuda/11.8

echo "Starting my job..."

# Start timing the process
start_time=$(date +%s)

#fileLocation=$1
#args=$2

# ########### PYTHON PROGRAM #############################
# Ensure output directory exists
mkdir -p "${PWD}/cluster_output"

# Correct output file name
output_names="${PWD}/cluster_output/output_${SLURM_JOB_ID}" #_$(basename ${fileLocation})"

# Activate the environment and run the script
conda activate gp
python Download_Data.py
python Parse_Data.py

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
