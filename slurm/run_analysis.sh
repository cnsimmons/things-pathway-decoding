#!/bin/bash
#SBATCH --job-name=things-pathway-decode
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --time=6:00:00
#SBATCH --output=slurm/job_%j.out
#SBATCH --error=slurm/job_%j.err

# Parse command line arguments
SUBJECT_ID=${1:-"01"}
SESSION_ID=${2:-"01"}

echo "Starting THINGs pathway decoding analysis"
echo "Subject: $SUBJECT_ID, Session: $SESSION_ID"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Started at: $(date)"

# Load any necessary modules (adjust for your cluster)
# module load python/3.9
# module load cuda/11.8  # if needed

# Navigate to project directory
cd $SLURM_SUBMIT_DIR
echo "Working directory: $(pwd)"

# Create conda environment or install requirements
echo "Installing requirements..."
pip install -r requirements.txt --quiet

# Create results directory
mkdir -p results

echo "=== Step 1: Download data ==="
python scripts/download_data.py $SUBJECT_ID $SESSION_ID

echo "=== Step 2: Preprocess data ==="
python scripts/preprocess.py $SUBJECT_ID $SESSION_ID

echo "=== Step 3: Run decoding analysis ==="
python scripts/decode_pathways.py $SUBJECT_ID $SESSION_ID

echo "=== Analysis complete ==="
echo "Finished at: $(date)"

# Optional: Clean up downloaded data to save space
# echo "Cleaning up data directory..."
# rm -rf data/

echo "Job completed successfully!"