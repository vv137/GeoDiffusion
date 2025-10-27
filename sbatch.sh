#!/bin/bash
#SBATCH --job-name=geo_diffusion
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --gpus-per-node=1
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

mkdir -p logs

echo "--- GeoDiffusion Experiment Starting ---"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on host: $(hostname)"
echo "Running on GPU: $(nvidia-smi -L)"

source ".venv/bin/activate"
# pip install torch numpy pandas matplotlib

# -- Step 1: Generate Dataset ---
echo "--- 1/4: Generating dataset ---"
python generate_dataset.py

# --- Step 2: Train EDM-style Model ---
echo "--- 2/4: Training EDM-style model (Control) ---"
python train.py --loss_type edm --log_file edm_log.csv --checkpoint_dir edm_checkpoints --epochs 100 --batch_size 64

# --- Step 3: Train AF3-style Model ---
echo "--- 3/4: Training AF3-style model (Experimental) ---"
python train.py --loss_type af3 --log_file af3_log.csv --checkpoint_dir af3_checkpoints --epochs 100 --batch_size 64

# --- Step 4: Plot Results ---
echo "--- 4/4: Plotting results ---"
python plot_results.py edm_log.csv af3_log.csv

echo "--- Experiment Finished. Results saved to experiment_results.png ---"
