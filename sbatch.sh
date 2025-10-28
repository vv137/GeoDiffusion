#!/bin/bash
#SBATCH --job-name=geo_diff_exp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # One training script per node
#SBATCH --cpus-per-task=4
#SBATCH --time=16:00:00              # Increased time for 4 runs
#SBATCH --gpus-per-node=1
#SBATCH --output=logs/exp_%j.out
#SBATCH --error=logs/exp_%j.err
#SBATCH --partition=multi            # Specify partition if needed

mkdir -p logs

echo "--- GeoDiffusion Experiments Starting ---"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on host: $(hostname)"
echo "Running on GPU: $(nvidia-smi -L)"

# --- Environment Setup ---
# Load necessary modules if required (e.g., cuda, python)
# module load cuda/ ... python/ ...
source ".venv/bin/activate" # Activate your virtual environment
# Ensure dependencies are installed (do this once before submitting)
# pip install torch numpy pandas matplotlib

# --- Shared Parameters ---
DATASET="zigzag_dataset.npz"
EPOCHS=200
BATCH_SIZE=128
LR=2e-4
SIGMA_DATA=1.0
LDDT_WEIGHT=1.0 # Weight for smooth LDDT when enabled
PLOT_EVERY=20 # Plotting frequency

# --- Experiment 1: EDM (MSE Only) ---
LOG_DIR_EXP1="logs"
LOG_FILE_EXP1="edm_no_lddt_log.csv"
CKPT_DIR_EXP1="edm_no_lddt_checkpoints"
echo "--- 1/5: Training EDM (MSE only) ---"
python train.py \
    --loss_type edm \
    --smooth_lddt_weight 0.0 \
    --data_file $DATASET \
    --log_dir $LOG_DIR_EXP1 \
    --log_file_name $LOG_FILE_EXP1 \
    --checkpoint_dir $CKPT_DIR_EXP1 \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --sigma_data $SIGMA_DATA \
    --plot_every $PLOT_EVERY

# --- Experiment 2: EDM + SmoothLDDT ---
LOG_DIR_EXP2="logs"
LOG_FILE_EXP2="edm_with_lddt_log.csv"
CKPT_DIR_EXP2="edm_with_lddt_checkpoints"
echo "--- 2/5: Training EDM + SmoothLDDT ---"
python train.py \
    --loss_type edm \
    --smooth_lddt_weight $LDDT_WEIGHT \
    --data_file $DATASET \
    --log_dir $LOG_DIR_EXP2 \
    --log_file_name $LOG_FILE_EXP2 \
    --checkpoint_dir $CKPT_DIR_EXP2 \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --sigma_data $SIGMA_DATA \
    --plot_every $PLOT_EVERY

# --- Experiment 3: AF3 Weighting (MSE Only) ---
LOG_DIR_EXP3="logs"
LOG_FILE_EXP3="af3_no_lddt_log.csv"
CKPT_DIR_EXP3="af3_no_lddt_checkpoints"
echo "--- 3/5: Training AF3 Weighting (MSE only) ---"
python train.py \
    --loss_type af3 \
    --smooth_lddt_weight 0.0 \
    --data_file $DATASET \
    --log_dir $LOG_DIR_EXP3 \
    --log_file_name $LOG_FILE_EXP3 \
    --checkpoint_dir $CKPT_DIR_EXP3 \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --sigma_data $SIGMA_DATA \
    --plot_every $PLOT_EVERY

# --- Experiment 4: AF3 Weighting + SmoothLDDT ---
LOG_DIR_EXP4="logs"
LOG_FILE_EXP4="af3_with_lddt_log.csv"
CKPT_DIR_EXP4="af3_with_lddt_checkpoints"
echo "--- 4/5: Training AF3 Weighting + SmoothLDDT ---"
python train.py \
    --loss_type af3 \
    --smooth_lddt_weight $LDDT_WEIGHT \
    --data_file $DATASET \
    --log_dir $LOG_DIR_EXP4 \
    --log_file_name $LOG_FILE_EXP4 \
    --checkpoint_dir $CKPT_DIR_EXP4 \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --sigma_data $SIGMA_DATA \
    --plot_every $PLOT_EVERY

# --- Step 5: Plot Combined Results ---
echo "--- 5/5: Plotting combined results ---"
python plot_results.py \
    "$LOG_DIR_EXP1/$LOG_FILE_EXP1" \
    "$LOG_DIR_EXP2/$LOG_FILE_EXP2" \
    "$LOG_DIR_EXP3/$LOG_FILE_EXP3" \
    "$LOG_DIR_EXP4/$LOG_FILE_EXP4" \
    --output "experiment_comparison.png"

echo "--- Experiments Finished. Results plot saved to experiment_comparison.png ---"
