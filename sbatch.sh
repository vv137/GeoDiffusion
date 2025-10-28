#!/bin/bash
#SBATCH --job-name=geo_diff_transformer
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=16:00:00
#SBATCH --gpus-per-node=1
#SBATCH --output=logs/transformer_exp_%j.out
#SBATCH --error=logs/transformer_exp_%j.err
#SBATCH --partition=multi

mkdir -p logs

echo "--- GeoDiffusion Transformer Experiments Starting ---"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on host: $(hostname)"
echo "Running on GPU: $(nvidia-smi -L)"

# --- Environment Setup ---
# module load cuda/... python/...
source ".venv/bin/activate" # Activate your virtual environment
# pip install torch numpy pandas matplotlib

# --- Shared Parameters ---
DATASET="zigzag_variable_dataset.npz" # <-- Updated dataset
EPOCHS=300 # Increased for transformer
BATCH_SIZE=64
LR=1e-4
SIGMA_DATA=1.0
LDDT_WEIGHT=1.0 # Weight for smooth LDDT when enabled
PLOT_EVERY=25
SAVE_EVERY=50

# --- Experiment 1: EDM (MSE Only) ---
LOG_DIR_EXP1="logs_transformer"
LOG_FILE_EXP1="edm_no_lddt_log.csv"
CKPT_DIR_EXP1="transformer_edm_no_lddt"
mkdir -p $CKPT_DIR_EXP1
echo "--- 1/5: Training Transformer EDM (MSE only) ---"
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
    --plot_every $PLOT_EVERY \
    --save_every $SAVE_EVERY

# --- Experiment 2: EDM + SmoothLDDT ---
LOG_DIR_EXP2="logs_transformer"
LOG_FILE_EXP2="edm_with_lddt_log.csv"
CKPT_DIR_EXP2="transformer_edm_with_lddt"
mkdir -p $CKPT_DIR_EXP2
echo "--- 2/5: Training Transformer EDM + SmoothLDDT ---"
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
    --plot_every $PLOT_EVERY \
    --save_every $SAVE_EVERY

# --- Experiment 3: AF3 Weighting (MSE Only) ---
LOG_DIR_EXP3="logs_transformer"
LOG_FILE_EXP3="af3_no_lddt_log.csv"
CKPT_DIR_EXP3="transformer_af3_no_lddt"
mkdir -p $CKPT_DIR_EXP3
echo "--- 3/5: Training Transformer AF3 Weighting (MSE only) ---"
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
    --plot_every $PLOT_EVERY \
    --save_every $SAVE_EVERY

# --- Experiment 4: AF3 Weighting + SmoothLDDT ---
LOG_DIR_EXP4="logs_transformer"
LOG_FILE_EXP4="af3_with_lddt_log.csv"
CKPT_DIR_EXP4="transformer_af3_with_lddt"
mkdir -p $CKPT_DIR_EXP4
echo "--- 4/5: Training Transformer AF3 Weighting + SmoothLDDT ---"
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
    --plot_every $PLOT_EVERY \
    --save_every $SAVE_EVERY

# --- Step 5: Plot Combined Results ---
echo "--- 5/5: Plotting combined results ---"
python plot_results.py \
    "$LOG_DIR_EXP1/$LOG_FILE_EXP1" \
    "$LOG_DIR_EXP2/$LOG_FILE_EXP2" \
    "$LOG_DIR_EXP3/$LOG_FILE_EXP3" \
    "$LOG_DIR_EXP4/$LOG_FILE_EXP4" \
    --output "transformer_experiment_comparison.png"

echo "--- Transformer Experiments Finished. Results plot saved to transformer_experiment_comparison.png ---"
