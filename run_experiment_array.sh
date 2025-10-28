#!/bin/bash
#SBATCH --job-name=geo_diff_array
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00 # 작업 1개당 예상 시간
#SBATCH --gpus-per-node=1
#SBATCH --array=0-3 # 4개의 작업을 배열로 생성 (0, 1, 2, 3)
#SBATCH --output=logs/transformer_exp_%A_%a.out # %A = Job ID, %a = Task ID
#SBATCH --error=logs/transformer_exp_%A_%a.err
#SBATCH --partition=multi

mkdir -p logs
mkdir -p logs_transformer

echo "--- GeoDiffusion Transformer Task Starting ---"
echo "Job ID: $SLURM_JOB_ID, Task ID: $SLURM_ARRAY_TASK_ID"
echo "Running on host: $(hostname)"
echo "Running on GPU: $(nvidia-smi -L)"

# --- Environment Setup ---
# module load cuda/... python/...
# .venv/bin/activate 로 가상 환경을 활성화하세요.
# source ".venv/bin/activate" 

# --- Shared Parameters ---
DATASET="zigzag_variable_dataset.npz"
EPOCHS=2000
BATCH_SIZE=64
LR=1e-4
SIGMA_DATA=1.0
PLOT_EVERY=25
SAVE_EVERY=50
SAMPLE_STEPS=40
LOG_DIR="logs_transformer"

# --- Task-Specific Parameters ---
# $SLURM_ARRAY_TASK_ID 값에 따라 다른 실험을 설정합니다.
case $SLURM_ARRAY_TASK_ID in
  0)
    # Experiment 1: EDM (MSE Only)
    LOSS_TYPE="edm"
    LDDT_WEIGHT=0.0
    CKPT_DIR="transformer_edm_no_lddt"
    LOG_FILE="edm_no_lddt_log.csv"
    ;;
  1)
    # Experiment 2: EDM + SmoothLDDT
    LOSS_TYPE="edm"
    LDDT_WEIGHT=1.0
    CKPT_DIR="transformer_edm_with_lddt"
    LOG_FILE="edm_with_lddt_log.csv"
    ;;
  2)
    # Experiment 3: AF3 Weighting (MSE Only)
    LOSS_TYPE="af3"
    LDDT_WEIGHT=0.0
    CKPT_DIR="transformer_af3_no_lddt"
    LOG_FILE="af3_no_lddt_log.csv"
    ;;
  3)
    # Experiment 4: AF3 Weighting + SmoothLDDT
    LOSS_TYPE="af3"
    LDDT_WEIGHT=1.0
    CKPT_DIR="transformer_af3_with_lddt"
    LOG_FILE="af3_with_lddt_log.csv"
    ;;
esac

echo "--- Running Task $SLURM_ARRAY_TASK_ID: $CKPT_DIR ---"
mkdir -p $CKPT_DIR

# --- Run Training ---
# 모든 작업이 train.py를 실행하되, 위에서 설정된 변수들을 인자로 받습니다.
python train.py \
    --loss_type $LOSS_TYPE \
    --smooth_lddt_weight $LDDT_WEIGHT \
    --data_file $DATASET \
    --log_dir $LOG_DIR \
    --log_file_name $LOG_FILE \
    --checkpoint_dir $CKPT_DIR \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --sigma_data $SIGMA_DATA \
    --plot_every $PLOT_EVERY \
    --save_every $SAVE_EVERY \
    --sample_steps $SAMPLE_STEPS

echo "--- Finished Task $SLURM_ARRAY_TASK_ID: $CKPT_DIR ---"
