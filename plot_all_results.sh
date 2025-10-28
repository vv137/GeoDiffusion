#!/bin/bash
# 이 스크립트는 Slurm에 제출하지 말고,
# 모든 작업이 완료된 후 터미널에서 직접 실행하세요.
# (예: bash plot_all_results.sh)

echo "--- Plotting combined results for Transformer experiments ---"

LOG_DIR="logs_transformer"
OUTPUT_FILE="transformer_experiment_comparison.png"

# plot_results.py를 호출하여 4개의 로그 파일을 비교합니다.
python plot_results.py \
    "$LOG_DIR/edm_no_lddt_log.csv" \
    "$LOG_DIR/edm_with_lddt_log.csv" \
    "$LOG_DIR/af3_no_lddt_log.csv" \
    "$LOG_DIR/af3_with_lddt_log.csv" \
    --labels "EDM (MSE)" "EDM + SmoothLDDT" "AF3 (MSE Weight)" "AF3 + SmoothLDDT" \
    --output $OUTPUT_FILE

echo "Plot saved to $OUTPUT_FILE"
