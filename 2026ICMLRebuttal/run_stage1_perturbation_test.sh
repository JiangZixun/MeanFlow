#!/usr/bin/env bash
set -euo pipefail

cd /mnt/data1/MeanFlow

PYTHON_BIN="${PYTHON_BIN:-/home/jzx/anaconda3/envs/torchcfm/bin/python}"
CONFIG="${CONFIG:-configs/JiT-B_RFDPIC.yaml}"
RFDPIC_CONFIG="${RFDPIC_CONFIG:-configs/rfdpic_config.yaml}"
RFDPIC_CKPT="${RFDPIC_CKPT:-pretrained_models/pretrained_RFDPIC_Dual_Rotation_Dyn.pt}"
MEANFLOW_CKPT="${MEANFLOW_CKPT:-logs/JiT-B/checkpoints/step_1000000-loss_0.1542.ckpt}"
ROOT_LOG_DIR="${ROOT_LOG_DIR:-logs/2026ICMLRebuttal/stage1_perturbation_7x2}"
SUMMARY_CSV="${SUMMARY_CSV:-${ROOT_LOG_DIR}/all_results.csv}"
mkdir -p "${ROOT_LOG_DIR}"
rm -f "${SUMMARY_CSV}"
COMMON_ARGS=(
  --config "${CONFIG}"
  --ckpt_path "${MEANFLOW_CKPT}"
  --rfdpic_config "${RFDPIC_CONFIG}"
  --rfdpic_ckpt "${RFDPIC_CKPT}"
  --sample_steps "${SAMPLE_STEPS:-10}"
  --gpus "${GPUS:-1}"
  --batch_size "${BATCH_SIZE:-8}"
  --summary_csv "${SUMMARY_CSV}"
)

run_case() {
  local case_name="$1"
  local severity="$2"
  shift
  shift
  echo "============================================================"
  echo "Running case: ${case_name}"
  echo "============================================================"
  "${PYTHON_BIN}" /mnt/data1/MeanFlow/2026ICMLRebuttal/rebuttal_stage1_perturbation_test.py \
    "${COMMON_ARGS[@]}" \
    --log_dir "${ROOT_LOG_DIR}/${case_name}" \
    --case_name "${case_name}" \
    --severity "${severity}" \
    "$@"
}

# Mild settings
run_case "blur_mild" "mild" \
  --perturbations blur \
  --blur_kernel 5 \
  --blur_sigma 1.0

run_case "noise_mild" "mild" \
  --perturbations noise \
  --noise_mode additive \
  --noise_std 0.02

run_case "deform_mild" "mild" \
  --perturbations deform \
  --affine_rotate 3.0 \
  --affine_translate 0.02 \
  --affine_scale_min 0.98 \
  --affine_scale_max 1.02 \
  --affine_shear 2.0

run_case "bias_mild" "mild" \
  --perturbations bias \
  --bias_value 0.03

run_case "scale_mild" "mild" \
  --perturbations scale \
  --scale_factor 0.95

run_case "dropout_mild" "mild" \
  --perturbations dropout \
  --dropout_prob 0.10

run_case "all6_mild" "mild" \
  --perturbations blur,noise,deform,bias,scale,dropout \
  --blur_kernel 5 \
  --blur_sigma 1.0 \
  --noise_mode additive \
  --noise_std 0.02 \
  --affine_rotate 3.0 \
  --affine_translate 0.02 \
  --affine_scale_min 0.98 \
  --affine_scale_max 1.02 \
  --affine_shear 2.0 \
  --bias_value 0.03 \
  --scale_factor 0.95 \
  --dropout_prob 0.10

# Strong settings
run_case "blur_strong" "strong" \
  --perturbations blur \
  --blur_kernel 9 \
  --blur_sigma 2.0

run_case "noise_strong" "strong" \
  --perturbations noise \
  --noise_mode additive \
  --noise_std 0.05

run_case "deform_strong" "strong" \
  --perturbations deform \
  --affine_rotate 8.0 \
  --affine_translate 0.05 \
  --affine_scale_min 0.94 \
  --affine_scale_max 1.06 \
  --affine_shear 5.0

run_case "bias_strong" "strong" \
  --perturbations bias \
  --bias_value 0.08

run_case "scale_strong" "strong" \
  --perturbations scale \
  --scale_factor 0.85

run_case "dropout_strong" "strong" \
  --perturbations dropout \
  --dropout_prob 0.25

run_case "all6_strong" "strong" \
  --perturbations blur,noise,deform,bias,scale,dropout \
  --blur_kernel 9 \
  --blur_sigma 2.0 \
  --noise_mode additive \
  --noise_std 0.05 \
  --affine_rotate 8.0 \
  --affine_translate 0.05 \
  --affine_scale_min 0.94 \
  --affine_scale_max 1.06 \
  --affine_shear 5.0 \
  --bias_value 0.08 \
  --scale_factor 0.85 \
  --dropout_prob 0.25
