#!/usr/bin/env bash
set -euo pipefail

# Phi-4-mini caption pipeline:
# Visual features [B,T,768] -> UniVL visual encoder -> Q-Former [B,32,768]
# -> phi_proj 768->3072 -> visual prefix [B,32,3072] -> Phi decoder -> caption.

TRAIN_CSV="data/msrvtt/MSRVTT_train.9k.csv"
VAL_CSV="data/msrvtt/MSRVTT_JSFUSION_test.csv"
DATA_PATH="data/msrvtt/MSRVTT_data.json"
FEATURES_PATH="/workspace/project/datasets/phihungcrr1701/msrvtt-clip-vitl14-features/versions/1/msrvtt_clip_vitl14_features.pickle"
OUTPUT_DIR="ckpts/ckpt_msrvtt_caption_phi4"

# Optional: use this only if the checkpoint contains useful visual/Q-Former weights.
# Legacy decoder keys are ignored by the current Phi pipeline; phi_model comes from HF
# and phi_proj is random unless the checkpoint was trained with Phi.
INIT_MODEL=""
# INIT_MODEL="ckpts/ckpt_msrvtt_caption_phi4/pytorch_model.bin.best.0"

INIT_ARGS=()
if [[ -n "${INIT_MODEL}" ]]; then
  INIT_ARGS+=(--init_model "${INIT_MODEL}")
fi

torchrun --nproc_per_node=1 --standalone main_task_caption.py \
  --do_train --stage_two --task_type caption --datatype msrvtt \
  --num_thread_reader=4 \
  --epochs=5 \
  --batch_size=128 \
  --batch_size_val=128 \
  --gradient_accumulation_steps=2 \
  --n_display=50 \
  --train_csv "${TRAIN_CSV}" \
  --val_csv "${VAL_CSV}" \
  --data_path "${DATA_PATH}" \
  --features_path "${FEATURES_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --bert_model bert-base-uncased \
  --do_lower_case \
  --lr 3e-5 \
  --lr_qformer 5e-6 \
  --lr_lora 2e-6 \
  --max_words 48 \
  --max_frames 48 \
  --video_dim 768 \
  --visual_num_hidden_layers 6 \
  --freeze_vit \
  --num_query_token 32 \
  --qformer_vision_width 1408 \
  --qformer_checkpoint Salesforce/blip2-opt-6.7b-coco \
  --qformer_diversity_weight 0.0 \
  --llm_model microsoft/Phi-4-mini-instruct \
  --max_txt_len 32 \
  --eval_beam_size 5 \
  --lora \
  --lora_r 32 \
  --lora_alpha 64 \
  --lora_dropout 0.05 \
  --lora_target_modules qkv_proj,o_proj \
  "${INIT_ARGS[@]}"

# After XE warm-up converges, run SCST fine-tuning by reusing the best Phi
# checkpoint above as INIT_MODEL and adding:
#   --scst --scst_alpha 0.5 --scst_num_samples 5
