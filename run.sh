#!/usr/bin/env bash
set -euo pipefail

# Required for gated Llama weights.
# Usage:
#   export HF_TOKEN=hf_xxx
#   bash run.sh
: "${HF_TOKEN:?Please export HF_TOKEN with access to meta-llama/Llama-3.2-3B-Instruct}"

# Optional warm-start checkpoint. The loader keeps only compatible tensors and
# leaves mismatched/new modules, including Llama, at their freshly loaded weights.
INIT_MODEL="${INIT_MODEL:-}"
INIT_MODEL_ARGS=()
if [[ -n "${INIT_MODEL}" ]]; then
  INIT_MODEL_ARGS=(--init_model "${INIT_MODEL}")
fi

torchrun --nproc_per_node=1 --standalone main_task_caption.py \
  --do_train --stage_two --task_type caption --datatype msrvtt \
  --num_thread_reader=4 --epochs=5 --batch_size=128 --n_display=50 \
  --train_csv data/msrvtt/MSRVTT_train.9k.csv \
  --val_csv data/msrvtt/MSRVTT_JSFUSION_test.csv \
  --data_path data/msrvtt/MSRVTT_data.json \
  --features_path /workspace/project/datasets/phihungcrr1701/msrvtt-clip-vitl14-features/versions/1/msrvtt_clip_vitl14_features.pickle \
  --output_dir ckpts/ckpt_msrvtt_qformer_llama_caption \
  --bert_model bert-base-uncased --do_lower_case \
  --llama_model meta-llama/Llama-3.2-3B-Instruct \
  --hf_token "${HF_TOKEN}" \
  --lr 3e-5 --max_words 48 --max_frames 48 --batch_size_val 128 \
  --visual_num_hidden_layers 6 --freeze_vit \
  "${INIT_MODEL_ARGS[@]}" \
  --gradient_accumulation_steps=2 \
  --video_dim 768 --qformer_vision_width 1408 \
  --qformer_checkpoint Salesforce/blip2-opt-6.7b-coco \
  --lora --lora_r=32 --lora_alpha=64 --lora_target_modules q_proj,v_proj \
  --lr_qformer 5e-7 --lr_lora 1e-6 \
  --scst --scst_alpha 0.5 --max_txt_len 20
