# torchrun --nproc_per_node=1 --standalone \
# main_task_caption.py \
# --do_train --stage_two --task_type caption --datatype msrvtt \
# --num_thread_reader=4 \
# --epochs=15 --batch_size=256 \
# --n_display=50 \
# --train_csv data/msrvtt/MSRVTT_train.9k.csv \
# --val_csv data/msrvtt/MSRVTT_JSFUSION_test.csv \
# --data_path data/msrvtt/MSRVTT_data.json \
# --features_path /workspace/project/datasets/phihungcrr1701/msrvtt-clip-vitl14-features/versions/1/msrvtt_clip_vitl14_features.pickle \
# --output_dir ckpts/ckpt_msrvtt_caption --bert_model bert-base-uncased \
# --do_lower_case --lr 5e-6 --max_words 48 --max_frames 48 \
# --batch_size_val 256 --visual_num_hidden_layers 6 \
# --freeze_vit \
# --init_model /workspace/project/datasets/trnphmngcminh/checkpoint-qformer-t5-model/versions/2/pytorch_model.bin.best.0 \
# --gradient_accumulation_steps=2 \
# --video_dim 768 \
# --qformer_vision_width 1408 \
# --qformer_checkpoint Salesforce/blip2-opt-6.7b-coco \
# --lora --lr_qformer 5e-6 --lr_lora 2e-6 --lora_r=32 --lora_alpha=64 \
# --scst --max_txt_len 20 \
# --qformer_diversity_weight 0.0


# # torchrun --nproc_per_node=1 --standalone main_task_caption.py --do_train --stage_two --task_type caption --datatype msrvtt --num_thread_reader=4 --epochs=1 --batch_size=128 --n_display=50 --train_csv data/msrvtt/MSRVTT_train.9k.csv --val_csv data/msrvtt/MSRVTT_JSFUSION_test.csv --data_path data/msrvtt/MSRVTT_data.json --features_path /workspace/project/datasets/phihungcrr1701/msrvtt-clip-vitl14-features/versions/1/msrvtt_clip_vitl14_features.pickle --output_dir ckpts/ckpt_msrvtt_caption --bert_model bert-base-uncased --do_lower_case --lr 3e-5 --max_words 48 --max_frames 48 --batch_size_val 128 --visual_num_hidden_layers 6 --freeze_vit --init_model /workspace/project/video-captioning/ckpts/ckpt_msrvtt_caption/pytorch_model.bin.best_v1.0 --gradient_accumulation_steps=2 --video_dim 768 --qformer_vision_width 1408 --qformer_checkpoint Salesforce/blip2-opt-6.7b-coco --lora --lr_qformer 2e-5 --lr_lora 1e-5 --lora_r=32 --lora_alpha=64 --scst --max_txt_len 20 
# ver_7 result: CIDEr 0.5917→0.5569→0.5390 (declining)
# Root causes: scst_alpha=1.0 (no XE regularization) → val loss explodes 2.5→7.3
# Fix: mix XE + SCST (scst_alpha=0.5), lower LR for stable RL updates
#
# evidents/ run (scst_alpha=1.0, lr_qformer/lr_lora ~1e-7): CIDEr plateaued at
# 0.6118 after epoch 2/6 (barely moved, LR too conservative to learn anything
# new) while self-BLEU-4 between generated captions was already 0.732-0.755
# (XE-only vs post-SCST) — i.e. ~1/3 of test captions are exact duplicates of
# another video's caption, mostly inherited from the XE checkpoint, not
# created by SCST. See evidents/ + utils/diversity_utils.py for the analysis.
# Two fixes applied for this run, both opt-in/backward compatible by default:
#   1. trainers/trainer.py now builds multi-reference gt_refs (all ~20 GT
#      captions/video, same set eval_epoch uses for CIDEr) instead of scoring
#      each beam against a single caption — automatic, no flag needed.
#   2. --scst_diversity_weight penalizes a candidate that n-gram-overlaps
#      heavily with another video's generated caption in the same batch, to
#      push back on mode collapse onto generic templates (CIDEr alone can't
#      see this, since it scores each sample independently).
torchrun --nproc_per_node=1 --standalone main_task_caption.py \
  --do_train --stage_two --task_type caption --datatype msrvtt \
  --num_thread_reader=4 --epochs=5 --batch_size=128 --n_display=50 \
  --train_csv data/msrvtt/MSRVTT_train.9k.csv \
  --val_csv data/msrvtt/MSRVTT_JSFUSION_test.csv \
  --data_path data/msrvtt/MSRVTT_data.json \
  --features_path /workspace/project/datasets/phihungcrr1701/msrvtt-clip-vitl14-features/versions/1/msrvtt_clip_vitl14_features.pickle \
  --output_dir ckpts/ckpt_msrvtt_caption \
  --bert_model bert-base-uncased --do_lower_case \
  --lr 3e-5 --max_words 48 --max_frames 48 --batch_size_val 128 \
  --visual_num_hidden_layers 6 --freeze_vit \
  --init_model /workspace/project/video-captioning/ckpts/ckpt_msrvtt_caption_ver_1/pytorch_model.bin.best.0 \
  --gradient_accumulation_steps=2 \
  --video_dim 768 --qformer_vision_width 1408 \
  --qformer_checkpoint Salesforce/blip2-opt-6.7b-coco \
  --lora --lora_r=32 --lora_alpha=64 \
  --lr_qformer 5e-7 --lr_lora 1e-6 \
  --scst --scst_alpha 0.5 --scst_diversity_weight 0.3 --max_txt_len 20
