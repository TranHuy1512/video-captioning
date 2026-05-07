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
  --scst --scst_alpha 0.5 --max_txt_len 20
