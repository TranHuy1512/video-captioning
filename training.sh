torchrun --nproc_per_node=1 --standalone main_task_caption.py \
	--do_train --stage_two --task_type caption --datatype msrvtt \
	--num_thread_reader=4 --epochs=6 --batch_size=128 --n_display=50 \
	--train_csv data/msrvtt/MSRVTT_train.9k.csv \
	--val_csv data/msrvtt/MSRVTT_JSFUSION_test.csv \
	--data_path data/msrvtt/MSRVTT_data.json \
	--features_path /workspace/project/datasets/elnino1512 \
	--output_dir ckpts/ckpt_msrvtt_caption \
	--bert_model bert-base-uncased --do_lower_case \
	--lr 5e-6 --max_words 48 --max_frames 48 --batch_size_val 256 \
	--visual_num_hidden_layers 6 --gradient_accumulation_steps=2 \
	--video_dim 3584 --qformer_vision_width 1408 \
	--qformer_checkpoint Salesforce/blip2-opt-6.7b-coco \
	--lora --lr_qformer 5e-6 --lr_lora 2e-6 --lora_r=32 --lora_alpha=64 \
	--max_txt_len 32 \
	--direct_qformer_input --qformer_adapter_type conv1d \
	# --scst --scst_alpha 0.7 --scst_num_samples 5 --eval_beam_size 5 \
	--warmup_proportion 0.02 \
	--init_model /workspace/project/video-captioning/ckpts/ckpt_msrvtt_caption_1/pytorch_model.bin.best.0
