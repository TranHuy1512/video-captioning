# Pipeline project hiện tại

Tài liệu này mô tả pipeline thực tế đang có trong repo ở thời điểm đọc code. Repo gốc là UniVL, nhưng nhánh caption hiện tại đã được chỉnh để dùng **VisualModel của UniVL + BLIP2 QFormer + Flan-T5**, có hỗ trợ LoRA và SCST.

## 1. Entrypoint chính

Pipeline chính cho video captioning chạy qua:

```text
run.sh
  -> torchrun main_task_caption.py
  -> data/dataloader_factory.py
  -> dataloaders/dataloader_msrvtt_caption.py hoặc dataloader_youcook_caption.py
  -> utils/model_utils.py
  -> modules/modeling.py::UniVL
  -> trainers/trainer.py
  -> inference/caption_generator.py
  -> metrics.py
```

Lệnh hiện tại trong `run.sh` train MSRVTT với:

- `--do_train --stage_two --task_type caption --datatype msrvtt`
- feature pickle CLIP ViT-L/14: `msrvtt_clip_vitl14_features.pickle`
- `--video_dim 768`
- `--qformer_vision_width 1408`
- QFormer checkpoint: `Salesforce/blip2-opt-6.7b-coco`
- Flan-T5 mặc định: `google/flan-t5-xl`
- `--lora`
- `--scst`
- `--freeze_vit`

## 2. Tổng quan luồng dữ liệu

```text
Video raw
  -> extract feature ngoài training runtime
  -> lưu thành pickle: {video_id hoặc feature_file: np.ndarray(T, D)}
  -> Caption DataLoader
       -> text BERT legacy tensors
       -> caption labels BERT legacy
       -> caption labels T5
       -> video tensor + video_mask
  -> UniVL.forward()
       -> NormalizeVideo
       -> VisualModel
       -> QFormer cross-attention với visual tokens
       -> Linear t5_proj
       -> concat prompt " A video of"
       -> Flan-T5
  -> train loss hoặc generated caption
  -> pycocoevalcap metrics
```

Quan trọng: code hiện tại **không có nhánh `direct_qformer_input`**, **không có `NormalizeVideoDirect`**, **không có `TemporalAdapter`**, và **không đọc feature theo thư mục `.pt/.pkl`** trong dataloader chính. Các feature phải nằm trong một pickle dictionary.

## 3. Feature extraction

Repo có thư mục `VideoFeatureExtractor/` để extract S3D feature từ video raw:

```text
input.csv(video_path, feature_path)
  -> VideoFeatureExtractor/extract.py
  -> VideoLoader
  -> Preprocessing
  -> S3D hoặc ResNeXt model
  -> .npy feature
  -> convert_video_feature_to_pickle.py
  -> pickle dictionary
```

Trong pipeline train hiện tại ở `run.sh`, feature không được extract trong repo tại runtime. `--features_path` trỏ tới pickle đã có sẵn. Vì vậy, dù feature đến từ S3D, CLIP, SigLIP hay module ngoài repo, điều kiện bắt buộc với code hiện tại là:

```text
features_path = pickle file
pickle content = dict[key -> array shape (T, D)]
D = --video_dim
```

Với lệnh trong `run.sh`, `D = 768`.

## 4. Dataloader

`data/dataloader_factory.py` chọn dataloader theo `--datatype`:

```text
--datatype msrvtt  -> MSRVTT_Caption_DataLoader
--datatype youcook -> Youcook_Caption_DataLoader
```

### MSRVTT

`dataloaders/dataloader_msrvtt_caption.py` nhận:

- `csv_path`: file csv train/test.
- `json_path`: `MSRVTT_data.json`.
- `features_path`: pickle dictionary.
- `split_type`: `train`, `val`, hoặc `test`.

Split được hard-code theo danh sách video trong JSON:

```text
train: video index 0..6512
val  : video index 6513..7009
test : video index 7010..
```

Với train thường, dataset expand theo từng caption. Với `--scst`, train chuyển sang một sample trên mỗi video và random một caption khi `__getitem__`.

### YouCookII

`dataloaders/dataloader_youcook_caption.py` nhận:

- `csv`
- `data_path`: pickle chứa transcript/caption theo segment.
- `features_path`: pickle feature.

YouCookII dùng `start/end` trong annotation để cắt đoạn feature tương ứng:

```text
start = int(start_time * feature_framerate)
end   = int(end_time * feature_framerate) + 1
video_slice = video_features[start:end]
```

### Tensor batch trả ra

Cả hai dataloader trả về cùng format:

```text
pairs_text
pairs_mask
pairs_segment
video
video_mask
pairs_masked_text
pairs_token_labels
masked_video
video_labels_index
pairs_input_caption_ids
pairs_decoder_mask
pairs_output_caption_ids
pairs_t5_output_caption_ids
sample_index
```

`sample_index` được `trainers/trainer.py` dùng để lấy toàn bộ ground-truth captions khi train SCST.

## 5. Xử lý video trong dataloader

Video feature được pad/truncate về:

```text
video      : (1, max_frames, feature_size)
video_mask : (1, max_frames)
```

Nếu `T > max_frames`, code hiện tại cắt phần đầu:

```text
video_slice = video_slice[:max_frames]
```

Không có average pooling/chunk pooling trong dataloader hiện tại.

MSRVTT dùng toàn bộ feature theo video. YouCookII cắt theo segment thời gian trước, rồi mới truncate theo `max_frames`.

## 6. Text và caption labels

Dataloader tạo hai nhóm label caption:

```text
BERT legacy caption:
  pairs_input_caption_ids
  pairs_decoder_mask
  pairs_output_caption_ids

T5 caption:
  pairs_t5_output_caption_ids
```

Nhánh caption hiện tại dùng **T5 labels** làm loss thật. Trong `modules/modeling.py`, `_get_t5_caption_loss()` sẽ raise lỗi nếu thiếu `t5_output_caption_ids`, vì `pairs_output_caption_ids` dùng BERT vocab và không hợp lệ cho Flan-T5.

Các tensor BERT legacy vẫn tồn tại để giữ tương thích với UniVL gốc, pretrain và một số nhánh cũ.

## 7. Khởi tạo model

`utils/model_utils.py::init_model()` gọi:

```text
UniVL.from_pretrained(
  bert_model,
  visual_model,
  cross_model,
  decoder_model,
  state_dict=init_model nếu có,
  task_config=args
)
```

`UniVLPreTrainedModel._filter_init_model_state_dict()` chỉ load các prefix:

```text
bert.
visual.
Qformer.
query_tokens
qformer_visual_proj.
t5_model.
t5_proj.
normalize_video.
```

Các tensor khác trong checkpoint bị bỏ qua.

## 8. Kiến trúc `UniVL` hiện tại

Trong `modules/modeling.py::UniVL`, các khối chính là:

| Khối | Vai trò |
| --- | --- |
| `BertModel` | Text encoder legacy, chỉ cần cho stage-one, pretrain hoặc retrieval. Caption fine-tune hiện tại thường skip encoder này. |
| `VisualModel` | Transformer visual encoder của UniVL. Feature sau `NormalizeVideo` đi qua module này trước QFormer. |
| `CrossModel` | Được khởi tạo ở stage two, chủ yếu còn phục vụ cấu trúc legacy/retrieval. |
| `Qformer` + `query_tokens` | BLIP2 QFormer cross-attend lên visual tokens. |
| `qformer_visual_proj` | Linear nếu `VisualModel.hidden_size != qformer_vision_width`, ví dụ `768 -> 1408`. |
| `T5ForConditionalGeneration` | Flan-T5 sinh caption. Base T5 bị freeze. |
| `t5_proj` | Map QFormer hidden size sang T5 hidden size. |
| `LoRA` | Nếu bật `--lora`, train LoRA trên attention modules `q`, `k`, `v`, `o` của T5. |

`--freeze_vit` freeze tham số `VisualModel`.

## 9. Forward khi train caption

Trong `UniVL.forward()`:

```text
input_ids/token_type_ids/attention_mask/video_mask
  -> flatten từ (B, 1, L) về (B, L)

video
  -> NormalizeVideo
  -> shape (B, max_frames, video_dim)
  -> LayerNorm(video_dim)
```

Với caption-only stage two:

```text
_need_text_encoder = False
visual_output = get_visual_output(video, video_mask, shaped=True)
```

`get_visual_output()` chạy:

```text
VisualModel(video, video_mask)
  -> visual_layers[-1]
  -> visual_output
```

Sau đó loss caption:

```text
visual_output + video_mask
  -> _get_t5_caption_loss()
```

## 10. QFormer + Flan-T5 caption branch

`_get_t5_caption_loss()` gọi `_build_t5_encoder_inputs()`:

```text
visual_output
  -> _get_cross_output()
  -> qformer_visual_proj
  -> QFormer(query_tokens cross-attend visual tokens)
  -> cross_output
  -> t5_proj
  -> inputs_t5
  -> concat prompt embedding " A video of"
  -> inputs_embeds, encoder_atts
```

T5 không nhận `input_ids` text encoder như bài toán text-to-text thông thường. Nó nhận `inputs_embeds` đã ghép từ:

```text
QFormer video-conditioned embeddings + prompt embeddings
```

Loss XE:

```text
t5_model(
  inputs_embeds=inputs_embeds,
  attention_mask=encoder_atts,
  decoder_attention_mask=output_mask,
  labels=t5_output_caption_ids với pad -> -100
)
```

## 11. SCST training

Nếu bật `--scst`, `_get_t5_caption_loss()` trộn:

```text
loss = scst_alpha * XE loss + (1 - scst_alpha) * SCST loss
```

`scst_alpha` không có CLI arg trong `main_task_caption.py`, nên mặc định trong model là `0.7` nếu không tự thêm vào `args`.

SCST flow:

```text
generate sampled captions:
  do_sample=True, top_p=0.9, temperature=0.8, num_return_sequences=scst_num_samples

generate baseline:
  do_sample=False

reward:
  CIDEr(sampled captions, all GT refs nếu có)
  - CIDEr(baseline caption, all GT refs nếu có)

loss:
  - mean_log_prob(sampled_caption) * advantage
```

Với MSRVTT, `trainers/trainer.py` lấy toàn bộ captions của video từ `video_sentences_dict` để làm references cho CIDEr.

## 12. Inference và evaluation

`inference/caption_generator.py::eval_epoch()` chạy:

```text
model.eval()
for batch in test_dataloader:
  loss, visual_output = model(...)
  generated_ids = model.generate_caption_ids(visual_output, video_mask)
  captions = model.t5_tokenizer.batch_decode(generated_ids)
```

`generate_caption_ids()` gọi Flan-T5:

```text
do_sample=False
num_beams=args.eval_beam_size hoặc beam_size hoặc 5
max_length=max_txt_len
repetition_penalty=1.2
length_penalty=1.0
```

Kết quả được lưu:

```text
output_dir/hyp.txt
output_dir/ref.txt
output_dir/hyp_complete_results.txt nếu dataset hỗ trợ
```

Metric dùng `pycocoevalcap`:

```text
Bleu_1, Bleu_2, Bleu_3, Bleu_4
METEOR
ROUGE_L
CIDEr
```

Với MSRVTT eval, code dùng nhiều reference caption trên mỗi video.

## 13. Optimizer

`utils/optimizer_utils.py::prep_optimizer()` chia learning rate theo prefix:

| Nhóm parameter | Prefix | LR |
| --- | --- | --- |
| BERT | `bert.` | `args.lr * coef_lr` |
| QFormer branch | `Qformer.`, `query_tokens`, `qformer_visual_proj.` | `args.lr_qformer` |
| T5 branch | `t5_model.`, `t5_proj.` | `args.lr_lora` |
| Khác | còn lại, gồm `visual.`, `cross.`, `normalize_video.` | `args.lr` |

Schedule:

```text
--scst false -> warmup_linear
--scst true  -> warmup_constant
```

Model được wrap bằng `DistributedDataParallel(find_unused_parameters=True)`.

## 14. Checkpoint và output

Trong training caption:

```text
mỗi epoch:
  train_epoch()
  save pytorch_model.bin.last.0
  eval_epoch()
  nếu CIDEr tốt hơn hoặc bằng CIDEr nhưng BLEU_4 tốt hơn:
    save pytorch_model.bin.best.0

sau train:
  load best
  eval lại
  save hyp_best.txt, ref_best.txt, hyp_complete_results_best.txt
```

`main_pretrain.py` có cơ chế checkpoint riêng:

```text
pytorch_model.bin.pretrain.{epoch}
pytorch_model.bin.checkpoint_{bert_model}_{max_words}_{max_frames}.checkpoint
```

## 15. Pretrain pipeline legacy

`main_pretrain.py` là pipeline pretrain UniVL gốc trên HowTo100M:

```text
main_pretrain.py
  -> Youtube_DataLoader
  -> UniVL
  -> stage one hoặc stage two pretrain
```

Loss pretrain gồm:

- retrieval/similarity loss.
- MLM: `_calculate_mlm_loss()`.
- MFM: `_calculate_mfm_loss()`.
- caption decoder loss nếu stage two có decoder input.

Đây là nhánh legacy, không phải lệnh đang chạy trong `run.sh`.

## 16. Các script thử nghiệm

Ngoài pipeline chính còn có:

- `main_task_caption_test.py`: biến thể test visual encoder, có logic loại bỏ `visual.` khỏi checkpoint để init VisualModel từ đầu.
- `main_task_caption_no_visual.py`: biến thể thí nghiệm dùng fake video features zero/random/gaussian.
- `test_t5_decoder.py`, `show_visual_hidden_size.py`: script kiểm tra.
- Notebook `univl.ipynb`, `run.ipynb`, `run_final.ipynb`, `extract_clip_features.ipynb`: phục vụ thử nghiệm/thao tác thủ công.

## 17. Shape summary cho lệnh hiện tại

Với MSRVTT trong `run.sh`:

```text
feature pickle:
  key: video_id
  value: (T, 768)

dataloader:
  truncate/pad -> video: (B, 1, 48, 768)
  video_mask: (B, 1, 48)

UniVL.forward:
  view -> video: (B, 48, 768)
  NormalizeVideo + LayerNorm(768)
  VisualModel -> visual_output: (B, 48, visual_hidden)

QFormer branch:
  qformer_visual_proj nếu cần -> (B, 48, 1408)
  QFormer query tokens -> (B, num_query_token, qformer_hidden)
  t5_proj -> (B, num_query_token, t5_hidden)
  concat prompt -> (B, num_query_token + prompt_len, t5_hidden)

Flan-T5:
  train -> XE hoặc XE + SCST
  eval -> beam search -> caption text
```

## 18. Điểm cần nhớ

- `features_path` của dataloader chính phải là pickle dictionary, không phải thư mục feature rời.
- Nếu đổi extractor, `--video_dim` phải khớp chiều cuối của feature.
- Caption loss hiện tại bắt buộc có `pairs_t5_output_caption_ids`.
- Caption fine-tune hiện tại skip BERT text encoder, nhưng vẫn chạy VisualModel trước QFormer.
- Base Flan-T5 bị freeze; khi bật `--lora`, phần trainable chính trong T5 là LoRA adapters cùng `t5_proj`.
- Tài liệu/README gốc vẫn mô tả UniVL nguyên bản; pipeline đang chạy trong repo này đã khác ở nhánh QFormer + Flan-T5.
