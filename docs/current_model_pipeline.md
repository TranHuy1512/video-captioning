# Pipeline luồng hoạt động model hiện tại

Tài liệu này mô tả luồng đang chạy trong repo cho bài toán video captioning, với giả định `--features_path` trỏ tới các feature đã được extract trước bằng **Google SigLIP** và đã đi qua module/class **STC** ở bước tiền xử lý. Phần SigLIP + STC là bước ngoài runtime của `modules/modeling.py`; model hiện tại nhận kết quả cuối cùng dưới dạng sequence feature `(T, D)`.

## 1. Tổng quan pipeline

```text
Video gốc
  -> Google SigLIP feature extractor
  -> STC feature module/class
  -> lưu feature vào --features_path (.pt / .pickle / .pkl)
  -> MSRVTT_Caption_DataLoader / Youcook_Caption_DataLoader
  -> UniVL.forward()
  -> NormalizeVideoDirect hoặc NormalizeVideo
  -> direct_qformer_input:
       TemporalAdapter / Conv1d / Linear
       optional AdaptiveAvgPool1d
       BLIP2 QFormer
       Linear t5_proj
       prompt " A video of"
       Flan-T5 decoder/generator
  -> caption text
```

Với cấu hình hiện tại nên hiểu model theo nhánh `--stage_two --task_type caption --direct_qformer_input`: feature từ `feature_path` được đưa trực tiếp vào QFormer pipeline, bỏ qua `VisualModel` Transformer cũ của UniVL.

## 2. Bước tiền xử lý: SigLIP + STC

Đầu vào ban đầu là video gốc. Bên ngoài repo/model runtime, video được encode bằng SigLIP của Google để lấy feature thị giác giàu ngữ nghĩa. Sau đó feature này đi qua class/module STC để bổ sung hoặc tái cấu trúc thông tin không gian-thời gian trước khi lưu xuống `feature_path`.

Kết quả mà repo cần là một tensor/array 2 chiều:

```text
(T, D)
```

Trong đó:

- `T`: số token/frame/segment sau SigLIP + STC.
- `D`: chiều feature, phải khớp với `--video_dim`.
- Ví dụ code comment có nhắc chuỗi dài từ STC như `T = 1352`; vì vậy repo có thêm `--adapter_pool_size` để giảm sequence trước khi vào QFormer.

Model hiện tại không gọi trực tiếp SigLIP hoặc STC trong `forward`. Nó chỉ đọc feature đã lưu sẵn từ `--features_path`.

## 3. Nạp feature từ `features_path`

Lớp chính cho MSRVTT là `MSRVTT_Caption_DataLoader` trong `dataloaders/dataloader_msrvtt_caption.py`.

Nó hỗ trợ 2 kiểu `features_path`:

- Một file pickle lớn chứa dictionary `{video_id: np.array}`.
- Một thư mục chứa từng file feature theo video: `.pt`, `.pickle`, hoặc `.pkl`.

Với thư mục feature, dataloader build index file bằng `_build_feature_file_index()`, sau đó đọc từng video bằng `_load_feature()`:

- `_load_pt_feature(video_id)`: đọc `video_id.pt` bằng `torch.load`.
- `_load_pickle_feature_file(video_id)`: đọc `video_id.pickle` hoặc `video_id.pkl`.
- `_tensor_from_pt_object()` và `_array_from_pickle_object()` cho phép object là tensor trực tiếp hoặc dict có key như `features`, `feature`, `video`, `video_features`, `embeddings`.

Feature hợp lệ sau khi đọc phải có shape `(T, D)`. Nếu có batch thừa `(1, T, D)`, dataloader squeeze về `(T, D)`.

## 4. Cắt/pool theo `max_frames` trong dataloader

Trong `_get_video()`, dataloader tạo:

```text
video      : (num_pair, max_frames, feature_size)
video_mask : (num_pair, max_frames)
```

Nếu `T <= max_frames`, feature được copy trực tiếp vào phần đầu của tensor `video`, phần còn lại padding zero, `video_mask` đánh dấu các vị trí thật bằng `1`.

Nếu `T > max_frames`, MSRVTT dataloader không lấy đơn giản `T` token đầu. Nó chia sequence thành `max_frames` chunks bằng `np.array_split`, rồi lấy mean từng chunk:

```text
(T, D) -> split thành max_frames chunk -> mean mỗi chunk -> (max_frames, D)
```

Điều này quan trọng với feature từ STC nếu số token dài. Nó giữ thông tin trải đều toàn bộ sequence trước khi đưa vào model.

## 5. Text/caption label trong dataloader

Dataloader tạo song song nhiều tensor text:

- `pairs_text`, `pairs_mask`, `pairs_segment`: input theo tokenizer BERT legacy.
- `pairs_masked_text`, `pairs_token_labels`: dùng cho nhánh pretrain MLM nếu bật.
- `pairs_input_caption_ids`, `pairs_decoder_mask`, `pairs_output_caption_ids`: caption theo BERT vocab, còn giữ để tương thích code cũ.
- `pairs_t5_output_caption_ids`: caption raw được tokenize bằng `T5TokenizerFast`, đây mới là label thật dùng cho Flan-T5 CE loss.

Trong caption fine-tuning hiện tại, nhánh T5 cần `pairs_t5_output_caption_ids`; code sẽ raise lỗi nếu thiếu label T5.

## 6. Khởi tạo model `UniVL`

Lớp model chính là `UniVL` trong `modules/modeling.py`.

Các lớp/cấu phần được khởi tạo đáng chú ý:

| Thành phần | Class/module | Vai trò |
| --- | --- | --- |
| Text encoder legacy | `BertModel` | Dùng cho stage-one/retrieval/pretrain; caption-only direct mode thường không cần chạy. |
| Visual encoder legacy | `VisualModel` | Transformer visual cũ của UniVL; bị skip khi bật `--direct_qformer_input`. |
| Cross encoder legacy | `CrossModel` | Vẫn được khởi tạo trong stage two, nhưng caption generation hiện dùng BLIP2 QFormer + T5. |
| Direct normalizer | `NormalizeVideoDirect` | Cast feature sang float và reshape; không LayerNorm vì feature đã được extract/normalize bên ngoài. |
| Legacy normalizer | `NormalizeVideo` | Dùng khi không direct; có `LayerNorm(video_dim)`. |
| Feature adapter | `TemporalAdapter`, `nn.Conv1d`, hoặc `nn.Linear` | Map `video_dim -> qformer_vision_width` trước QFormer. |
| Pooling tùy chọn | `nn.AdaptiveAvgPool1d` | Giảm số token sau adapter nếu `--adapter_pool_size > 0`. |
| Query transformer | `Blip2Base.init_Qformer()` | Tạo `Qformer` và `query_tokens`. |
| Caption LM | `T5ForConditionalGeneration` | Flan-T5 sinh caption; base weights bị freeze. |
| LoRA tùy chọn | `peft.get_peft_model` | Train LoRA trên các module attention `q`, `k`, `v`, `o` nếu bật `--lora`. |
| Projection sang T5 | `nn.Linear` `t5_proj` | Map hidden size QFormer sang hidden size của T5. |

## 7. Nhánh `direct_qformer_input`

Khi bật `--direct_qformer_input`, model set:

```python
self.visual = None
self.normalize_video = NormalizeVideoDirect()
```

Ý nghĩa:

- Feature từ SigLIP + STC không đi qua `VisualModel`.
- Không bị giới hạn bởi positional embedding của visual encoder cũ.
- Feature được đưa thẳng sang adapter rồi QFormer.

Điều kiện bắt buộc trong `main_task_caption.py`:

- `--task_type caption`
- `--stage_two`
- không dùng `--do_pretrain`

## 8. `TemporalAdapter`: adapter STC-like trong model

`TemporalAdapter` là adapter mặc định khi `--qformer_adapter_type temporal`.

Luồng trong `TemporalAdapter.forward(x)`:

```text
x: (B, T, video_dim)
  -> down_proj Linear(video_dim, qformer_vision_width)
  -> transpose sang (B, qformer_vision_width, T)
  -> depthwise Conv1d theo trục thời gian
  -> GELU
  -> transpose về (B, T, qformer_vision_width)
  -> gate Sigmoid(Linear(residual))
  -> trộn gated temporal feature với residual
  -> LayerNorm
```

Adapter này không phải class STC tiền xử lý mà user nhắc tới. Nó là adapter temporal trong model, được comment là “STC-like” vì có depthwise temporal convolution và gating để trộn thông tin theo trục thời gian sau khi feature đã được extract.

## 9. Forward pass khi training caption

Trong `UniVL.forward()`:

1. Dataloader batch được flatten:

```text
input_ids, attention_mask, video_mask -> view(-1, seq_len)
video -> normalize_video(video)
```

2. Với caption-only direct mode, text encoder không cần chạy:

```python
_need_text_encoder = False
visual_output = self.get_visual_output(video, video_mask, shaped=True)
```

3. `get_visual_output()` trả về trực tiếp `video` nếu `direct_qformer_input=True`.

4. Caption loss gọi `_get_t5_caption_loss(visual_output, video_mask, ..., t5_output_caption_ids)`.

5. `_build_t5_encoder_inputs()` tạo input cho T5 encoder:

```text
visual_output
  -> _get_cross_output()
  -> _project_visual_for_qformer()
  -> QFormer query cross-attention
  -> t5_proj
  -> concat prompt embedding " A video of"
  -> inputs_embeds, encoder_atts
```

6. `_compute_xe_caption_loss()` gọi Flan-T5:

```python
self.t5_model(
    inputs_embeds=inputs_embeds,
    attention_mask=encoder_atts,
    decoder_attention_mask=output_mask,
    labels=targets,
)
```

Loss là teacher-forcing cross entropy trên `pairs_t5_output_caption_ids`, padding được đổi thành `-100` để ignore.

Nếu bật `--scst`, `_get_t5_caption_loss()` trộn:

```text
scst_alpha * XE loss + (1 - scst_alpha) * SCST loss
```

SCST dùng generated captions, reward CIDEr và all ground-truth refs nếu dataloader cung cấp được `gt_refs`.

## 10. QFormer và T5 encoder input

Trong `_get_cross_output()`:

- `query_tokens`: learnable query tokens, mặc định `num_query_token=32`.
- `visual_for_qformer`: feature sau adapter/pooling, shape `(B, T_or_pool, qformer_vision_width)`.
- `image_atts`: attention mask cho visual tokens. Nếu có `adapter_pool_size`, tất cả token sau pooling được xem là valid.

QFormer chạy cross-attention:

```python
self.Qformer.bert(
    query_embeds=query_tokens,
    encoder_hidden_states=visual_for_qformer,
    encoder_attention_mask=image_atts,
    return_dict=True,
)
```

Output:

```text
cross_output: (B, num_query_token, QFormer_hidden)
```

Sau đó:

```text
cross_output -> t5_proj -> (B, num_query_token, T5_hidden)
prompt " A video of" -> T5 token embedding
concat -> inputs_embeds cho T5 encoder
```

Tức T5 không nhận token text đầu vào dạng `input_ids` như thông thường; nó nhận trực tiếp `inputs_embeds` gồm video-conditioned QFormer embeddings cộng prompt embedding.

## 11. Inference/generation

Trong evaluation, `inference/caption_generator.py` gọi:

```python
loss, visual_output = model(...)
generated_ids = model.generate_caption_ids(
    visual_output,
    video_mask,
    num_beams=eval_beam_size,
    max_length=max_length,
)
```

`generate_caption_ids()` build lại T5 encoder inputs từ `visual_output`, rồi gọi:

```python
self.t5_model.generate(
    inputs_embeds=inputs_embeds,
    attention_mask=encoder_atts,
    do_sample=False,
    num_beams=num_beams,
    max_length=max_length,
    repetition_penalty=1.2,
    length_penalty=1.0,
)
```

Sau đó `batch_decode()` bằng `self.t5_tokenizer` để ra caption text.

## 12. Các lớp chính được dùng

### Dataloader

- `MSRVTT_Caption_DataLoader`
- `Youcook_Caption_DataLoader`
- `DATALOADER_DICT`

### Model core

- `UniVLPreTrainedModel`
- `UniVL`
- `NormalizeVideoDirect`
- `NormalizeVideo`
- `TemporalAdapter`

### UniVL legacy modules

- `BertModel`
- `BertConfig`
- `BertOnlyMLMHead`
- `VisualModel`
- `VisualConfig`
- `VisualOnlyMLMHead`
- `CrossModel`
- `CrossConfig`
- `DecoderConfig`

### QFormer/T5 caption branch

- `Blip2Base`
- `Qformer` từ `Blip2Base.init_Qformer()`
- `query_tokens`
- `T5TokenizerFast`
- `T5Config`
- `T5ForConditionalGeneration`
- `LoraConfig`
- `get_peft_model`

### Training/eval helpers

- `train_epoch`
- `eval_epoch`
- `CaptionEvaluator`
- `BertAdam`

## 13. Optimizer và tham số trainable

`utils/optimizer_utils.py` gom parameter theo prefix:

- BERT: prefix `bert.`, learning rate `args.lr * coef_lr`.
- QFormer branch: `Qformer.`, `query_tokens`, `qformer_visual_proj.`, `normalize_video.`, learning rate `lr_qformer`.
- T5 branch: `t5_model.`, `t5_proj.`, learning rate `lr_lora`.
- Other params: learning rate mặc định `args.lr`.

Base T5 được freeze ngay khi khởi tạo. Nếu bật `--lora`, chỉ LoRA adapter trong T5 attention và `t5_proj`/QFormer branch là phần chính được fine-tune.

## 14. Tóm tắt luồng dữ liệu shape

Ví dụ với direct mode:

```text
feature file từ SigLIP + STC:
  (T, D=video_dim)

dataloader:
  nếu T > max_frames: average chunk -> (max_frames, D)
  batch -> video: (B, max_frames, D)
  video_mask: (B, max_frames)

NormalizeVideoDirect:
  (B, max_frames, D)

TemporalAdapter:
  (B, max_frames, D) -> (B, max_frames, qformer_vision_width)

optional adapter_pool:
  (B, max_frames, qformer_vision_width)
  -> (B, adapter_pool_size, qformer_vision_width)

QFormer:
  query_tokens cross-attend visual tokens
  -> (B, num_query_token, qformer_hidden)

t5_proj:
  -> (B, num_query_token, t5_hidden)

concat prompt embedding:
  -> (B, num_query_token + prompt_len, t5_hidden)

Flan-T5:
  training -> CE/SCST loss
  inference -> generated caption ids -> caption text
```

## 15. Điểm cần nhớ

- `feature_path` hiện là đầu vào feature đã extract sẵn, không phải video raw.
- SigLIP và STC nằm trước dataloader/model; repo chỉ load output của chúng.
- Với `--direct_qformer_input`, `VisualModel` của UniVL bị bỏ qua.
- `TemporalAdapter` trong `modules/modeling.py` là adapter temporal STC-like trong model, không thay thế bước STC tiền xử lý.
- Caption branch hiện đại của repo là QFormer + Flan-T5, không còn decoder Transformer legacy của UniVL gốc cho caption generation.
