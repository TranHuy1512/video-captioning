# PROJECT OVERVIEW — Video Captioning (UniVL VisualModel + BLIP-2 Q-Former + Flan-T5 LoRA + SCST)

> Tài liệu phân tích **chỉ đọc** (read-only) — không sửa code, không chạy train.
> Phạm vi: trạng thái working tree ngày 2026-10-20 (branch `main`, HEAD `dde5c76`, **có thay đổi chưa commit** ở `main_task_caption.py`, `modules/modeling.py`, `trainers/trainer.py`, `run.sh`, cùng file mới `utils/diversity_utils.py`, `tests/test_diversity_utils.py`, `evidents/`).
>
> Ký hiệu: ✅ = đã kiểm chứng trực tiếp từ code/log · ⚠️ = mâu thuẫn hoặc rủi ro phát hiện khi đọc · ❓ = **cần bạn xác nhận** (xem mục 8).

---

## 0. Tóm tắt nhanh

| Hạng mục | Thực tế trong code |
|---|---|
| Entry point | `run.sh` → `torchrun main_task_caption.py` (bắt buộc chạy qua `torchrun`, vì `init_process_group("nccl")` gọi ngay khi import — [main_task_caption.py:21-22](main_task_caption.py#L21-L22)) |
| Visual input | Feature **trích sẵn offline** (pickle `{video_id: (T, D)}`), không có vision encoder ảnh trong runtime |
| Temporal | `VisualModel` = Transformer 6 lớp của UniVL (position embedding học được), **bị freeze** khi có `--freeze_vit` |
| Bridge | BLIP-2 Q-Former (32 query, cross-attn mỗi 2 lớp) + `Linear(768→1408)` trước Q-Former + `t5_proj Linear(768→d_T5)` |
| LLM | `google/flan-t5-xl` (mặc định; `run.sh` không ghi đè), **freeze toàn bộ, bf16**, train **LoRA** (`r=32, α=64, dropout=0.05`, target **`q,v`**) |
| Loss | XE (teacher forcing) hoặc SCST (beam-based, reward CIDEr, baseline = mean reward của các beam) trộn XE theo `scst_alpha` |
| Data | MSRVTT, split cứng theo **index trong JSON**: 6513 train / 497 val / 2990 test; train mở rộng thành 130,260 cặp (video, caption) |
| Eval | pycocoevalcap BLEU-1..4, METEOR, ROUGE-L, CIDEr trên **test 2990 video, 20 ref/video**, beam 5 |
| Kết quả kiểm chứng được | Run `evidents/` (SCST α=1.0, LR ~1e-7): best CIDEr **0.6118**, BLEU-4 0.4332, METEOR 0.3095, ROUGE-L 0.6387 ([evidents/log.txt](evidents/log.txt)) |

---

## 1. Kiến trúc tổng thể & luồng dữ liệu

### 1.1 Sơ đồ luồng (đúng với code hiện tại)

```text
Video .mp4 (OFFLINE, ngoài train runtime)
  └─ trích frame 1 fps → CLIP encode_image từng frame (hoặc BLIP-2 feature extractor / S3D)
     → pickle {video_id: np.ndarray (T, D)}                      D = --video_dim = 768

MSRVTT_Caption_DataLoader.__getitem__                            dataloaders/dataloader_msrvtt_caption.py
  ├─ video  (1, max_frames=48, 768) zero-pad, cắt phần đầu nếu T>48;  video_mask (1, 48)
  ├─ BERT tensors legacy (input_ids chỉ là [CLS][SEP], output_caption_ids BERT vocab)
  └─ t5_output_caption_ids (1, max_txt_len=20)  ← nhãn thật cho loss

UniVL.forward                                                     modules/modeling.py
  NormalizeVideo: LayerNorm(768)                                 (B, 48, 768)
  VisualModel (6 lớp Transformer UniVL, frozen)                  (B, 48, 768)
  qformer_visual_proj: Linear 768 → 1408                         (B, 48, 1408)
  Q-Former.bert(query_embeds=32 query, cross-attn lên 48 token)  (B, 32, 768)
  t5_proj: Linear 768 → d_model T5 (2048 cho flan-t5-xl)         (B, 32, 2048)
  concat [32 query-embeds ; embed(" A video of")]                (B, 32+P, 2048)  → T5 encoder (inputs_embeds)
  Flan-T5 decoder
    ├─ train: XE  /  SCST(+XE)
    └─ eval : beam search (5 beam, max_length=20, repetition_penalty=1.2) → caption text
```

### 1.2 Các module và vị trí code

| Module | Vai trò | Code |
|---|---|---|
| `NormalizeVideo` | LayerNorm trên chiều feature, reshape về `(B, T, D)` | [modules/modeling.py:323-332](modules/modeling.py#L323-L332) |
| `VisualModel` (UniVL) | Linear `video_dim→768` + position embedding (max 512) + N lớp self-attention; `visual_num_hidden_layers=6` ghi đè config mặc định 1 lớp | [modules/module_visual.py:104-131](modules/module_visual.py#L104-L131), [modules/visual-base/visual_config.json](modules/visual-base/visual_config.json), [modules/modeling.py:374-376](modules/modeling.py#L374-L376) |
| Freeze VisualModel | `--freeze_vit` freeze **VisualModel của UniVL** (không phải ViT — không có ViT nào trong runtime) | [modules/modeling.py:377-383](modules/modeling.py#L377-L383) |
| `qformer_visual_proj` | Chỉ tạo khi `qformer_vision_width (1408) ≠ hidden_size (768)` | [modules/modeling.py:389-399](modules/modeling.py#L389-L399) |
| Q-Former | `BertLMHeadModel` từ `bert-base-uncased`, `encoder_width=1408`, `cross_attention_freq=2`, `query_length=32`; nạp trọng số từ `Salesforce/blip2-opt-6.7b-coco`; xoá `cls`, word/position embeddings, FFN nhánh text (giữ `intermediate_query/output_query`) | [modules/blip2.py:50-79](modules/blip2.py#L50-L79), [modules/modeling.py:400-412](modules/modeling.py#L400-L412) |
| Q-Former forward | query tokens cross-attend lên visual token, `encoder_attention_mask=video_mask` | [modules/modeling.py:599-620](modules/modeling.py#L599-L620) |
| `t5_proj` + prompt | Linear Q-Former→T5, nối với embedding prompt `" A video of"` | [modules/modeling.py:433](modules/modeling.py#L433), [modules/modeling.py:469-471](modules/modeling.py#L469-L471), [modules/modeling.py:622-649](modules/modeling.py#L622-L649) |
| Flan-T5 | `T5ForConditionalGeneration`, `dense_act_fn="gelu"` (giống LAVIS), toàn bộ tham số `requires_grad=False` và ép **bf16** | [modules/modeling.py:435-444](modules/modeling.py#L435-L444) |
| `BertModel` (text encoder UniVL) | Vẫn được khởi tạo nhưng **không chạy** ở caption fine-tune (`_need_text_encoder=False`) | [modules/modeling.py:367-371](modules/modeling.py#L367-L371), [modules/modeling.py:516-529](modules/modeling.py#L516-L529) |
| Sinh caption | `generate_caption_ids` / `generate_caption_text` | [modules/modeling.py:932-959](modules/modeling.py#L932-L959) |

⚠️ **README nói model "luôn khởi tạo cả BertModel + CrossModel"** — code hiện tại **không còn tạo `CrossModel`** (chỉ import `CrossConfig` để đọc config và kiểm tra `max_words + max_frames ≤ 1024`, [modules/modeling.py:357](modules/modeling.py#L357)). Log `evidents/` (đời code cũ hơn) vẫn in các key `cross.*` "not initialized" ([evidents/log.txt:149-190](evidents/log.txt)), chứng tỏ CrossModel đã bị xoá sau run đó.

### 1.3 Khởi tạo & nạp trọng số

1. `init_model` ([utils/model_utils.py:8-20](utils/model_utils.py#L8-L20)) → `UniVL.from_pretrained` ([modules/modeling.py:80-106](modules/modeling.py#L80-L106)).
2. Trong `__init__`: Q-Former nạp checkpoint BLIP-2 **trước**, sau đó `init_weights` cho mọi module **trừ** `Qformer, t5_model, t5_proj, qformer_visual_proj, query_tokens, normalize_video` ([modules/modeling.py:493-504](modules/modeling.py#L493-L504)) → `t5_proj` và `qformer_visual_proj` dùng init mặc định của PyTorch nếu không có `--init_model`.
3. Nếu có `--init_model`: lọc chỉ giữ prefix `bert. visual. Qformer. query_tokens qformer_visual_proj. t5_model. t5_proj. normalize_video.` ([modules/modeling.py:109-138](modules/modeling.py#L109-L138)), rồi chuẩn hoá key LoRA ↔ non-LoRA, kể cả **merge** adapter `B·A·(α/r)` vào trọng số gốc khi checkpoint có LoRA trên module mà model mới không target ([modules/modeling.py:141-321](modules/modeling.py#L141-L321)). Trọng số từ `init_model` **ghi đè** trọng số BLIP-2 vừa nạp.

⚠️ **Q-Former chỉ nạp được một phần trọng số BLIP-2.** Log ghi: `Loaded QFormer checkpoint ... tensors=144 query_tokens=True missing=192` ([evidents/log.txt:146](evidents/log.txt)). Nguyên nhân khả dĩ (suy luận từ code, chưa kiểm tra file checkpoint): repo `Salesforce/blip2-opt-6.7b-coco` theo định dạng HF transformers, key dạng `qformer.encoder.layer.N.attention.attention.query.*`, còn Q-Former local theo kiểu LAVIS dùng `attention.self.query.*` ([modules/Qformer.py:311](modules/Qformer.py#L311)). `map_blip2_qformer_key` chỉ thay prefix `qformer.`→`bert.` ([modules/blip2.py:134-161](modules/blip2.py#L134-L161)) nên các key Q/K/V self-attn & cross-attn (và `layernorm` của embeddings) không khớp và bị **bỏ qua im lặng** (`strict=False`). Các phần như `intermediate_query`, `output_query`, `*.output.dense` có tên trùng nên vẫn nạp được. → ❓ Q3.

---

## 2. Kết hợp với LLM: frozen vs trainable

| Thành phần | Trạng thái | LR (theo `run.sh` hiện tại) | Nguồn |
|---|---|---|---|
| Flan-T5-XL (base weights) | **Frozen**, bf16 | — | [modules/modeling.py:442-444](modules/modeling.py#L442-L444) |
| LoRA adapters (`q`,`v` của mọi attention T5, cả encoder & decoder) | Train, fp32 (PEFT mặc định autocast adapter về fp32 — theo comment) | `lr_lora = 1e-6` | [modules/modeling.py:446-467](modules/modeling.py#L446-L467) |
| `t5_proj` | Train | `lr_lora = 1e-6` (cùng nhóm "t5_decoder") | [utils/optimizer_utils.py:15-16](utils/optimizer_utils.py#L15-L16) |
| Q-Former + `query_tokens` + `qformer_visual_proj` | Train | `lr_qformer = 5e-7` | [utils/optimizer_utils.py:12-13](utils/optimizer_utils.py#L12-L13) |
| `VisualModel` | **Frozen** (`--freeze_vit`), `train()` bị vô hiệu hoá (`disabled_train`) | — | [modules/modeling.py:377-383](modules/modeling.py#L377-L383) |
| `normalize_video` (LayerNorm) | Train — rơi vào nhóm "other" **không có `lr` riêng → dùng `--lr 3e-5`** | `3e-5` | [utils/optimizer_utils.py:44,48](utils/optimizer_utils.py#L44) |
| `BertModel` | `requires_grad=True` nhưng không có gradient (không dùng trong forward); DDP `find_unused_parameters=True` | `lr * coef_lr` (= `lr` khi có `--init_model`) | [main_task_caption.py:197-200](main_task_caption.py#L197-L200), [utils/optimizer_utils.py:60-61](utils/optimizer_utils.py#L60-L61) |

Ghi chú:
- ⚠️ `normalize_video` được cập nhật với LR **gấp 30–60 lần** LoRA/Q-Former. Có thể không cố ý. → ❓ Q11.
- ⚠️ Tên biến trong `prep_optimizer` bị đảo nghĩa: `no_decay_*` thực chất là nhóm **có** weight decay 0.01, `decay_*` là nhóm **không** decay ([utils/optimizer_utils.py:27-49](utils/optimizer_utils.py#L27-L49)). Hành vi đúng (bias/LayerNorm không decay), chỉ là tên gây nhầm.
- Precision: forward caption chạy dưới `torch.amp.autocast(bfloat16)` ([modules/modeling.py:766](modules/modeling.py#L766), [modules/modeling.py:938](modules/modeling.py#L938)). Không dùng apex dù có cờ `--fp16`.

---

## 3. Huấn luyện

### 3.1 Cấu hình LoRA

| Tham số | Default CLI | `run.sh` hiện tại | Run `evidents/` | Nguồn |
|---|---|---|---|---|
| `--lora` | off | **on** | on | [main_task_caption.py:127](main_task_caption.py#L127) |
| `lora_r` | 16 | **32** | 32 | [main_task_caption.py:128](main_task_caption.py#L128) |
| `lora_alpha` | 32 | **64** (scaling α/r = 2) | 64 | [main_task_caption.py:129](main_task_caption.py#L129) |
| `lora_dropout` | 0.05 | 0.05 | 0.05 | [main_task_caption.py:130](main_task_caption.py#L130) |
| `lora_target_modules` | `'q,v'` | (không set → **`q,v`**) | `['q','v']` | [main_task_caption.py:131-133](main_task_caption.py#L131-L133), [main_task_caption.py:142](main_task_caption.py#L142) |
| `task_type` | `SEQ_2_SEQ_LM` | | | [modules/modeling.py:452-459](modules/modeling.py#L452-L459) |

⚠️ README (mục 4 và Hạn chế #7) nói LoRA target **`q,k,v,o`**, nhưng code/run.sh hiện dùng **`q,v`**. Help text gợi ý `q,k,v,o` là "old 4-module config", và code có nhánh harmonise checkpoint `q,k,v,o` → model `q,v` bằng cách merge LoRA `k,o` vào trọng số gốc ([modules/modeling.py:239-319](modules/modeling.py#L239-L319)). → ❓ Q6.

### 3.2 Các giai đoạn train

Code hỗ trợ (cờ `--stage_two` bắt buộc cho caption; không có nó, model đi nhánh stage-one retrieval và không tạo Q-Former/T5 — [modules/modeling.py:359-365](modules/modeling.py#L359-L365), [modules/modeling.py:386](modules/modeling.py#L386)):

| Giai đoạn | Cách bật | Bằng chứng trong repo |
|---|---|---|
| (Legacy) UniVL pretrain HowTo100M | `main_pretrain.py` | Có code, không thấy được dùng cho nhánh hiện tại |
| **Stage A – XE fine-tune** | `--do_train --stage_two` không `--scst` | ⚠️ **Không có lệnh nào trong `run.sh`**. Chỉ biết gián tiếp: SCST luôn khởi động từ checkpoint `ckpts/ckpt_msrvtt_caption_ver_1/...best.0` / `ckpt_msrvtt_caption_1/...` và `evidents/hyp_xe.txt` được gọi là output "XE-only". → ❓ Q4, Q9 |
| **Stage B – SCST (+XE)** | thêm `--scst [--scst_alpha a] [--scst_diversity_weight w]` + `--init_model <ckpt XE>` | [run.sh:44-61](run.sh#L44-L61) |

Lịch LR: `BertAdam` với `warmup_proportion=0.1`; schedule **`warmup_constant` khi SCST**, `warmup_linear` khi XE ([utils/optimizer_utils.py:53-58](utils/optimizer_utils.py#L53-L58)). Gradient clipping 2 lần: global `clip_grad_norm_(…,1.0)` trong trainer ([trainers/trainer.py:57](trainers/trainer.py#L57)) **và** clip theo từng tham số trong `BertAdam.step` ([modules/optimization.py:134-135](modules/optimization.py#L134-L135)). Batch: `--batch_size 128` chia `gradient_accumulation_steps=2` → **64 mẫu/step** ([main_task_caption.py:150](main_task_caption.py#L150)).

Cấu hình `run.sh` hiện tại (chưa có kết quả trong repo): 5 epoch, `lr 3e-5`, `lr_qformer 5e-7`, `lr_lora 1e-6`, `scst_alpha 0.5`, `scst_diversity_weight 0.3`, `max_txt_len 20`, `max_frames 48`, feature CLIP ViT-L/14 ([run.sh:44-61](run.sh#L44-L61)).

### 3.3 Hàm loss tổng

`_get_t5_caption_loss` ([modules/modeling.py:754-793](modules/modeling.py#L754-L793)):

```text
cross_output = QFormer(visual_output)                          (chạy 1 lần, dùng lại)
L_div  = mean_{b, i≠j} ReLU(cos(q_i, q_j))        nếu train & qformer_diversity_weight>0   (L733-752)
nếu train & --scst:
    L_cap = α·L_SCST + (1-α)·L_XE    nếu α<1      (L782-789)
    L_cap = L_SCST                   nếu α=1 (mặc định)
ngược lại:
    L_cap = L_XE
L = L_cap + qformer_diversity_weight · L_div
```

- **L_XE** ([modules/modeling.py:651-665](modules/modeling.py#L651-L665)): CE chuẩn của HF T5 với `labels = t5_output_caption_ids` (pad → -100). Mỗi mẫu train chỉ mang **1 caption GT**.
- `t5_output_caption_ids` bắt buộc — nếu thiếu sẽ raise, vì `output_caption_ids` là BERT vocab ([modules/modeling.py:757-761](modules/modeling.py#L757-L761)).

### 3.4 Cài đặt SCST chi tiết — `_compute_scst_caption_loss` ([modules/modeling.py:818-930](modules/modeling.py#L818-L930))

| Bước | Chi tiết | Dòng |
|---|---|---|
| **Sampling** | `generate()` dưới `no_grad`: **beam search xác định** (`do_sample=False`), `num_beams = num_return_sequences = scst_num_samples = 5`, `max_length = max_txt_len (20)`, `repetition_penalty=1.0` → K=5 candidate/video (320 candidate/step) | [L834-847](modules/modeling.py#L834-L847) |
| **Re-score có gradient** | Teacher-forced forward lại trên chính chuỗi sinh ra (`decoder_input_ids = seq[:, :-1]`, `labels = seq[:, 1:]`), lấy log-prob token, **trung bình theo độ dài** (bỏ pad) → `sequences_scores (B, K)`. Đây là bản fix cho lỗi `element 0 of tensors does not require grad` trong [error.txt](error.txt) | [L852-876](modules/modeling.py#L852-L876) |
| **Reference** | Nếu trainer truyền `gt_refs` (MSRVTT: luôn có) → **toàn bộ 20 caption GT** của video; ngược lại fallback 1 caption | [L884-895](modules/modeling.py#L884-L895), [trainers/trainer.py:34-40](trainers/trainer.py#L34-L40) |
| **Reward** | PTBTokenizer cả ref & cand → CIDEr-D (n=4, σ=6) bằng `CorpusCider` (DF tính trước từ 6513 video train, 130,260 ref) | [L897-899](modules/modeling.py#L897-L899), [utils/cider_utils.py](utils/cider_utils.py), [main_task_caption.py:203-205](main_task_caption.py#L203-L205) |
| **Diversity penalty (mới, opt-in)** | `reward -= w · max_{j≠i} overlap(cand, top-beam_j)` với overlap = trung bình tỉ lệ unigram+bigram của cand xuất hiện trong caption top-beam của video khác cùng batch (không IDF) | [L909-921](modules/modeling.py#L909-L921), [utils/diversity_utils.py:42-82](utils/diversity_utils.py#L42-L82) |
| **Baseline** | **mean reward của K beam cùng video** (không phải greedy baseline cổ điển) | [L924-925](modules/modeling.py#L924-L925) |
| **Loss** | `mean( -sequences_scores · (r - b).detach() )` | [L929-930](modules/modeling.py#L929-L930) |

Thiết kế này bám sát bản tham chiếu LAVIS trong [our_blip2_t5.py:267-300](our_blip2_t5.py#L267-L300) (beam K, baseline mean-beam), khác ở: re-score teacher-forced thay vì dùng `sequences_scores` từ `generate`, multi-reference, corpus-IDF, diversity penalty.

⚠️ **Nghi vấn quan trọng — IDF của `CorpusCider` có thể bị sai lệch.** `CorpusCider.compute_score` gán `scorer.ref_len = log(N_corpus)` rồi gọi `compute_cider()` ([utils/cider_utils.py:84-88](utils/cider_utils.py#L84-L88)), nhưng trong pycocoevalcap `compute_cider()` **tự ghi đè** `self.ref_len = np.log(len(self.crefs))` ([tasks/pycocoevalcap/cider/cider_scorer.py:162](tasks/pycocoevalcap/cider/cider_scorer.py#L162) — bản vendored; bản pip `salaniz/pycocoevalcap` có cùng dòng theo hiểu biết của tôi, **chưa kiểm tra bản cài trên máy train**). Hệ quả: trọng số n-gram = `log(B·K) − log(df_corpus)` = `log(320) − log(df)`, với `df` đếm trên 6513 video → mọi n-gram xuất hiện ở >320 video train (ví dụ "a", "man", "is", "a man") nhận **trọng số âm** với độ lớn tăng theo độ phổ biến. Khi đó reward không còn là CIDEr đúng nghĩa và có thể **ưu ái n-gram phổ biến** — trùng khớp với hiện tượng caption khuôn mẫu ghi trong `run.sh`. Đây là suy luận từ code, cần kiểm chứng bằng thí nghiệm nhỏ. → ❓ Q5.

⚠️ Khác biệt train/eval trong decode: SCST sinh với `repetition_penalty=1.0` ([L841](modules/modeling.py#L841)), eval với `1.2` ([L948](modules/modeling.py#L948)). → ❓ Q10.

⚠️ Re-score chạy ở `model.train()` nên LoRA dropout (0.05) và dropout của T5/Q-Former đang bật → log-prob dùng cho gradient khác nhẹ so với policy đã sinh candidate. Không chắc là vấn đề, chỉ ghi nhận.

---

## 4. Dữ liệu

### 4.1 Dataset & split

- MSRVTT, annotation `MSRVTT_data.json` (10,000 video, 200,000 câu, **đúng 20 caption/video** — kiểm tra trên [evidents/MSRVTT_data.json](evidents/MSRVTT_data.json); trường `split` trong JSON đều là `"train"` và **bị code bỏ qua**).
- Split **cứng theo vị trí trong `data['videos']`** ([dataloaders/dataloader_msrvtt_caption.py:63-79](dataloaders/dataloader_msrvtt_caption.py#L63-L79)): train `[:6513]`, val `[6513:7010]`, test `[7010:]`. Trong JSON hiện có, index i ≙ `video{i}` → đây là **split chuẩn MSRVTT 10k (6513/497/2990)**.
- `--use_small_dataset`: chia theo số trong `video_id` và lọc video có feature ([L64-81](dataloaders/dataloader_msrvtt_caption.py#L64-L81)).
- Loader dùng cho eval là `"val"` của factory nhưng gọi `split_type="test"` mặc định ([data/dataloader_factory.py:96](data/dataloader_factory.py#L96), [main_task_caption.py:185](main_task_caption.py#L185)) → **test 2990 video** (log: `Num examples = 2990`).

⚠️ **`--train_csv` / `--val_csv` gần như không có tác dụng**: CSV chỉ được đọc để chuẩn hoá key feature và tìm `feature_size` ([L34](dataloaders/dataloader_msrvtt_caption.py#L34), [L39-57](dataloaders/dataloader_msrvtt_caption.py#L39-L57)); danh sách video **không** lấy từ CSV. README/FEATURE_NOTES nói train dùng `MSRVTT_train.9k.csv` (9k video) và test là `MSRVTT_JSFUSION_test.csv` — trong khi thực tế train **6513** video, test **2990** video. (Bản thân file JSFusion test thường là split 1k-A 1000 video; README lại ghi "2990 video" — mâu thuẫn nội tại.) Các file CSV/`MSRVTT_data.json` không nằm trong `data/msrvtt/` của repo nên tôi không kiểm tra được nội dung CSV. → ❓ Q1.

### 4.2 Mẫu huấn luyện

- Train (cả XE lẫn SCST): mỗi (video, caption) là một mẫu → **130,260 mẫu/epoch**, mỗi video xuất hiện 20 lần/epoch ([L86-99](dataloaders/dataloader_msrvtt_caption.py#L86-L99)). Hai nhánh `scst`/XE hiện **giống hệt nhau**.
- Test: 1 mẫu/video, caption đi kèm là caption GT đầu tiên (chỉ dùng cho val loss) ([L100-105](dataloaders/dataloader_msrvtt_caption.py#L100-L105)).
- ⚠️ Comment ở [L87-90](dataloaders/dataloader_msrvtt_caption.py#L87-L90) ("mỗi beam chấm với 1 GT") và [docs/traditional-pipelines.md](docs/traditional-pipelines.md) ("SCST: một sample trên mỗi video", batch 14 tensor) đã **lỗi thời** — code trả về 8 phần tử ([L233-234](dataloaders/dataloader_msrvtt_caption.py#L233-L234)) và SCST đã dùng multi-reference qua trainer.

### 4.3 Tiền xử lý text

- T5: `t5_tokenizer(caption, padding="max_length", truncation=True, max_length=max_txt_len)` → `max_txt_len=20` token T5 (gồm `</s>`) ([L166-176](dataloaders/dataloader_msrvtt_caption.py#L166-L176)).
- BERT (legacy): `input_ids` chỉ là `[CLS][SEP]` (không có transcript), `output_caption_ids` = caption tokenized BERT + `[SEP]`, dài `max_words=48` ([L127-164](dataloaders/dataloader_msrvtt_caption.py#L127-L164)); chỉ dùng để in `ref.txt` khi eval.

### 4.4 Lấy mẫu frame & feature

| Bước | Thực tế | Nguồn |
|---|---|---|
| Lấy frame | **1 fps**: đọc tuần tự bằng OpenCV, giữ frame mỗi `round(fps_video/1)` frame | [extract_clip_features.ipynb](extract_clip_features.ipynb) cell 11 |
| Encode | CLIP `preprocess` (resize, center-crop 224, normalize) → `encode_image` **từng frame độc lập** → 1 vector/frame | notebook cell 13 |
| Model trong notebook | **ViT-B/32 (dim 512)** mặc định | notebook cell 7 |
| Model thực sự dùng | `run.sh`: pickle **CLIP ViT-L/14 (dim 768)** từ dataset ngoài (`phihungcrr1701/msrvtt-clip-vitl14-features`); run `evidents/`: `nguyhao/blip2-feature-extractor/.../msrvtt_blip2_features.pickle` (dim 768) | [run.sh:50](run.sh#L50), [evidents/log.txt:21](evidents/log.txt) |
| Trong dataloader | Không resample: lấy `T` vector đầu, **cắt `[:max_frames]`** nếu `T>48`, zero-pad + mask nếu ngắn hơn | [L193-214](dataloaders/dataloader_msrvtt_caption.py#L193-L214) |
| S3D (legacy UniVL) | `VideoFeatureExtractor/` : 16 fps, clip 16 frame không chồng → 1 vector 1024-d / giây | [FEATURE_NOTES.md](FEATURE_NOTES.md), [VideoFeatureExtractor/](VideoFeatureExtractor/) |

- ⚠️ Script tạo pickle ViT-L/14 và BLIP-2 feature **không có trong repo** → không xác minh được cách lấy frame/pooling của chúng. Đặc biệt feature "BLIP-2" có dim 768 (không phải 1408 của EVA ViT-g) — có thể là output Q-Former/projection đã pool theo frame. → ❓ Q7.
- Ghi chú: video MSRVTT thường 10–30 giây, nên ở 1 fps việc cắt ở 48 frame **hiếm khi xảy ra**; Hạn chế #9 trong README có thể bị phóng đại (JSON hiện có không chứa thời lượng nên tôi chưa đếm được).

---

## 5. Đánh giá

`eval_epoch` ([inference/caption_generator.py:7-137](inference/caption_generator.py#L7-L137)):

1. `model.eval()`, mỗi batch (128): forward để lấy **val loss XE** (với 1 GT) và `visual_output` ([L44-50](inference/caption_generator.py#L44-L50)).
2. Sinh caption: beam `eval_beam_size=5`, `max_length=max_txt_len=20`, `repetition_penalty=1.2`, `length_penalty=1.0`, `do_sample=False` ([L54-59](inference/caption_generator.py#L54-L59), [modules/modeling.py:932-952](modules/modeling.py#L932-L952)).
3. Reference: lấy **toàn bộ 20 caption** của mỗi video từ `video_sentences_dict` ([L89-97](inference/caption_generator.py#L89-L97)); `zip(*...)` sẽ cắt về số ref nhỏ nhất (an toàn vì MSRVTT luôn 20).
4. PTBTokenizer cho cả hyp & ref ([L101-128](inference/caption_generator.py#L101-L128)) → `CaptionEvaluator` (pycocoevalcap): **BLEU-1..4, METEOR, ROUGE-L, CIDEr** ([metrics.py:24-63](metrics.py#L24-L63)). CIDEr ở đây là CIDEr-D chuẩn với IDF tính trên **test refs** (không dùng `CorpusCider`). Thang điểm 0–1 trong log tương ứng ×100 trong paper (0.6118 ≈ 61.2).
5. Ghi `hyp.txt` (prediction) và `ref.txt` (**chỉ 1 ref BERT-decoded / video**, không phải tập ref dùng để chấm) ([inference/eval_utils.py:23-42](inference/eval_utils.py#L23-L42)).
6. **Chọn checkpoint**: sau mỗi epoch, lưu `best.0` nếu CIDEr tăng (hoà thì so BLEU-4) ([main_task_caption.py:226-241](main_task_caption.py#L226-L241)); cuối cùng nạp lại `best` và eval ra `hyp_best.txt` ([L245-259](main_task_caption.py#L245-L259)).

⚠️ **Model selection trên chính test set** (2990) — không có tập held-out; val 497 video tồn tại trong split nhưng không được dùng. → ❓ Q2.

Công cụ chẩn đoán đa dạng caption offline (không cần GPU): `python -m utils.diversity_utils --hyp <hyp.txt>` (duplicate ratio, opener n-gram, distinct-n, self-BLEU) — [utils/diversity_utils.py](utils/diversity_utils.py), test ở [tests/test_diversity_utils.py](tests/test_diversity_utils.py).

### 5.1 Kết quả có thể kiểm chứng trong repo

| Run | Cấu hình | Epoch → CIDEr | Best | Nguồn |
|---|---|---|---|---|
| `evidents/` (2026-05-07) | SCST α=1.0, K=5, lr=2e-7, lr_qformer=lr_lora=1e-7, LoRA q,v r32/α64, BLIP-2 feature 768-d, max_txt_len=22, 6 epoch, init `ckpt_msrvtt_caption_1/best.0` | 0.6090 → 0.6117 → 0.6111 → 0.6113 → 0.6111 → 0.6118 | **CIDEr 0.6118**, B4 0.4332, METEOR 0.3095, ROUGE-L 0.6387; val loss 2.3255 → 2.3518 | [evidents/log.txt](evidents/log.txt) |

⚠️ Bảng kết quả `ver_2`…`ver_5` trong README **không còn kiểm chứng được** — các file `ckpts/ver_*/log*.txt` đang ở trạng thái *deleted* trong working tree (vẫn còn trong git HEAD). Kết quả `ver_7` chỉ có trong comment [run.sh:26-28](run.sh#L26-L28). Run `run.sh` hiện tại (α=0.5, diversity 0.3) chưa có log.

---

## 6. Ý tưởng / claim chính mà code thể hiện

Mỗi claim kèm mức bằng chứng **có trong repo** (không phải mức đúng/sai của ý tưởng).

| # | Claim / ý tưởng | Hiện thực | Bằng chứng trong repo |
|---|---|---|---|
| C1 | Dùng feature frame-level đóng băng (CLIP/BLIP-2) + Transformer thời gian UniVL làm "video encoder" rẻ, thay cho chạy ViT trên frame | `NormalizeVideo → VisualModel` | Kiến trúc; chưa có ablation |
| C2 | Q-Former BLIP-2 (pretrain ảnh) có thể làm cầu nối nén chuỗi frame thành 32 query cho LLM | `_get_cross_output`, `qformer_visual_proj` | Kiến trúc; lưu ý nạp chỉ một phần (mục 1.3) |
| C3 | Fine-tune hiệu quả tham số: freeze Flan-T5-XL, chỉ LoRA q,v + projection + Q-Former | mục 2 | Kiến trúc |
| C4 | SCST tối ưu trực tiếp CIDEr cải thiện so với XE | `_compute_scst_caption_loss` | Yếu: run `evidents/` chỉ +0.0028 CIDEr (0.6090→0.6118) ở LR 1e-7; không có số XE-only tương ứng trong log hiện có |
| C5 | SCST thuần (α=1) không ổn định; trộn XE (α<1) giúp ổn định | `scst_alpha` ([modules/modeling.py:782-789](modules/modeling.py#L782-L789)) | Chỉ có ghi chú `ver_7` trong `run.sh` (val loss 2.5→7.3); chưa có run α=0.5 |
| C6 | Multi-reference reward (20 GT) nhất quán với metric eval hơn single-reference | [trainers/trainer.py:34-40](trainers/trainer.py#L34-L40) | Lập luận trong comment; chưa có kết quả |
| C7 | IDF corpus-level ổn định hơn IDF theo batch cho reward | `CorpusCider` | ⚠️ Có thể không đạt mục tiêu do `ref_len` bị ghi đè (mục 3.4) |
| C8 | Phạt trùng lặp n-gram giữa các video trong batch chống mode-collapse về caption khuôn mẫu | `--scst_diversity_weight` | Chẩn đoán: comment `run.sh` nêu self-BLEU-4 0.732–0.755, ~1/3 caption trùng hệt; chưa có run có penalty |
| C9 | Regularizer đa dạng giữa các query Q-Former (giảm cosine) | `_compute_diversity_loss` | Có code, các run đã biết đều đặt weight 0.0 |
| C10 | Re-score teacher-forced để có gradient hợp lệ cho SCST | [modules/modeling.py:852-876](modules/modeling.py#L852-L876) | Fix lỗi thực tế trong [error.txt](error.txt) |
| C11 | Constant LR (sau warmup) phù hợp hơn linear decay cho pha RL | [utils/optimizer_utils.py:51-53](utils/optimizer_utils.py#L51-L53) | Lập luận trong comment |

---

## 7. Danh sách mâu thuẫn & rủi ro (tổng hợp)

| # | Vấn đề | Mức | Vị trí |
|---|---|---|---|
| R1 | Split thực tế 6513/2990 theo index JSON; CSV không quyết định split; tài liệu nói 9k/JSFusion | Cao (ảnh hưởng so sánh với paper) | [dataloader_msrvtt_caption.py:63-79](dataloaders/dataloader_msrvtt_caption.py#L63-L79) |
| R2 | Chọn checkpoint trên test | Cao | [main_task_caption.py:185](main_task_caption.py#L185), [L228-234](main_task_caption.py#L228-L234) |
| R3 | `CorpusCider` IDF lệch do `ref_len` bị ghi đè → reward SCST có thể sai | Cao (nếu xác nhận) | [utils/cider_utils.py:84-88](utils/cider_utils.py#L84-L88), [cider_scorer.py:162](tasks/pycocoevalcap/cider/cider_scorer.py#L162) |
| R4 | Q-Former BLIP-2 nạp thiếu 192 tensor (khả năng do lệch tên key HF vs LAVIS), lỗi im lặng | Cao cho run khởi tạo từ đầu; thấp nếu luôn có `--init_model` đã train | [modules/blip2.py:134-161](modules/blip2.py#L134-L161), log L146 |
| R5 | `VisualModel` frozen: nếu checkpoint XE gốc cũng frozen thì VisualModel = **random init** suốt quá trình | Chưa rõ | [modules/modeling.py:377-383](modules/modeling.py#L377-L383) |
| R6 | LoRA target: README `q,k,v,o` vs code `q,v` | Trung bình | mục 3.1 |
| R7 | `repetition_penalty` khác nhau giữa SCST (1.0) và eval (1.2) | Thấp–TB | [modeling.py:841](modules/modeling.py#L841), [L948](modules/modeling.py#L948) |
| R8 | `normalize_video` học với `lr=3e-5` ≫ các nhóm khác | Thấp–TB | [optimizer_utils.py:44,48](utils/optimizer_utils.py#L44) |
| R9 | Tài liệu lỗi thời: README (CrossModel, LoRA q,k,v,o, prompt viết thường), `docs/traditional-pipelines.md` (14-tensor batch, SCST 1 sample/video), `docs/siglip-stc-pipelines.md` (nhánh không tồn tại), comment dataloader L87-90 | Thấp | — |
| R10 | `requirements.txt` thiếu `transformers, peft, pycocoevalcap, huggingface_hub, safetensors, omegaconf, pyyaml` (được import trong code) | Thấp | [requirements.txt](requirements.txt) |
| R11 | `--freeze_vit` đặt tên gây hiểu nhầm (freeze Transformer thời gian, không phải ViT) | Thấp | — |
| R12 | Diversity penalty dùng unigram bao gồm stopword ("a", "is") → gần như mọi caption có penalty >0; chỉ chênh lệch giữa các beam mới ảnh hưởng advantage | Ghi nhận | [diversity_utils.py:42-82](utils/diversity_utils.py#L42-L82) |

---

## 8. Câu hỏi cần bạn xác nhận

Tôi **không tự đoán** các điểm sau; câu trả lời sẽ quyết định cách diễn giải kết quả:

1. **Q1 – Split:** Bạn muốn train trên 6513 video (split chuẩn 10k, như code đang làm) hay 9k video theo `MSRVTT_train.9k.csv` như README? Test bạn báo cáo là 2990 video (code) hay JSFusion 1k?
2. **Q2 – Val/test:** Việc dùng test 2990 để chọn `best.0` là cố ý (theo protocol UniVL gốc) hay nên chuyển sang val 497?
3. **Q3 – Q-Former:** Bạn có biết dòng `tensors=144 … missing=192` không? Checkpoint XE đầu tiên (gốc của chuỗi `ver_1`/`ckpt_msrvtt_caption_1`) được khởi tạo từ BLIP-2 nạp một phần như vậy, hay từ nguồn khác (LAVIS `.pth`)?
4. **Q4 – VisualModel:** Ở pha XE ban đầu có bật `--freeze_vit` không? Nếu có, `visual.*` trong checkpoint có phải là random init không? Trọng số `visual.*` ban đầu đến từ đâu (UniVL pretrain HowTo100M)?
5. **Q5 – CorpusCider:** Phiên bản `pycocoevalcap` trên máy train là gì (pip `salaniz` hay vendored `tasks/pycocoevalcap`)? Nếu cùng dòng `self.ref_len = np.log(len(self.crefs))`, bạn có muốn tôi lên kế hoạch thí nghiệm nhỏ (không train) để so reward CorpusCider vs CIDEr chuẩn trên `evidents/hyp.txt`?
6. **Q6 – LoRA:** Checkpoint `ver_1` dùng target `q,v` hay `q,k,v,o`? (Ảnh hưởng nhánh merge LoRA khi nạp.)
7. **Q7 – Feature:** Pickle CLIP ViT-L/14 và BLIP-2 768-d được trích bằng script nào (fps, 1 vector/frame hay nhiều token/frame, có pooling không)? Có nên thêm script đó vào repo?
8. **Q8 – Stage XE:** Lệnh/hyper-param của pha XE tạo ra `ckpt_msrvtt_caption_ver_1` là gì (epoch, LR, feature)? Kết quả XE-only (CIDEr) là bao nhiêu?
9. **Q9 – Kết quả README:** Các số `ver_2…ver_5` trong README lấy từ log nay đã bị xoá khỏi working tree — bạn có muốn khôi phục `ckpts/ver_*/log*.txt` (vẫn còn trong git HEAD) để đối chiếu không?
10. **Q10 – Decode:** `repetition_penalty` 1.0 (SCST) vs 1.2 (eval) có chủ ý không?
11. **Q11 – LR:** `normalize_video` (và mọi tham số "other") dùng `--lr 3e-5` — có chủ ý không?

---

## 9. Bản đồ file

```text
run.sh                          lệnh train SCST hiện tại + ghi chú các run trước
main_task_caption.py            entry: args, dataloader, optimizer, vòng train/eval, chọn best
modules/modeling.py             UniVL: VisualModel → Q-Former → T5(LoRA); XE / SCST / diversity
modules/blip2.py, Qformer.py    khởi tạo & nạp Q-Former BLIP-2 (LAVIS-style)
modules/module_visual.py        Transformer thời gian UniVL
modules/optimization.py         BertAdam + schedule warmup_linear/constant
trainers/trainer.py             train_epoch, ghép gt_refs đa tham chiếu cho SCST
inference/caption_generator.py  eval_epoch: val loss, beam search, metric
metrics.py                      wrapper pycocoevalcap
utils/cider_utils.py            CorpusCider (IDF corpus cho reward)
utils/diversity_utils.py        overlap penalty + chẩn đoán mode-collapse (mới, chưa commit)
utils/optimizer_utils.py        nhóm tham số & LR
dataloaders/dataloader_msrvtt_caption.py  split, mẫu, tokenize T5/BERT, pad/truncate feature
extract_clip_features.ipynb     trích CLIP 1 fps (ViT-B/32 mặc định)
VideoFeatureExtractor/          trích S3D (legacy UniVL)
our_blip2_t5.py                 bản tham chiếu LAVIS Blip2T5 + SCST (không dùng trong runtime)
evidents/                       log + hyp/ref của 1 run SCST (bằng chứng duy nhất còn trong working tree)
Legacy/không đồng bộ: main_pretrain.py, main_task_caption_test.py, main_task_caption_no_visual.py,
                      univl.ipynb, test_t5_decoder.py, dataloaders/*youcook*, *retrieval*, *howto100m*
```
