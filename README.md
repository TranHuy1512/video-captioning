# Video Captioning: UniVL + BLIP-2 QFormer + Flan-T5 (LoRA + SCST)

Project video captioning cho **MSRVTT** (và YoucookII ở nhánh legacy), xuất phát từ [UniVL](https://arxiv.org/abs/2002.06353) nhưng nhánh caption hiện tại đã được chỉnh lại đáng kể: video feature (trích sẵn ngoài runtime) đi qua `VisualModel` (Transformer 6 lớp của UniVL) rồi vào **BLIP-2 QFormer** để tạo query embedding, cuối cùng được đưa vào **Flan-T5** (fine-tune bằng LoRA) để sinh caption. Có hỗ trợ huấn luyện thêm bằng **SCST** (self-critical, tối ưu trực tiếp CIDEr).

> README gốc mô tả UniVL nguyên bản đã được thay bằng tài liệu này để phản ánh đúng pipeline đang chạy. Chi tiết luồng dữ liệu/shape từng bước xem [docs/traditional-pipelines.md](docs/traditional-pipelines.md) (khớp với code hiện tại) — không dùng [docs/siglip-stc-pipelines.md](docs/siglip-stc-pipelines.md) làm tham chiếu, vì nhánh `--direct_qformer_input` mô tả ở đó **không tồn tại** trong `modules/modeling.py`/`main_task_caption.py` hiện tại (xem mục Hạn chế).

![UniVL framework](assets/imgs/UniVL_framework.jpg)

## 1. Kiến trúc pipeline hiện tại

```text
Video gốc
  → trích feature offline (CLIP ViT-L/14, hoặc BLIP-2 vision encoder, ...) → pickle {video_id: (T, D)}
  → MSRVTT_Caption_DataLoader (pad/truncate về max_frames, cắt phần đầu nếu T > max_frames)
  → UniVL.forward()
       NormalizeVideo (LayerNorm) → VisualModel (Transformer 6 lớp, freeze_vit tuỳ chọn)
       → QFormer (query_tokens học cross-attention lên visual_output; qformer_visual_proj nếu D lệch qformer_vision_width)
       → t5_proj → nối với prompt "a video of"
       → Flan-T5 (base freeze, LoRA train trên q/k/v/o)
  → Loss = XE (teacher forcing) hoặc XE + SCST (nếu --scst)
  → Inference: beam search → caption text
  → pycocoevalcap: BLEU-1..4, METEOR, ROUGE-L, CIDEr
```

Entry point: `run.sh` → `main_task_caption.py` → `data/dataloader_factory.py` → `dataloaders/dataloader_msrvtt_caption.py` → `modules/modeling.py::UniVL` → `trainers/trainer.py` → `inference/caption_generator.py` → `metrics.py`.

Chi tiết đầy đủ (shape từng bước, optimizer, checkpoint prefix...) xem [docs/traditional-pipelines.md](docs/traditional-pipelines.md).

## 2. Cài đặt

`requirements.txt` hiện chỉ liệt kê `torch, tqdm, boto3, requests, pandas` — **chưa đủ** để chạy caption pipeline. Cần cài thêm:

```bash
pip install torch tqdm boto3 requests pandas
pip install transformers accelerate sentencepiece peft
pip install git+https://github.com/salaniz/pycocoevalcap.git   # BLEU/METEOR/ROUGE/CIDEr
```

Yêu cầu Java (>=1.8) cho METEOR. Model đã được chạy thực tế trên Python 3.12 + PyTorch mới (khác với `python==3.6.9 / torch==1.7.0` ghi trong hướng dẫn UniVL gốc — thông tin đó đã lỗi thời với nhánh caption hiện tại).

## 3. Dữ liệu & feature

- Annotation: `data/msrvtt/MSRVTT_data.json`, split: `data/msrvtt/MSRVTT_train.9k.csv`, `data/msrvtt/MSRVTT_JSFUSION_test.csv` (2990 video, dùng làm cả **val lẫn test** — xem Hạn chế #1).
- `--features_path` phải là **một file pickle** dạng `dict[video_id] -> np.ndarray(T, D)`, `D` khớp `--video_dim`. Dataloader chính không đọc thư mục feature rời `.pt/.pkl` (khác mô tả trong `docs/siglip-stc-pipelines.md`).
- `VideoFeatureExtractor/` trích feature S3D từ video raw (pipeline gốc của UniVL). Các lần train gần đây trong `ckpts/` dùng feature CLIP ViT-L/14 hoặc feature từ một BLIP-2 vision extractor riêng — xem so sánh S3D vs CLIP trong [FEATURE_NOTES.md](FEATURE_NOTES.md).

## 4. Huấn luyện & suy luận

Lệnh mẫu nằm trong `run.sh` (đã comment sẵn một cấu hình từng gây sụp CIDEr, giữ lại làm tài liệu). Các nhóm tham số chính (`main_task_caption.py`):

| Nhóm | Tham số | Ghi chú |
|---|---|---|
| Video/text | `--max_frames`, `--max_words`, `--video_dim` | `video_dim` phải khớp chiều cuối của feature pickle |
| QFormer | `--num_query_token`, `--qformer_vision_width`, `--qformer_checkpoint`, `--qformer_diversity_weight` | checkpoint mặc định `Salesforce/blip2-opt-6.7b-coco` |
| LoRA | `--lora`, `--lora_r`, `--lora_alpha`, `--lora_target_modules` | Flan-T5 base bị freeze, chỉ LoRA + `t5_proj` + QFormer train |
| Learning rate | `--lr`, `--lr_qformer`, `--lr_lora` | mỗi nhóm module có LR riêng, xem `utils/optimizer_utils.py` |
| SCST | `--scst`, `--scst_alpha`, `--scst_num_samples`, `--eval_beam_size` | `scst_alpha` mặc định `1.0` (SCST thuần, xem Hạn chế #2) |

Đánh giá dùng `pycocoevalcap` (BLEU-1..4, METEOR, ROUGE-L, CIDEr), kết quả ghi vào `output_dir/hyp.txt`, `ref.txt`, và checkpoint tốt nhất theo CIDEr (tie-break bằng BLEU-4) lưu ở `pytorch_model.bin.best.0`.

## 5. Kết quả / Metrics hiện tại

Trên **MSRVTT test split (JSFUSION, 2990 video)**, tổng hợp từ log thực tế trong `ckpts/`:

| Run | Cấu hình nổi bật | CIDEr (best) | BLEU-4 | METEOR | ROUGE-L |
|---|---|---|---|---|---|
| UniVL gốc (baseline, paper/README cũ) | không SCST, decoder Transformer legacy | 0.5004 | 0.4179 | 0.2894 | 0.6078 |
| `ver_2` | SCST thuần (`alpha=1.0`), `lr_lora=2e-4` (cao) | **0.0000** (sập) | 0.0000 | 0.0587 | 0.2722 |
| `ver_3` | tiếp tục từ checkpoint tốt, LR thấp hơn (`lr_lora=3e-6`) | 0.5826 | 0.4145 | 0.3020 | 0.6308 |
| `ver_5` | đổi sang feature BLIP-2 vision, LR rất thấp (~2-3e-6) | **0.5943** | 0.4210 | 0.3069 | 0.6284 |
| `ver_7` (theo ghi chú trong `run.sh`) | SCST thuần kéo dài nhiều epoch | 0.5917 → 0.5569 → 0.5390 (giảm dần) | — | — | — |

Best hiện tại: **CIDEr ≈ 0.594, BLEU-4 ≈ 0.421, METEOR ≈ 0.307, ROUGE-L ≈ 0.628** (`ckpts/ver_5`), cao hơn baseline UniVL gốc về CIDEr nhưng cần đọc cùng phần Hạn chế bên dưới vì số liệu này chưa hoàn toàn đáng tin cậy để so sánh trực tiếp.

## 6. Hạn chế (Limitations)

1. **Không có tập test độc lập với việc chọn checkpoint.** `MSRVTT_JSFUSION_test.csv` được dùng làm "val" ngay trong vòng lặp train (chọn `best.0` theo CIDEr trên chính split này) và cũng là split duy nhất để báo cáo số liệu cuối. Không có tập held-out thứ ba → số liệu CIDEr/BLEU ở trên có nguy cơ lạc quan hơn thực tế (implicit overfitting vào test set qua model selection).
2. **SCST dễ sập (catastrophic collapse) khi chạy thuần RL.** `scst_alpha` mặc định là `1.0` (100% SCST, không XE). Log `ckpts/ver_2` cho thấy khi kết hợp LR cao, CIDEr rơi thẳng về `0.0000` (BLEU-3/4 cũng về 0 — caption suy biến/lặp). Ghi chú trong `run.sh` (`ver_7`) mô tả cùng hiện tượng ở tốc độ chậm hơn: CIDEr giảm dần qua các epoch (0.59 → 0.55 → 0.54) trong khi validation loss tăng vọt (2.5 → 7.3). Bản thân `ver_5` — kết quả tốt nhất hiện có — cũng chạy với `alpha=1.0` (do CLI `--scst_alpha` chưa tồn tại tại thời điểm đó) và đã bắt đầu thấy CIDEr giảm nhẹ ở epoch 2 (0.5943 → 0.5903); checkpoint tốt chỉ "may mắn" được lưu trước khi suy giảm tiếp diễn.
3. **Reward CIDEr không nhất quán giữa train và eval.** Khi không truyền đủ `gt_refs` đa tham chiếu, SCST tính CIDEr so với **một** caption ground-truth mỗi sample, trong khi evaluation cuối cùng gộp toàn bộ (thường ~20) caption/video làm multi-reference. Model được tối ưu theo một proxy reward khác với chính metric dùng để báo cáo.
4. **Candidate cho SCST lấy từ beam search xác định (`do_sample=False`), không phải sampling.** Độ đa dạng giữa các beam thấp → nhiều trường hợp advantage (`reward − mean_beam_reward`) gần 0, tín hiệu gradient yếu; `--qformer_diversity_weight` chỉ tác động gián tiếp lên embedding của QFormer, không trực tiếp ép caption sinh ra khác nhau.
5. **Baseline reward là trung bình reward của các beam cùng sample**, không phải baseline greedy cổ điển của SCST — dễ bị lệch khi các beam gần giống hệt nhau (baseline ≈ reward → model gần như không học được) hoặc khi có một beam ngoại lệ kéo baseline lệch hẳn.
6. **Kiến trúc lai còn cồng kềnh và có phần thừa.** Model luôn khởi tạo cả `BertModel` + `CrossModel` (module legacy của UniVL) dù nhánh caption hiện tại chạy `_need_text_encoder=False` và caption generation chỉ thực sự dùng `VisualModel` → QFormer → Flan-T5. Tốn bộ nhớ khởi tạo và dễ gây nhầm lẫn khi đọc code.
7. **Base Flan-T5-XL bị freeze hoàn toàn**, phần học được chỉ là LoRA (r=32) trên `q,k,v,o` + `t5_proj` + QFormer. Nếu bottleneck nằm ở khả năng suy luận ngôn ngữ của T5 (chứ không phải ở phía visual), cấu hình này khó cải thiện thêm.
8. **Feature video phụ thuộc extractor không có temporal.** Cấu hình `run.sh` hiện dùng CLIP ViT-L/14 (1 khung hình/giây, mỗi khung xử lý độc lập, không có temporal) hoặc feature BLIP-2 tương tự; toàn bộ thông tin chuyển động phải được bù bởi self-attention 6 lớp của `VisualModel`. Có thể yếu với caption mô tả hành động phức tạp/chuyển động nhanh (xem so sánh với S3D trong `FEATURE_NOTES.md`).
9. **Dataloader cắt phần đầu khi video dài hơn `max_frames`** (`video_slice[:max_frames]`) thay vì pooling/chia đều toàn chuỗi → mất thông tin ở phần cuối video đối với các video MSRVTT dài hơn 48 giây.
10. **Tài liệu trong repo chưa đồng bộ với code.** `docs/siglip-stc-pipelines.md` mô tả một nhánh `--direct_qformer_input` / `TemporalAdapter` / `NormalizeVideoDirect` **không tồn tại** trong `modules/modeling.py` và `main_task_caption.py` hiện tại; `SCST_TRAINING_SUMMARY.md` mô tả cách một project tham khảo khác (`mllm-video-captioner`, dựa trên LAVIS) implement SCST, không phải chính xác code trong `modules/modeling.py` (dù thiết kế thực tế khá tương đồng — beam-based, baseline = mean-beam reward). Người đọc dễ nhầm các tài liệu này là mô tả pipeline đang chạy.
11. **Môi trường/test chưa production-ready.** `requirements.txt` thiếu `transformers/peft/pycocoevalcap`; `tests/test_small_features_load.py` phụ thuộc đường dẫn tuyệt đối trên máy cá nhân (`/Users/mac/Documents/NCKH/...`) qua biến môi trường mặc định, dễ bị skip âm thầm ở máy khác. Từng có lỗi runtime thực tế (`error.txt`: `element 0 of tensors does not require grad`) khi lấy điểm số trực tiếp từ `generate()` — đã fix bằng cách chấm điểm lại qua forward pass teacher-forcing (`_compute_scst_caption_loss`), nhưng cho thấy nhánh SCST khá mong manh trước các thay đổi code.
12. **Nhiều script/notebook thử nghiệm không được bảo trì đồng bộ** (`main_task_caption_test.py`, `main_task_caption_no_visual.py`, `univl.ipynb`, `extract_clip_features.ipynb`...), dễ lỗi thời và khó dùng lại để reproduce kết quả trong `ckpts/`.

## 7. Định hướng cải thiện performance

Ưu tiên theo mức độ ảnh hưởng, bám theo các hạn chế ở trên:

- **Tách val/test thật sự**: trích một phần từ `MSRVTT_train.9k.csv` làm val để chọn checkpoint, chỉ chạy trên `MSRVTT_JSFUSION_test.csv` một lần cuối để báo cáo — tránh model selection leak vào test set (hạn chế #1).
- **Ổn định hoá SCST**: luôn dùng `--scst_alpha` trong khoảng 0.3–0.7 (không để mặc định 1.0), theo dõi validation loss/CIDEr sau mỗi vài trăm step để dừng sớm trước khi "nổ" (hạn chế #2), cân nhắc thêm gradient clipping riêng cho pha SCST.
- **Đồng bộ reward train/eval**: luôn truyền đủ `gt_refs` đa tham chiếu (đã có sẵn qua `video_sentences_dict` trong `trainers/trainer.py`) cho `_compute_scst_caption_loss`, tránh single-reference reward (hạn chế #3).
- **Tăng đa dạng candidate cho SCST**: thử sampling (top-p/top-k, temperature > 0) cho một phần candidate thay vì thuần beam search, hoặc tăng `--scst_num_samples`; cân nhắc baseline kiểu greedy-decode cổ điển để so sánh với baseline mean-beam hiện tại (hạn chế #4, #5).
- **Tinh gọn kiến trúc**: thêm cờ tắt hẳn `BertModel`/`CrossModel` khi chạy caption-only + QFormer để giảm chi phí khởi tạo/bộ nhớ (hạn chế #6).
- **Mở rộng phần trainable của Flan-T5**: thử unfreeze thêm vài layer cuối, hoặc mở rộng `lora_target_modules` (thêm `wi`, `wo`) nếu ngân sách GPU cho phép, để kiểm tra xem bottleneck có nằm ở khả năng ngôn ngữ của T5 hay không (hạn chế #7).
- **Thử lại feature có temporal (S3D) hoặc kết hợp CLIP+S3D** theo phân tích trong `FEATURE_NOTES.md`, đặc biệt cho các caption cần mô tả hành động/chuyển động (hạn chế #8).
- **Đổi truncate sang pooling theo chunk** (chia `max_frames` đoạn rồi lấy mean, như logic đã có ở nhánh `youcook`/dataloader cũ) thay vì cắt phần đầu, để giữ thông tin toàn bộ video dài (hạn chế #9).
- **Đồng bộ hoá tài liệu với code**: cập nhật hoặc xoá `docs/siglip-stc-pipelines.md` nếu nhánh `direct_qformer_input` không còn trong roadmap; ghi rõ ở đầu `SCST_TRAINING_SUMMARY.md` rằng đó là tài liệu tham khảo từ project khác (hạn chế #10).
- **Củng cố hạ tầng thử nghiệm**: hoàn thiện `requirements.txt` (pin version cụ thể), bỏ path tuyệt đối cứng trong test, thêm một smoke test chạy forward 1 batch nhỏ trước khi train full để bắt sớm lỗi kiểu `error.txt` (hạn chế #11), và gom kết quả các lần train (`ver_2`..`ver_7`...) vào một bảng theo dõi thí nghiệm duy nhất (CSV/W&B) thay vì rải rác nhiều thư mục `ckpts/` (hạn chế #12).

## 8. Cấu trúc thư mục chính

```text
main_task_caption.py        # entrypoint train/eval caption (pipeline đang dùng)
main_pretrain.py            # pretrain UniVL gốc trên HowTo100M (legacy)
modules/modeling.py         # UniVL, VisualModel, QFormer + Flan-T5 + LoRA + SCST
trainers/trainer.py         # train_epoch, lấy gt_refs đa tham chiếu cho SCST
inference/caption_generator.py  # eval_epoch, sinh caption bằng beam search
dataloaders/                # MSRVTT_Caption_DataLoader, Youcook_Caption_DataLoader
VideoFeatureExtractor/      # trích feature S3D từ video raw
metrics.py                  # wrapper pycocoevalcap (BLEU/METEOR/ROUGE/CIDEr)
docs/traditional-pipelines.md   # mô tả pipeline hiện tại (khớp code)
docs/siglip-stc-pipelines.md    # tài liệu nhánh SigLIP+STC — hiện KHÔNG có trong code
FEATURE_NOTES.md            # so sánh feature S3D vs CLIP
SCST_TRAINING_SUMMARY.md    # tham khảo cách một project khác (LAVIS) implement SCST
ckpts/                      # log & kết quả các lần train (ver_2..ver_5)
```

## 9. Citation & License

Dựa trên [UniVL: A Unified Video and Language Pre-Training Model](https://arxiv.org/abs/2002.06353):

```
@Article{Luo2020UniVL,
  author  = {Huaishao Luo and Lei Ji and Botian Shi and Haoyang Huang and Nan Duan and Tianrui Li and Jason Li and Taroon Bharti and Ming Zhou},
  title   = {UniVL: A Unified Video and Language Pre-Training Model for Multimodal Understanding and Generation},
  journal = {arXiv preprint arXiv:2002.06353},
  year    = {2020},
}
```

Giấy phép: xem [LICENSE](LICENSE) và [NOTICE.md](NOTICE.md). Code dựa trên [pytorch-transformers v0.4.0](https://github.com/huggingface/transformers/tree/v0.4.0), [howto100m](https://github.com/antoine77340/howto100m), và tích hợp thêm [BLIP-2 QFormer](https://github.com/salesforce/LAVIS) cùng [PEFT/LoRA](https://github.com/huggingface/peft).
