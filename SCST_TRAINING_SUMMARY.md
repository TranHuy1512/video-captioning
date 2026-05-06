# Training Với SCST Loss Trong Project

Tài liệu này tóm tắt cách project `mllm-video-captioner` huấn luyện video captioning bằng SCST/self-critical style loss. Các file chính:

- `lavis/models/blip2_models/blip2_t5.py`: implementation SCST cho BLIP-2 + FLAN-T5.
- `lavis/models/blip2_models/blip2_opt.py`: implementation SCST cho BLIP-2 + OPT.
- `lavis/datasets/datasets/video_caption_datasets.py`: dataset trả video tensor và caption ground-truth.
- `lavis/projects/blip2/train/*_scst.yaml`: config bật SCST.
- `lavis/tasks/pycocoevalcap/cider/*`: implementation CIDEr reward.
- `lavis/tasks/captioning.py`: validation/evaluation captioning.

## Tổng Quan Phương Pháp

Project dùng BLIP-2 để biến video thành input cho LLM decoder. Video có shape `B, C, T, H, W`, sau đó được đổi thành `B*T, C, H, W` để đưa từng frame qua vision encoder. Feature của các frame được reshape lại thành một chuỗi visual token `B, T*h*w, D`, rồi Q-Former tạo query representation và linear projection đưa sang hidden size của T5/OPT.

Có hai chế độ training trong `forward()`:

- Cross-entropy training khi `self.scst == False`.
- SCST training khi `self.scst == True`.

SCST trong project tối ưu trực tiếp reward CIDEr của caption được generate. Với mỗi video trong batch, model generate `beam_size` caption bằng beam search, tính CIDEr cho từng caption so với ground-truth caption, rồi dùng advantage `reward - reward_baseline` để weight log-prob/sequence score.

Lưu ý quan trọng: implementation này không dùng greedy caption làm baseline như SCST cổ điển. Baseline ở đây là trung bình reward của các beam trong cùng sample.

## Cách Bật SCST

SCST được bật trong YAML bằng:

```yaml
model:
  scst: True
```

Ví dụ FLAN-T5 MSRVTT:

```yaml
model:
  arch: blip2_t5
  model_type: pretrain_flant5xl
  pretrained: ".../checkpoint_best.pth"
  prompt: "a video of"
  scst: True

run:
  task: captioning
  max_len: 32
  min_len: 5
  num_beams: 5
  batch_size_train: 2
```

Các script trong README:

```bash
python -m torch.distributed.run --nproc_per_node=8 train.py --cfg-path lavis/projects/blip2/train/caption_msrvtt_flant5xl_scst.yaml
python -m torch.distributed.run --nproc_per_node=8 train.py --cfg-path lavis/projects/blip2/train/caption_msvd_flant5xl_scst.yaml
```

Trong `from_config()`, config `scst` được đọc và truyền vào model. Với T5, `beam_size` mặc định trong class là `5`; YAML có `run.num_beams: 5` cho evaluation, còn nhánh SCST trong model dùng `self.beam_size`.

## Dữ Liệu Ground-Truth Caption

`VideoCaptionDataset.__getitem__()` đọc annotation:

```python
ann = self.annotation[index]
video = self.vis_processor(video_path)
caption = self.text_processor(ann["caption"])
return {
    "image": video,
    "text_input": caption,
    "image_id": self.img_ids[ann["image_id"]],
}
```

Caption ground-truth đi vào model qua `samples["text_input"]`. Text processor `blip_caption` chuẩn hóa caption bằng cách lowercase, xóa một số dấu câu, gom khoảng trắng, trim newline/space và truncate tối đa `max_words` mặc định 50.

Điểm cần chú ý: trong training SCST, mỗi sample dùng đúng caption đang có ở dòng annotation đó làm reference. Nếu dataset có nhiều caption cho cùng `image_id`, code training hiện tại không gom tất cả caption của cùng video để làm multi-reference. Multi-reference chỉ được gom trong evaluation ở `caption_eval()`, nơi annotations được group theo `image_id`.

## Luồng SCST Forward Với BLIP-2 T5

Nhánh SCST của `Blip2T5.forward()` làm các bước sau:

1. Nhận `samples["image"]` và lấy batch size `B`.
2. Chuyển video từ `B, C, T, H, W` sang `B*T, C, H, W`.
3. Chạy vision encoder + layer norm.
4. Reshape frame features thành một chuỗi visual feature cho mỗi video.
5. Chạy Q-Former với query tokens.
6. Project Q-Former output sang hidden size của T5.
7. Dùng prompt làm encoder text input:

```python
text = samples["text_input"]
samples["text_input"] = [self.prompt] * B
samples["text_output"] = text
```

8. Gọi `t5_model.generate()` với beam search:

```python
outputs = self.t5_model.generate(
    inputs_embeds=inputs_embeds,
    attention_mask=encoder_atts,
    do_sample=False,
    top_p=0.9,
    temperature=1,
    num_beams=self.beam_size,
    max_length=32,
    repetition_penalty=1.0,
    length_penalty=1.0,
    num_return_sequences=self.beam_size,
    return_dict_in_generate=True,
    output_scores=True,
)
```

Vì `do_sample=False`, đây là deterministic beam search, không phải sampling-based policy rollout.

## Luồng SCST Forward Với BLIP-2 OPT

`Blip2OPT.forward()` có cùng logic reward/loss, nhưng khác decoder:

- OPT là causal LM.
- SCST generate trực tiếp từ `inputs_opt` và `atts_opt`.
- Prompt code trong nhánh SCST OPT đang bị comment, nên generation SCST OPT không nối prompt text vào input như T5.
- EOS token là newline `"\n"`.

Phần reward và loss giống T5:

```python
caps_gen = self.opt_tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
caps_gen = [text.strip() for text in caps_gen]
caps_gt = list(itertools.chain(*([c, ] * self.beam_size for c in samples["text_input"])))
caps_gt = [[c] for c in caps_gt]
caps_gen, caps_gt = tokenize(caps_gt, caps_gen)
reward = Cider().compute_score(caps_gt, caps_gen)[1].astype(np.float32)
```

## Cách Tạo Generated Caption Và Ground-Truth Để So Sánh

Sau generation, output có tổng cộng `B * beam_size` sequence. Ví dụ `B=2`, `beam_size=5` thì có 10 generated captions.

Generated captions:

```python
caps_gen = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
caps_gen = [text.strip() for text in caps_gen]
```

Ground-truth captions được duplicate theo beam:

```python
caps_gt = list(itertools.chain(*([c, ] * self.beam_size for c in samples["text_output"])))
caps_gt = [[c] for c in caps_gt]
```

Với T5, `samples["text_output"]` là caption gốc trước khi `samples["text_input"]` bị thay bằng prompt. Với OPT, code dùng trực tiếp `samples["text_input"]`.

Ví dụ:

```text
B = 2, beam_size = 3
GT video 0 = "a man is playing guitar"
GT video 1 = "a dog runs on grass"

caps_gt =
[
  ["a man is playing guitar"],
  ["a man is playing guitar"],
  ["a man is playing guitar"],
  ["a dog runs on grass"],
  ["a dog runs on grass"],
  ["a dog runs on grass"],
]

caps_gen =
[
  gen_0_beam_0,
  gen_0_beam_1,
  gen_0_beam_2,
  gen_1_beam_0,
  gen_1_beam_1,
  gen_1_beam_2,
]
```

Sau đó `tokenize(caps_gt, caps_gen)` chuyển sang format pycocoevalcap:

```python
refs = {idx: [{"caption": r} for r in c_refs] for idx, c_refs in enumerate(refs)}
cands = {idx: [{"caption": c}] for idx, c in enumerate(cands)}
refs = PTBTokenizer().tokenize(refs)
cands = PTBTokenizer().tokenize(cands)
```

Nghĩa là mỗi generated caption được so với một list reference caption. Trong training hiện tại list này chỉ có một caption.

## Cách Tính Reward CIDEr

Reward là per-caption CIDEr score:

```python
reward = Cider().compute_score(caps_gt, caps_gen)[1]
```

`Cider().compute_score()` trả về:

- `score`: mean CIDEr trên toàn batch candidate.
- `scores`: mảng CIDEr riêng cho từng candidate.

SCST dùng `scores`, không dùng mean `score`.

CIDEr implementation:

1. Tokenized caption được split thành word.
2. Tạo n-gram từ 1 đến 4.
3. Với mỗi n-gram, tính term frequency.
4. Tính document frequency từ reference captions trong batch reward hiện tại.
5. Tạo vector TF-IDF cho candidate và reference:

```python
vec[n][ngram] = term_freq * (ref_len - log(document_frequency[ngram]))
```

6. Tính cosine similarity giữa vector candidate và từng reference.
7. Áp dụng Gaussian length penalty:

```python
penalty = exp(-((length_hyp - length_ref) ** 2) / (2 * sigma ** 2))
```

với `sigma = 6.0`.

8. Lấy mean qua 1-gram đến 4-gram, chia cho số reference, rồi nhân `10.0`.

Vì training đang dùng một reference cho mỗi generated caption, CIDEr score chủ yếu đo overlap TF-IDF n-gram giữa generated caption và caption ground-truth đơn lẻ, có penalty nếu độ dài lệch nhiều.

## Cách Tính Sequence Score

Sau generation, model lấy transition score của từng token:

```python
transition_scores = model.compute_transition_scores(
    outputs.sequences,
    outputs.scores,
    outputs.beam_indices,
    normalize_logits=False,
)
```

Sau đó tính độ dài output bằng số transition score âm:

```python
output_length = torch.sum(transition_scores < 0, dim=1)
```

Sequence score là tổng transition score chia cho độ dài:

```python
sequences_scores = transition_scores.sum(dim=1) / (output_length ** 1.0)
sequences_scores = sequences_scores.view(B, self.beam_size)
```

Về mặt ý nghĩa, `sequences_scores` đóng vai trò log-prob/score của caption được generate. Caption có score cao hơn sẽ được reinforce hoặc suppress tùy advantage.

## Công Thức Loss

Reward được reshape thành `B, beam_size`:

```python
reward = torch.from_numpy(reward).to(image.device).view(B, self.beam_size)
```

Baseline là mean reward của các beam trong cùng sample:

```python
reward_baseline = torch.mean(reward, -1, keepdim=True)
```

Advantage:

```python
advantage = reward - reward_baseline
```

Loss:

```python
loss = -sequences_scores * (reward - reward_baseline)
loss = loss.mean()
```

Diễn giải:

- Nếu caption beam có reward cao hơn trung bình các beam của cùng video, `advantage > 0`; loss khuyến khích tăng score của caption đó.
- Nếu caption beam có reward thấp hơn trung bình, `advantage < 0`; loss khuyến khích giảm score của caption đó.
- Baseline giúp giảm variance và chỉ so sánh tương đối giữa các beam của cùng video.

Đây là policy-gradient style objective:

```text
L = - E[ log p_theta(y | x) * (R(y) - b) ]
```

Trong code, `log p_theta(y | x)` được xấp xỉ bằng normalized beam `sequences_scores`, `R(y)` là CIDEr, và `b` là mean CIDEr trong beam set.

## Evaluation Khác Training Như Nào

Validation/evaluation không dùng SCST loss. `CaptionTask.valid_step()` gọi:

```python
captions = model.generate(
    samples,
    use_nucleus_sampling=False,
    num_beams=self.num_beams,
    max_length=self.max_len,
    min_length=self.min_len,
)
```

Sau đó lưu `{caption, image_id}` và `caption_eval()` đọc ground-truth JSON. Khác với training reward, evaluation group tất cả annotation caption theo `image_id`:

```python
references[ann["image_id"]].append(ann["caption"])
```

Do đó evaluation có thể là multi-reference CIDEr/BLEU/METEOR/ROUGE, còn training SCST trong code là single-reference theo sample.

## Các Điểm Cần Lưu Ý Khi Đọc Kết Quả

- SCST trong project là beam-based SCST/RL loss, không phải đúng greedy-baseline SCST cổ điển.
- Reward chỉ là CIDEr, không kết hợp BLEU/METEOR/ROUGE/CLIPScore.
- Training reward dùng one-reference caption per sample; evaluation có thể dùng nhiều reference per video.
- Generation trong SCST dùng `do_sample=False`, nên các candidates là beam outputs, không phải sampled rollouts.
- `top_p=0.9` và `temperature=1` được truyền vào generate nhưng không có tác dụng chính khi `do_sample=False`.
- T5 SCST dùng prompt `"a video of"` từ config; OPT SCST hiện không đưa prompt text vào generation vì phần prompt trong nhánh SCST bị comment.
- Trong T5 config SCST thường load checkpoint đã fine-tune bằng cross-entropy trước, rồi mới chạy RL/SCST.

## Tóm Tắt Ngắn

Project setup SCST bằng `model.scst: True`. Trong mỗi training step, model generate `beam_size` caption cho mỗi video, duplicate ground-truth caption tương ứng cho từng beam, tokenize cả generated và ground-truth bằng PTBTokenizer, tính per-caption CIDEr, lấy mean CIDEr của các beam làm baseline, rồi tối ưu:

```text
loss = mean(- sequence_score * (CIDEr(candidate, gt) - mean_beam_CIDEr))
```

Caption generated được so sánh với ground-truth bằng overlap TF-IDF n-gram 1..4 theo CIDEr, có length penalty Gaussian. Training so sánh mỗi candidate với caption ground-truth của chính sample đó; evaluation mới gom nhiều ground-truth captions theo `image_id`.
