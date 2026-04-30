# CE loss vs 20 ground-truth captions trong project

## Kết luận ngắn

Trong quá trình **training bằng CE/XE loss hiện tại**, model **không sinh generated caption rồi so với 20 ground-truth captions**.

Thay vào đó, mỗi training sample chỉ dùng **1 caption ground truth cụ thể** làm label cho teacher forcing. CE loss được tính token-by-token giữa phân phối output của T5 decoder và **1 GT caption** đó.

Việc so sánh **1 generated caption với toàn bộ GT captions của video** chỉ xảy ra ở bước **evaluation metrics** như BLEU, METEOR, ROUGE_L, CIDEr, không phải ở CE loss.

## 1. Dataloader training: mỗi sample là 1 video-caption pair

Với MSRVTT ở training XE mode, dataloader expand toàn bộ captions thành nhiều sample:

```python
elif split_type == "train":  # XE mode: expand all sentences
    for itm in self.data['sentences']:
        if itm['video_id'] in choiced_video_ids:
            self.sentences_dict[len(self.sentences_dict)] = (itm['video_id'], itm['caption'])
            self.video_sentences_dict[itm['video_id']].append(itm['caption'])
```

File: `dataloaders/dataloader_msrvtt_caption.py`

Ý nghĩa:

- Nếu 1 video có 20 captions GT, dataloader tạo khoảng 20 sample khác nhau cho video đó.
- Mỗi sample có cùng `video_id` nhưng khác `caption`.
- Khi batch đi vào model, mỗi dòng trong batch chỉ có **1 caption label**.

Vì vậy CE training không lấy một generated caption rồi đối chiếu đồng thời với 20 GT. Nó học từng cặp `(video, one_caption)`.

## 2. Caption label được tokenized làm target cho T5

Trong `_get_text`, caption của sample được tokenize:

```python
if caption is not None:
    caption_words = self.tokenizer.tokenize(caption)
else:
    caption_words = self._get_single_text(video_id)
```

Sau đó project cũng tokenize raw caption bằng T5 tokenizer:

```python
t5_tokens = self.t5_tokenizer(
    raw_caption,
    padding="max_length",
    truncation=True,
    max_length=t5_max_len,
    return_tensors="np",
)
pairs_t5_output_caption_ids[i] = t5_tokens.input_ids[0]
```

File: `dataloaders/dataloader_msrvtt_caption.py`

`pairs_t5_output_caption_ids` chính là label dùng cho T5 CE loss. Nó tương ứng với **1 raw caption** của sample hiện tại.

## 3. Training loop đưa 1 caption label vào model

Trong trainer:

```python
loss = model(
    input_ids,
    segment_ids,
    input_mask,
    video,
    video_mask,
    ...
    output_caption_ids=pairs_output_caption_ids,
    t5_output_caption_ids=pairs_t5_output_caption_ids,
    gt_refs=gt_refs,
)
```

File: `trainers/trainer.py`

Ở CE training bình thường, `gt_refs` không được dùng. `gt_refs` chỉ được build khi bật `--scst`.

## 4. CE loss thực chất tính thế nào

Trong `modules/modeling.py`, caption loss đi vào `_get_t5_caption_loss`, sau đó gọi `_compute_xe_caption_loss`:

```python
targets = output_tokens.masked_fill(output_tokens.eq(pad_token_id), -100)

outputs = self.t5_model(
    inputs_embeds=inputs_embeds,
    attention_mask=encoder_atts,
    decoder_attention_mask=output_mask,
    return_dict=True,
    labels=targets,
)
return outputs.loss
```

File: `modules/modeling.py`

Đây là standard teacher-forcing cross entropy của T5:

- input encoder: video representation từ QFormer + prompt `" A video of"`;
- decoder label: `t5_output_caption_ids`;
- padding token bị ignore bằng `-100`;
- loss là trung bình token-level negative log-likelihood trên caption label.

Do đó CE loss đang trả lời câu hỏi:

> Với video này và prefix decoder đúng ở từng bước, xác suất model gán cho token tiếp theo của **caption GT hiện tại** là bao nhiêu?

Nó không trả lời câu hỏi:

> Caption mà model generate tự do giống nhất với caption nào trong 20 GT?

## 5. Validation CE loss cũng chỉ dùng 1 GT

Ở validation/test MSRVTT, dataloader không expand 20 captions thành 20 samples. Nó tạo 1 sample mỗi video và chọn caption đầu tiên làm caption label:

```python
elif split_type == "val" or split_type == "test":
    for itm in self.data['sentences']:
        if itm['video_id'] in choiced_video_ids:
            self.video_sentences_dict[itm['video_id']].append(itm['caption'])
    for vid in choiced_video_ids:
        self.sentences_dict[len(self.sentences_dict)] = (vid, self.video_sentences_dict[vid][0])
```

File: `dataloaders/dataloader_msrvtt_caption.py`

Trong eval:

```python
loss, visual_output = model(...)
total_loss += float(loss)
avg_val_loss = total_loss / len(test_dataloader)
```

File: `inference/caption_generator.py`

Vì vậy `Average Validation Loss` là CE loss với **caption đầu tiên của mỗi video**, không phải CE loss trên 20 captions.

## 6. Generated caption được so với 20 GT ở đâu?

Sau khi tính validation CE loss, eval mới generate caption:

```python
generated_ids = model.generate_caption_ids(
    visual_output,
    video_mask,
    num_beams=eval_beam_size,
    max_length=max_length,
)
```

File: `inference/caption_generator.py`

Sau đó, nếu datatype là MSRVTT, code thay reference list bằng toàn bộ captions của từng video:

```python
if args.datatype == "msrvtt":
    all_caption_lists = []
    sentences_dict = test_dataloader.dataset.sentences_dict
    video_sentences_dict = test_dataloader.dataset.video_sentences_dict
    for idx in range(len(sentences_dict)):
        video_id, _ = sentences_dict[idx]
        sentences = video_sentences_dict[video_id]
        all_caption_lists.append(sentences)
    all_caption_lists = [list(itms) for itms in zip(*all_caption_lists)]
```

File: `inference/caption_generator.py`

Sau đó `CaptionEvaluator` chuyển sang COCO format:

```python
gts[i] = [ref_list[j][i] for j in range(len(ref_list))]
res[i] = [hyp]
```

File: `metrics.py`

Ý nghĩa:

- `res[i]`: 1 generated caption của video `i`;
- `gts[i]`: toàn bộ reference captions của video `i`, ví dụ 20 captions trong MSRVTT;
- metrics: BLEU_1..4, METEOR, ROUGE_L, CIDEr.

Đây là nơi **1 generated caption được đánh giá với nhiều GT captions**.

## 7. Phân biệt rõ CE loss và metric evaluation

| Giai đoạn | Có generate caption không? | So với mấy GT captions? | Dùng để làm gì? |
| --- | --- | --- | --- |
| Training CE/XE | Không, dùng teacher forcing | 1 GT caption/sample | Backprop optimize model |
| Validation CE loss | Không, dùng teacher forcing | 1 GT caption/video, hiện là caption đầu tiên | Log `Average Validation Loss` |
| Eval metrics | Có, dùng beam search | Tất cả GT captions/video, MSRVTT thường là 20 | Tính BLEU/METEOR/ROUGE/CIDEr |
| Best checkpoint | Có | Tất cả GT captions/video | Chọn best theo CIDEr, tie-break BLEU_4 |

## 8. Vấn đề cần chú ý

Nếu mục tiêu của bạn là “generated caption phải được train trực tiếp theo reward so với 20 GT”, CE loss hiện tại **chưa làm điều đó**.

CE loss hiện tại chỉ học maximum likelihood trên từng caption đơn lẻ. Một video có 20 captions thì trong train video đó xuất hiện nhiều lần với 20 labels khác nhau, nhưng mỗi forward/backward step vẫn chỉ nhìn thấy 1 label caption cho sample đó.

Muốn training dùng toàn bộ 20 GT theo generated caption thì cần cơ chế khác, ví dụ:

- SCST/CIDEr reward với `--scst`, vì code đã có nhánh build `gt_refs` gồm all captions per video;
- multi-reference CE tự thiết kế, ví dụ tính CE trên nhiều captions rồi lấy min/mean, nhưng hiện project chưa làm như vậy cho CE bình thường.

