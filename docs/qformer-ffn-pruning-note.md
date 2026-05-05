# QFormer FFN pruning note

Question checked:

```python
for layer in self.Qformer.bert.encoder.layer:
    layer.output = None
    layer.intermediate = None
```

Short answer: for this repo, do not enable those two lines in `modules/modeling.py`.

They are not needed for using pretrained BLIP2 QFormer weights, and they do not make the active query-token QFormer stronger. In the current main caption pipeline, the lines are already commented out:

```python
self.Qformer.cls = None
self.Qformer.bert.embeddings.word_embeddings = None
self.Qformer.bert.embeddings.position_embeddings = None
# for layer in self.Qformer.bert.encoder.layer:
#     layer.output = None
#     layer.intermediate = None
```

## What those lines remove

In `modules/Qformer.py`, each `BertLayer` has two separate feed-forward paths:

```python
self.intermediate = BertIntermediate(config)
self.output = BertOutput(config)

self.intermediate_query = BertIntermediate(config)
self.output_query = BertOutput(config)
```

The normal `intermediate/output` path is used by `feed_forward_chunk()`. The query-token path is `intermediate_query/output_query`, used by `feed_forward_chunk_query()`.

When QFormer receives only `query_embeds`, `query_length` equals the whole sequence length. The layer runs this branch:

```python
layer_output = apply_chunking_to_forward(
    self.feed_forward_chunk_query,
    self.chunk_size_feed_forward,
    self.seq_len_dim,
    query_attention_output,
)
```

So the active query tokens use `intermediate_query` and `output_query`, not `intermediate` and `output`.

## Why old BLIP2-T5 code used it

`modules/blip2_t5.py` still contains the pruning lines. That file follows the original BLIP2-FlanT5 style where the QFormer is used only as a visual query transformer before T5. Since no text tokens are passed through the QFormer in that path, the normal text FFN blocks are unused and can be removed to save parameters/memory.

That pruning is an optimization, not a better-QFormer trick.

## Current repo behavior

The active caption branch in `modules/modeling.py` calls:

```python
self.Qformer.bert(
    query_embeds=query_tokens,
    encoder_hidden_states=visual_for_qformer,
    encoder_attention_mask=image_atts,
    return_dict=True,
)
```

There is no `input_ids` argument here. This means QFormer receives query tokens only.

For this exact path:

- Keeping `layer.output` and `layer.intermediate` does not change the query output.
- Setting them to `None` also should not change the query output, because they are not called.
- But setting them to `None` removes compatibility with any future path that sends text tokens through QFormer.
- Keeping them preserves the pretrained module structure and makes checkpoint loading/debugging easier.

## Recommendation

Use this in `modules/modeling.py`:

```python
self.Qformer.cls = None
self.Qformer.bert.embeddings.word_embeddings = None
self.Qformer.bert.embeddings.position_embeddings = None
# Keep layer.output and layer.intermediate.
```

Do not enable:

```python
for layer in self.Qformer.bert.encoder.layer:
    layer.output = None
    layer.intermediate = None
```

This is especially the safer choice while experimenting with pretrained QFormer checkpoints such as:

```bash
--qformer_checkpoint Salesforce/blip2-opt-6.7b-coco
```

## If you want to compare experimentally

A fair ablation is:

1. Train/evaluate with the current `modules/modeling.py` behavior.
2. Enable the pruning lines.
3. Keep the same checkpoint, seed, data split, learning rates, LoRA settings, and SCST setting.
4. Compare validation CIDEr/BLEU/METEOR and generated captions.

Expected result for the current query-only path: almost no metric difference from the pruning itself. If there is a difference, it is more likely from secondary effects such as optimizer parameter grouping, checkpoint missing/unexpected keys, memory pressure, or a code path accidentally using QFormer text tokens.

## Final decision

For this project: leave `layer.output` and `layer.intermediate` intact. The advice "do not use it to enhance better QFormer" is correct for this repo because those lines are not an enhancement; at best they remove unused text-token FFN modules, and at worst they break future QFormer text-token usage.
