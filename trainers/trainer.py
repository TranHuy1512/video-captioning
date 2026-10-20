import torch
import time


def train_epoch(epoch, args, model, train_dataloader, device, n_gpu, optimizer, scheduler,
                global_step, logger, local_rank=0):
    torch.cuda.empty_cache()
    model.train()
    log_step = args.n_display
    start_time = time.time()
    total_loss = 0

    # Check if SCST is enabled — if so, we need to look up all GT refs per video
    use_scst = getattr(args, 'scst', False)
    dataset = train_dataloader.dataset

    for step, batch in enumerate(train_dataloader):
        # Last element is sample indices (ints), rest are tensors
        sample_indices = batch[-1]
        tensor_batch = batch[:-1]
        tensor_batch = tuple(t.to(device=device, non_blocking=True) for t in tensor_batch)

        input_ids, input_mask, segment_ids, video, video_mask, \
        pairs_output_caption_ids, pairs_t5_output_caption_ids = tensor_batch

        # Multi-reference SCST reward: score each beam against ALL ground-truth
        # captions for that video (same set eval_epoch uses for CIDEr), not just
        # the single caption this row happens to carry. Without this, the reward
        # target flips to a different random-ish caption every step, which lets
        # a generic/"safe" caption look decent on average across many steps —
        # a plausible contributor to templated, non-specific captions. Falls
        # back to single-reference (previous behaviour) for any dataset that
        # doesn't expose `sentences_dict`/`video_sentences_dict` (e.g. YouCookII).
        gt_refs = None
        if use_scst and hasattr(dataset, "sentences_dict") and hasattr(dataset, "video_sentences_dict"):
            idx_list = sample_indices.tolist() if torch.is_tensor(sample_indices) else list(sample_indices)
            gt_refs = [
                dataset.video_sentences_dict[dataset.sentences_dict[idx][0]]
                for idx in idx_list
            ]

        loss = model(input_ids, segment_ids, input_mask, video, video_mask,
                     output_caption_ids=pairs_output_caption_ids,
                     t5_output_caption_ids=pairs_t5_output_caption_ids,
                     gt_refs=gt_refs)

        if n_gpu > 1:
            loss = loss.mean()
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        loss.backward()

        total_loss += float(loss)
        if (step + 1) % args.gradient_accumulation_steps == 0:

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            if scheduler is not None:
                scheduler.step()

            optimizer.step()
            optimizer.zero_grad()

            global_step += 1
            if global_step % log_step == 0 and local_rank == 0:
                logger.info("Epoch: %d/%s, Step: %d/%d, Lr: %s, Loss: %f, Time/step: %f", epoch + 1,
                            args.epochs, step + 1,
                            len(train_dataloader), "-".join([str('%.6f'%itm) for itm in sorted(list(set(optimizer.get_lr())))]),
                            float(loss),
                            (time.time() - start_time) / (log_step * args.gradient_accumulation_steps))
                start_time = time.time()

    total_loss = total_loss / len(train_dataloader)
    return total_loss, global_step
