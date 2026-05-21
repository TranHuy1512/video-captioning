#!/usr/bin/env python3
"""Smoke test the UniVL -> QFormer -> Phi caption decoder path."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from types import SimpleNamespace

import torch


REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

LOGGER = logging.getLogger("test_phi_decoder")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test the Phi decoder path used by modules/modeling.py.")
    parser.add_argument("--llm_model", default="microsoft/Phi-4-mini-instruct")
    parser.add_argument("--bert_model", default="bert-base-uncased")
    parser.add_argument("--visual_model", default="visual-base")
    parser.add_argument("--cross_model", default="cross-base")
    parser.add_argument("--decoder_model", default="decoder-base")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--caption", default="a person is cooking in a kitchen")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_words", type=int, default=20)
    parser.add_argument("--max_frames", type=int, default=8)
    parser.add_argument("--video_dim", type=int, default=768)
    parser.add_argument("--max_txt_len", type=int, default=24)
    parser.add_argument("--num_query_token", type=int, default=8)
    parser.add_argument("--qformer_vision_width", type=int, default=768)
    parser.add_argument("--qformer_checkpoint", default="")
    parser.add_argument("--qformer_checkpoint_file", default="")
    parser.add_argument("--qformer_checkpoint_local_files_only", action="store_true")
    parser.add_argument("--visual_num_hidden_layers", type=int, default=1)
    parser.add_argument("--cross_num_hidden_layers", type=int, default=1)
    parser.add_argument("--text_num_hidden_layers", type=int, default=1)
    parser.add_argument("--beam_size", type=int, default=1)
    parser.add_argument("--lora", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested, but CUDA is not available.")
        return torch.device("cuda")
    if name == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_task_config(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        do_pretrain=False,
        do_train=False,
        do_eval=True,
        stage_two=True,
        task_type="caption",
        datatype="synthetic",
        local_rank=0,
        world_size=1,
        n_gpu=1,
        batch_size=args.batch_size,
        batch_size_val=args.batch_size,
        n_pair=1,
        margin=0.1,
        hard_negative_rate=0.5,
        negative_weighting=1,
        use_mil=False,
        sampled_use_mil=False,
        max_words=args.max_words,
        max_frames=args.max_frames,
        video_dim=args.video_dim,
        text_num_hidden_layers=args.text_num_hidden_layers,
        visual_num_hidden_layers=args.visual_num_hidden_layers,
        cross_num_hidden_layers=args.cross_num_hidden_layers,
        decoder_num_hidden_layers=1,
        freeze_vit=False,
        scst=False,
        beam_size=args.beam_size,
        eval_beam_size=args.beam_size,
        scst_num_samples=args.beam_size,
        llm_model=args.llm_model,
        max_txt_len=args.max_txt_len,
        num_query_token=args.num_query_token,
        qformer_vision_width=args.qformer_vision_width,
        qformer_checkpoint=args.qformer_checkpoint,
        qformer_checkpoint_file=args.qformer_checkpoint_file,
        qformer_checkpoint_local_files_only=args.qformer_checkpoint_local_files_only,
        qformer_diversity_weight=0.0,
        lora=args.lora,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        lora_target_modules=["q_proj", "v_proj"],
    )


def assert_finite_scalar(name: str, value: torch.Tensor) -> None:
    if value.ndim != 0:
        raise AssertionError(f"{name} should be scalar, got shape={tuple(value.shape)}")
    if not torch.isfinite(value.detach().float()).item():
        raise AssertionError(f"{name} is not finite: {value}")


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s: %(message)s")
    device = choose_device(args.device)

    from modules.modeling import UniVL

    LOGGER.info("Building UniVL + Phi decoder on %s", device)
    model = UniVL.from_pretrained(
        args.bert_model,
        args.visual_model,
        args.cross_model,
        args.decoder_model,
        state_dict=None,
        task_config=make_task_config(args),
    ).to(device)
    model.eval()

    input_ids = torch.zeros(args.batch_size, 1, args.max_words, dtype=torch.long, device=device)
    segment_ids = torch.zeros_like(input_ids)
    input_mask = torch.zeros_like(input_ids)
    input_mask[:, :, :2] = 1
    video = torch.randn(args.batch_size, 1, args.max_frames, args.video_dim, device=device)
    video_mask = torch.ones(args.batch_size, 1, args.max_frames, dtype=torch.long, device=device)
    input_caption_ids = torch.ones(args.batch_size, 1, args.max_words, dtype=torch.long, device=device)
    decoder_mask = torch.ones_like(input_caption_ids)

    decoder_tokens = model.phi_tokenizer(
        [args.caption] * args.batch_size,
        padding="max_length",
        truncation=True,
        max_length=args.max_txt_len,
        return_tensors="pt",
        add_special_tokens=True,
    ).to(device)
    decoder_output_caption_ids = decoder_tokens.input_ids.view(args.batch_size, 1, -1)

    with torch.no_grad():
        loss, visual_output = model(
            input_ids,
            segment_ids,
            input_mask,
            video,
            video_mask,
            input_caption_ids=input_caption_ids,
            decoder_mask=decoder_mask,
            output_caption_ids=torch.zeros(args.batch_size, 1, args.max_words, dtype=torch.long, device=device),
            decoder_output_caption_ids=decoder_output_caption_ids,
        )
        flat_video_mask = video_mask.view(-1, video_mask.shape[-1])
        cross_output, _ = model._get_cross_output(visual_output, flat_video_mask)
        visual_prefix = model.phi_proj(cross_output)
        generated_ids = model.generate_caption_ids(visual_output, flat_video_mask, num_beams=args.beam_size, max_length=8)

    assert_finite_scalar("phi_caption_loss", loss)
    expected_prefix_shape = (args.batch_size, args.num_query_token, model.phi_model.config.hidden_size)
    if tuple(visual_prefix.shape) != expected_prefix_shape:
        raise AssertionError(f"visual prefix shape mismatch: got {tuple(visual_prefix.shape)}, expected {expected_prefix_shape}")

    LOGGER.info("phi_caption_loss=%.6f", float(loss.detach().float().cpu()))
    LOGGER.info("visual_prefix_shape=%s", tuple(visual_prefix.shape))
    LOGGER.info("generated=%r", model.phi_tokenizer.batch_decode(generated_ids, skip_special_tokens=True))
    LOGGER.info("Phi decoder smoke test passed.")


if __name__ == "__main__":
    main()
