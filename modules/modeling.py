# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch BERT model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import numpy as np
import itertools

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM, AutoTokenizer


from modules.until_module import PreTrainedModel, LayerNorm, CrossEn, MILNCELoss, MaxMarginRankingLoss
from modules.module_bert import BertModel, BertConfig, BertOnlyMLMHead
from modules.module_visual import VisualModel, VisualConfig, VisualOnlyMLMHead
from modules.module_cross import CrossModel, CrossConfig
from modules.module_decoder import DecoderConfig
from modules.blip2 import Blip2Base, disabled_train
from peft import LoraConfig, TaskType, get_peft_model

logger = logging.getLogger(__name__)


def tokenize(refs, cands, no_op=False):
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

    tokenizer = PTBTokenizer()

    if no_op:
        refs = {idx: [r for r in c_refs] for idx, c_refs in enumerate(refs)}
        cands = {idx: [c] for idx, c in enumerate(cands)}
    else:
        refs = {idx: [{'caption': r} for r in c_refs] for idx, c_refs in enumerate(refs)}
        cands = {idx: [{'caption': c}] for idx, c in enumerate(cands)}

        refs = tokenizer.tokenize(refs)
        cands = tokenizer.tokenize(cands)

    return refs, cands


class UniVLPreTrainedModel(PreTrainedModel, nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    def __init__(self, bert_config, visual_config, cross_config, decoder_config, *inputs, **kwargs):
        # utilize bert config as base config
        super(UniVLPreTrainedModel, self).__init__(bert_config)
        self.bert_config = bert_config
        self.visual_config = visual_config
        self.cross_config = cross_config
        self.decoder_config = decoder_config

        self.bert = None
        self.visual = None
        self.cross = None
        self.decoder = None

    @classmethod
    def from_pretrained(cls, pretrained_bert_name, visual_model_name, cross_model_name, decoder_model_name,
                        state_dict=None, cache_dir=None, type_vocab_size=2, *inputs, **kwargs):

        task_config = None
        if "task_config" in kwargs.keys():
            task_config = kwargs["task_config"]
            if not hasattr(task_config, "local_rank"):
                task_config.__dict__["local_rank"] = 0
            elif task_config.local_rank == -1:
                task_config.local_rank = 0

        bert_config, state_dict = BertConfig.get_config(pretrained_bert_name, cache_dir, type_vocab_size, state_dict, task_config=task_config)
        visual_config, _ = VisualConfig.get_config(visual_model_name, cache_dir, type_vocab_size, state_dict=None, task_config=task_config)
        visual_config = update_attr("visual_config", visual_config, "vocab_size", task_config, "video_dim")
        cross_config, _ = CrossConfig.get_config(cross_model_name, cache_dir, type_vocab_size, state_dict=None, task_config=task_config)
        decoder_config, _ = DecoderConfig.get_config(decoder_model_name, cache_dir, type_vocab_size, state_dict=None, task_config=task_config)

        model = cls(bert_config, visual_config, cross_config, decoder_config, *inputs, **kwargs)

        assert model.bert is not None
        assert model.visual is not None

        if state_dict is not None:
            state_dict = cls._filter_init_model_state_dict(state_dict, task_config=task_config)
            state_dict = cls._filter_mismatched_init_model_state_dict(
                state_dict, model, task_config=task_config
            )
            model = cls.init_preweight(model, state_dict, task_config=task_config)

        return model

    @staticmethod
    def _filter_init_model_state_dict(state_dict, task_config=None):
        allowed_prefixes = (
            "bert.",
            "visual.",
            "Qformer.",
            "query_tokens",
            "qformer_visual_proj.",
            "llama_model.",
            "llama_proj.",
            "normalize_video.",
        )
        filtered_state_dict = state_dict.__class__(
            (key, value) for key, value in state_dict.items()
            if key.startswith(allowed_prefixes)
        )
        metadata = getattr(state_dict, "_metadata", None)
        if metadata is not None:
            filtered_state_dict._metadata = metadata

        skipped = len(state_dict) - len(filtered_state_dict)
        show_log(
            task_config,
            "Load init_model weights only for {} Skipped {} other tensors.".format(
                ", ".join(allowed_prefixes), skipped
            )
        )
        return filtered_state_dict

    @staticmethod
    def _filter_mismatched_init_model_state_dict(state_dict, model, task_config=None):
        target_state = model.state_dict()
        kept_state_dict = state_dict.__class__()
        metadata = getattr(state_dict, "_metadata", None)
        if metadata is not None:
            kept_state_dict._metadata = metadata

        skipped_missing = []
        skipped_shape = []
        for key, value in state_dict.items():
            if key not in target_state:
                skipped_missing.append(key)
                continue

            target_value = target_state[key]
            if tuple(value.shape) != tuple(target_value.shape):
                skipped_shape.append((key, tuple(value.shape), tuple(target_value.shape)))
                continue

            kept_state_dict[key] = value

        show_log(
            task_config,
            "Init checkpoint shape filter: kept {} tensors, skipped {} missing keys, skipped {} shape mismatches.".format(
                len(kept_state_dict), len(skipped_missing), len(skipped_shape)
            )
        )
        if skipped_missing:
            show_log(task_config, "First skipped missing init keys: {}".format(skipped_missing[:10]))
        if skipped_shape:
            show_log(task_config, "First skipped shape-mismatch init keys: {}".format(skipped_shape[:10]))

        return kept_state_dict

class NormalizeVideo(nn.Module):
    def __init__(self, task_config):
        super(NormalizeVideo, self).__init__()
        self.visual_norm2d = LayerNorm(task_config.video_dim)

    def forward(self, video):
        video = torch.as_tensor(video).float()
        video = video.view(-1, video.shape[-2], video.shape[-1])
        video = self.visual_norm2d(video)
        return video

def show_log(task_config, info):
    if task_config is None or task_config.local_rank == 0:
        logger.warning(info)

def update_attr(target_name, target_config, target_attr_name, source_config, source_attr_name, default_value=None):
    if hasattr(source_config, source_attr_name):
        if default_value is None or getattr(source_config, source_attr_name) != default_value:
            setattr(target_config, target_attr_name, getattr(source_config, source_attr_name))
            show_log(source_config, "Set {}.{}: {}.".format(target_name,
                                                            target_attr_name, getattr(target_config, target_attr_name)))
    return target_config

def check_attr(target_name, task_config):
    return hasattr(task_config, target_name) and task_config.__dict__[target_name]

class UniVL(UniVLPreTrainedModel):
    def __init__(self, bert_config, visual_config, cross_config, decoder_config, task_config):
        super(UniVL, self).__init__(bert_config, visual_config, cross_config, decoder_config)
        self.task_config = task_config
        self.ignore_video_index = -1

        assert self.task_config.max_words <= bert_config.max_position_embeddings
        assert self.task_config.max_words <= decoder_config.max_target_embeddings
        assert self.task_config.max_frames <= visual_config.max_position_embeddings
        assert self.task_config.max_words + self.task_config.max_frames <= cross_config.max_position_embeddings

        self._stage_one = True
        self._stage_two = False

        if check_attr('stage_two', self.task_config):
            self._stage_one = False
            self._stage_two = self.task_config.stage_two
        show_log(task_config, "Stage-One:{}, Stage-Two:{}".format(self._stage_one, self._stage_two))

        self.train_sim_after_cross = False
        if self._stage_one and check_attr('train_sim_after_cross', self.task_config):
            self.train_sim_after_cross = True
            show_log(task_config, "Test retrieval after cross encoder.")

        # Text Encoder ===>
        bert_config = update_attr("bert_config", bert_config, "num_hidden_layers",
                                   self.task_config, "text_num_hidden_layers")
        self.bert = BertModel(bert_config)
        bert_word_embeddings_weight = self.bert.embeddings.word_embeddings.weight
        # <=== End of Text Encoder

        # Video Encoder ===>
        visual_config = update_attr("visual_config", visual_config, "num_hidden_layers",
                                    self.task_config, "visual_num_hidden_layers")
        self.visual = VisualModel(visual_config)
        self.freeze_vit = getattr(self.task_config, "freeze_vit", False)
        if self.freeze_vit:
            for param in self.visual.parameters():
                param.requires_grad = False
            self.visual = self.visual.eval()
            self.visual.train = disabled_train
            show_log(task_config, "Freeze vision encoder.")
        visual_word_embeddings_weight = self.visual.embeddings.word_embeddings.weight
        # <=== End of Video Encoder

        if self._stage_one is False or self.train_sim_after_cross:
            # Cross Encoder ===>
            cross_config = update_attr("cross_config", cross_config, "num_hidden_layers",
                                        self.task_config, "cross_num_hidden_layers")
            self.cross = CrossModel(cross_config)
            self.num_query_token = getattr(self.task_config, "num_query_token", 32)
            self.qformer_vision_width = getattr(self.task_config, "qformer_vision_width", visual_config.hidden_size)
            if self.qformer_vision_width != visual_config.hidden_size:
                self.qformer_visual_proj = nn.Linear(visual_config.hidden_size, self.qformer_vision_width)
                show_log(
                    task_config,
                    "Add QFormer visual projection: {} -> {}.".format(
                        visual_config.hidden_size, self.qformer_vision_width
                    )
                )
            else:
                self.qformer_visual_proj = nn.Identity()
            self.Qformer, self.query_tokens = Blip2Base.init_Qformer(
                self.num_query_token,
                self.qformer_vision_width,
                qformer_checkpoint=getattr(self.task_config, "qformer_checkpoint", None),
                qformer_checkpoint_file=getattr(self.task_config, "qformer_checkpoint_file", None),
                local_files_only=getattr(self.task_config, "qformer_checkpoint_local_files_only", False),
            )
            self.Qformer.cls = None
            self.Qformer.bert.embeddings.word_embeddings = None
            self.Qformer.bert.embeddings.position_embeddings = None
            for layer in self.Qformer.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None
            # <=== End of Cross Encoder

            if self.train_sim_after_cross is False:
                # Decoder ===>
                self.scst = getattr(self.task_config, "scst", False)
                self.eval_beam_size = getattr(
                    self.task_config,
                    "eval_beam_size",
                    getattr(self.task_config, "beam_size", 5),
                )
                self.scst_num_samples = getattr(
                    self.task_config,
                    "scst_num_samples",
                    getattr(self.task_config, "beam_size", self.eval_beam_size),
                )
                # Backwards compatibility for older call sites/check scripts.
                self.beam_size = self.eval_beam_size
                self.max_txt_len = getattr(self.task_config, "max_txt_len", 32)
                self.prompt = "Describe the video in one concise sentence:"

                llama_model_name = getattr(self.task_config, "llama_model", "meta-llama/Llama-3.2-3B-Instruct")
                hf_token = getattr(self.task_config, "hf_token", None)
                hf_kwargs = {"token": hf_token} if hf_token else {}
                self.llama_tokenizer = AutoTokenizer.from_pretrained(llama_model_name, **hf_kwargs)
                if self.llama_tokenizer.pad_token is None:
                    self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
                self.llama_tokenizer.padding_side = "right"
                self.llama_model = AutoModelForCausalLM.from_pretrained(
                    llama_model_name,
                    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                    **hf_kwargs,
                )
                self.llama_model.config.pad_token_id = self.llama_tokenizer.pad_token_id
                llama_hidden_size = self.llama_model.config.hidden_size
                for name, param in self.llama_model.named_parameters():
                    param.requires_grad = False

                lora = getattr(self.task_config, "lora", False)
                lora_r = getattr(self.task_config, "lora_r", 16)
                lora_alpha = getattr(self.task_config, "lora_alpha", 32)
                lora_dropout = getattr(self.task_config, "lora_dropout", 0.05)
                lora_target_modules = getattr(self.task_config, 'lora_target_modules', ['q_proj', 'v_proj'])
                peft_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    inference_mode=False,
                    r=lora_r,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    target_modules=lora_target_modules,
                )

                if lora:
                    self.llama_model = get_peft_model(self.llama_model, peft_config)
                    self.llama_model.print_trainable_parameters()

                self.llama_proj = nn.Linear(
                    self.Qformer.config.hidden_size, llama_hidden_size
                )
                # <=== End of Decoder

            if self.task_config.do_pretrain:
                self.cls = BertOnlyMLMHead(bert_config, bert_word_embeddings_weight)
                self.cls_visual = VisualOnlyMLMHead(visual_config, visual_word_embeddings_weight)
                self.alm_loss_fct = CrossEntropyLoss(ignore_index=-1)
                
            self.similarity_dense = nn.Linear(bert_config.hidden_size, 1)

        self.normalize_video = NormalizeVideo(task_config)

        mil_nce_loss = MILNCELoss(batch_size=task_config.batch_size // task_config.n_gpu, n_pair=task_config.n_pair, )
        max_margin_ranking_loss = MaxMarginRankingLoss(margin=task_config.margin,
                                   negative_weighting=task_config.negative_weighting,
                                   batch_size=task_config.batch_size // task_config.n_gpu,
                                   n_pair=task_config.n_pair,
                                   hard_negative_rate=task_config.hard_negative_rate, )

        if task_config.use_mil:
            self.loss_fct = CrossEn() if self._stage_two else mil_nce_loss
            self._pretrain_sim_loss_fct = mil_nce_loss
        else:
            self.loss_fct = CrossEn() if self._stage_two else max_margin_ranking_loss
            self._pretrain_sim_loss_fct = max_margin_ranking_loss
        
        self._cider_scorer = None  # lazy init
        self._init_weights_except_pretrained_submodules()

    def _init_weights_except_pretrained_submodules(self):
        skip_roots = {"Qformer", "llama_model", "llama_proj", "qformer_visual_proj", "query_tokens", "normalize_video"}

        def init_module(module):
            for name, child in module._modules.items():
                if child is None or name in skip_roots:
                    continue
                init_module(child)
            self.init_weights(module)

        init_module(self)

    def forward(self, input_ids, token_type_ids, attention_mask, video, video_mask=None,
                pairs_masked_text=None, pairs_token_labels=None, masked_video=None, video_labels_index=None,
                input_caption_ids=None, decoder_mask=None, output_caption_ids=None,
                llama_output_caption_ids=None, llama_output_caption_mask=None, gt_refs=None):

        input_ids = input_ids.view(-1, input_ids.shape[-1])
        token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
        attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
        video_mask = video_mask.view(-1, video_mask.shape[-1])
        video = self.normalize_video(video)

        if input_caption_ids is not None:
            input_caption_ids = input_caption_ids.view(-1, input_caption_ids.shape[-1])
            decoder_mask = decoder_mask.view(-1, decoder_mask.shape[-1])

        # Skip text encoder when it's not needed (caption-only fine-tuning)
        _need_text_encoder = (
            self._stage_one
            or (self._stage_two and self.task_config.do_pretrain)
            or (self._stage_two and self.task_config.task_type == "retrieval")
        )

        if _need_text_encoder:
            sequence_output, visual_output = self.get_sequence_visual_output(
                input_ids, token_type_ids, attention_mask, video, video_mask, shaped=True
            )
        else:
            visual_output = self.get_visual_output(video, video_mask, shaped=True)
            sequence_output = None

        if self.training:
            loss = 0.
            if self._stage_one:
                sim_matrix = self.get_similarity_logits(sequence_output, visual_output, attention_mask,
                                                        video_mask, shaped=True)
                sim_loss = self.loss_fct(sim_matrix)
                loss += sim_loss

            if self._stage_two:
                if self.task_config.do_pretrain:
                    pairs_masked_text = pairs_masked_text.view(-1, pairs_masked_text.shape[-1])
                    pairs_token_labels = pairs_token_labels.view(-1, pairs_token_labels.shape[-1])

                    masked_video = self.normalize_video(masked_video)
                    video_labels_index = video_labels_index.view(-1, video_labels_index.shape[-1])

                    sequence_output_alm, visual_output_alm = self.get_sequence_visual_output(pairs_masked_text, token_type_ids,
                                                                                             attention_mask, masked_video, video_mask, shaped=True)

                    sequence_cross_output = sequence_output_alm
                    visual_cross_output = visual_output_alm

                    alm_loss = self._calculate_mlm_loss(sequence_cross_output, pairs_token_labels)
                    loss += alm_loss

                    nce_loss = self._calculate_mfm_loss(visual_cross_output, video, video_mask, video_labels_index)
                    loss += nce_loss

                    sim_matrix = self.get_similarity_logits(sequence_output, visual_output, attention_mask, video_mask,
                                                            shaped=True, _pretrain_joint=True)
                    sim_loss_joint = self._pretrain_sim_loss_fct(sim_matrix)
                    loss += sim_loss_joint

                if (input_caption_ids is not None) and \
                        (self.task_config.do_pretrain
                         or (self.task_config.do_pretrain is False and self.task_config.task_type == "caption")):
                    if self.task_config.do_pretrain:
                        decoder_loss = self._get_llama_caption_loss(visual_output_alm,
                                                                    video_mask,
                                                                    output_caption_ids,
                                                                    llama_output_caption_ids,
                                                                    llama_output_caption_mask)
                    elif self.task_config.task_type == "caption":
                        decoder_loss = self._get_llama_caption_loss(visual_output,
                                                                    video_mask,
                                                                    output_caption_ids,
                                                                    llama_output_caption_ids,
                                                                    llama_output_caption_mask,
                                                                    gt_refs=gt_refs)
                    else:
                        raise NotImplementedError
                    loss += decoder_loss

                if self.task_config.do_pretrain or self.task_config.task_type == "retrieval":
                    if self.task_config.do_pretrain:
                        sim_matrix_text_visual = self.get_similarity_logits(sequence_output_alm, visual_output_alm,
                                                                            attention_mask, video_mask, shaped=True)
                    elif self.task_config.task_type == "retrieval":
                        sim_matrix_text_visual = self.get_similarity_logits(sequence_output, visual_output,
                                                                            attention_mask, video_mask, shaped=True)
                    else:
                        raise NotImplementedError

                    sim_loss_text_visual = self.loss_fct(sim_matrix_text_visual)
                    loss += sim_loss_text_visual

            return loss
        else:
            # During evaluation, return (loss, visual_output) so callers can
            # reuse visual_output for generation without re-encoding.
            if (self._stage_two and 
                input_caption_ids is not None and 
                output_caption_ids is not None and
                self.task_config.task_type == "caption"):
                decoder_loss = self._get_llama_caption_loss(visual_output,
                                                            video_mask,
                                                            output_caption_ids,
                                                            llama_output_caption_ids,
                                                            llama_output_caption_mask)
                return decoder_loss, visual_output
            else:
                return None, visual_output

    def _calculate_mlm_loss(self, sequence_output_alm, pairs_token_labels):
        alm_scores = self.cls(sequence_output_alm)
        alm_loss = self.alm_loss_fct(alm_scores.view(-1, self.bert_config.vocab_size), pairs_token_labels.view(-1))
        return alm_loss

    def _calculate_mfm_loss(self, visual_output_alm, video, video_mask, video_labels_index):
        afm_scores = self.cls_visual(visual_output_alm)
        afm_scores_tr = afm_scores.view(-1, afm_scores.shape[-1])

        video_tr = video.permute(2, 0, 1)
        video_tr = video_tr.view(video_tr.shape[0], -1)

        logits_matrix = torch.mm(afm_scores_tr, video_tr)
        video_mask_float = video_mask.to(dtype=torch.float)
        mask_matrix = torch.mm(video_mask_float.view(-1, 1), video_mask_float.view(1, -1))
        masked_logits = logits_matrix + (1. - mask_matrix) * -1e8

        logpt = F.log_softmax(masked_logits, dim=-1)
        logpt = torch.diag(logpt)
        nce_loss = -logpt

        video_labels_index_mask = (video_labels_index != self.ignore_video_index)
        nce_loss = nce_loss.masked_select(video_labels_index_mask.view(-1))
        nce_loss = nce_loss.mean()
        return nce_loss

    def get_sequence_visual_output(self, input_ids, token_type_ids, attention_mask, video, video_mask, shaped=False):
        if shaped is False:
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])
            video = self.normalize_video(video)

        encoded_layers, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=True)
        sequence_output = encoded_layers[-1]

        visual_layers, _ = self.visual(video, video_mask, output_all_encoded_layers=True)
        visual_output = visual_layers[-1]

        return sequence_output, visual_output

    def get_visual_output(self, video, video_mask, shaped=False):
        if shaped is False:
            video_mask = video_mask.view(-1, video_mask.shape[-1])
            video = self.normalize_video(video)

        visual_layers, _ = self.visual(video, video_mask, output_all_encoded_layers=True)
        visual_output = visual_layers[-1]
        return visual_output

    def _get_cross_output(self, visual_output, video_mask, num_query_token=32):
        # Use BLIP2 Qformer query cross-attention and expose query tokens as encoder outputs.
        b_visual, _, _ = visual_output.size()
        query_len = min(num_query_token, self.query_tokens.size(1))
        qformer_dtype = self.query_tokens.dtype
        query_tokens = self.query_tokens[:, :query_len, :].expand(b_visual, -1, -1).to(
            device=visual_output.device,
            dtype=qformer_dtype,
        )
        visual_for_qformer = self.qformer_visual_proj(visual_output).to(dtype=qformer_dtype)
        image_atts = video_mask.long()
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=visual_for_qformer,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        cross_output = query_output.last_hidden_state.to(dtype=visual_output.dtype)
        pooled_output = cross_output[:, 0]

        return cross_output, pooled_output

    def _build_llama_prefix_inputs(self, visual_output, video_mask, cross_output=None):
        """Build Llama visual-prefix embeddings from visual features via Q-Former.

        Args:
            visual_output: Visual encoder output [B, T, visual_dim]
            video_mask:    Binary mask for valid frames [B, T]
            cross_output:  Optional pre-computed Q-Former output [B, Q, hidden].
                           Pass this to reuse a Q-Former forward already done in
                           the caller (avoids a redundant second Q-Former pass).
        """
        if cross_output is None:
            cross_output, _ = self._get_cross_output(visual_output, video_mask)
        visual_prefix = self.llama_proj(cross_output)
        visual_atts = torch.ones(visual_prefix.size()[:-1], dtype=torch.long, device=visual_prefix.device)

        prompt = [self.prompt] * visual_prefix.size(0)
        prompt_tokens = self.llama_tokenizer(
            prompt,
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
            add_special_tokens=False,
        ).to(visual_prefix.device)

        prompt_embeds = self.llama_model.get_input_embeddings()(prompt_tokens.input_ids)
        inputs_embeds = torch.cat([visual_prefix, prompt_embeds], dim=1)
        attention_mask = torch.cat([visual_atts, prompt_tokens.attention_mask], dim=1)
        return inputs_embeds, attention_mask

    def _compute_xe_caption_loss(self, prefix_embeds, prefix_atts, output_caption_ids, output_caption_mask=None):
        pad_token_id = self.llama_tokenizer.pad_token_id
        output_tokens = output_caption_ids.clone()
        output_tokens = output_tokens.masked_fill(output_tokens.lt(0), pad_token_id)
        if output_caption_mask is None:
            output_mask = output_tokens.ne(pad_token_id).long()
        else:
            output_mask = output_caption_mask.long()
        targets = output_tokens.masked_fill(output_mask.eq(0), -100)

        caption_embeds = self.llama_model.get_input_embeddings()(output_tokens)
        inputs_embeds = torch.cat([prefix_embeds, caption_embeds], dim=1)
        attention_mask = torch.cat([prefix_atts, output_mask], dim=1)
        prefix_targets = torch.full(
            prefix_atts.shape, -100, dtype=targets.dtype, device=targets.device
        )
        labels = torch.cat([prefix_targets, targets], dim=1)

        outputs = self.llama_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            labels=labels,
        )
        return outputs.loss

    def _compute_diversity_loss(self, cross_output):
        """Penalise high cosine similarity between different Q-Former query tokens.

        This encourages the 32 query tokens to specialise to different temporal/
        semantic aspects of the video instead of collapsing to near-identical
        representations.

        Args:
            cross_output: Q-Former output [B, Q, hidden]
        Returns:
            Scalar diversity loss (mean off-diagonal cosine similarity).
        """
        # Work in float32 for numerical stability; cross_output may be bfloat16
        q = F.normalize(cross_output.float(), dim=-1)          # [B, Q, H]
        sim = torch.bmm(q, q.transpose(1, 2))                  # [B, Q, Q]  values in [-1, 1]
        Q = sim.size(1)
        # Mask self-similarity and penalize only positive off-diagonal similarity.
        off_diag = 1.0 - torch.eye(Q, device=sim.device).unsqueeze(0)  # [1, Q, Q]
        diversity_loss = (F.relu(sim) * off_diag).sum() / (off_diag.sum() * sim.size(0))
        return diversity_loss.to(cross_output.dtype)

    def _get_llama_caption_loss(self, visual_output, video_mask, output_caption_ids, llama_output_caption_ids=None,
                                llama_output_caption_mask=None, gt_refs=None):
        if output_caption_ids is None:
            return torch.tensor(0.0, device=visual_output.device)
        if llama_output_caption_ids is None:
            raise ValueError(
                "llama_output_caption_ids is required for Llama caption loss. "
                "output_caption_ids uses the BERT vocab and must not be used as Llama labels."
            )

        output_caption_ids = output_caption_ids.view(-1, output_caption_ids.shape[-1])
        llama_output_caption_ids = llama_output_caption_ids.view(-1, llama_output_caption_ids.shape[-1])
        if llama_output_caption_mask is not None:
            llama_output_caption_mask = llama_output_caption_mask.view(-1, llama_output_caption_mask.shape[-1])

        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            cross_output, _ = self._get_cross_output(visual_output, video_mask)

            diversity_weight = getattr(self.task_config, "qformer_diversity_weight", 0.0)
            diversity_loss = (
                self._compute_diversity_loss(cross_output)
                if (self.training and diversity_weight > 0.0)
                else 0.0
            )

            prefix_embeds, prefix_atts = self._build_llama_prefix_inputs(
                visual_output, video_mask, cross_output=cross_output
            )
            if self.training and getattr(self, "scst", False):
                alpha = getattr(self.task_config, "scst_alpha", 1.0)
                scst_loss = self._compute_scst_caption_loss(prefix_embeds, prefix_atts, output_caption_ids, llama_output_caption_ids,
                                                            llama_output_caption_mask, gt_refs=gt_refs)
                if alpha < 1.0:
                    xe_loss = self._compute_xe_caption_loss(prefix_embeds, prefix_atts, llama_output_caption_ids,
                                                            llama_output_caption_mask)
                    caption_loss = alpha * scst_loss + (1 - alpha) * xe_loss
                else:
                    caption_loss = scst_loss
            else:
                caption_loss = self._compute_xe_caption_loss(prefix_embeds, prefix_atts, llama_output_caption_ids,
                                                             llama_output_caption_mask)

            return caption_loss + diversity_weight * diversity_loss


    def init_corpus_cider(self, video_sentences_dict):
        """Initialize corpus-level CIDEr scorer with IDF from the full training set.

        Call this once after creating the dataloader so that SCST reward
        uses stable, corpus-level IDF statistics instead of noisy batch-level IDF.
        """
        from utils.cider_utils import CorpusCider
        self._cider_scorer = CorpusCider()
        self._cider_scorer.init_corpus_df(video_sentences_dict)
        logger.info("Corpus-level CIDEr scorer initialized for SCST training.")

    def _get_cider_scorer(self):
        if self._cider_scorer is None:
            # Fallback to batch-level CIDEr if corpus CIDEr was not initialized
            from pycocoevalcap.cider.cider import Cider
            logger.warning(
                "Using batch-level CIDEr (corpus CIDEr not initialized). "
                "Call model.init_corpus_cider(video_sentences_dict) for better IDF."
            )
            self._cider_scorer = Cider()
        return self._cider_scorer

    def _compute_scst_caption_loss(self, prefix_embeds, prefix_atts, output_caption_ids, llama_output_caption_ids=None,
                                   llama_output_caption_mask=None, gt_refs=None):
        """SCST loss: generate candidates under no_grad, then re-score with a
        differentiable teacher-forced forward pass to get valid gradients.
        """
        batch_size = prefix_embeds.size(0)
        pad_token_id = self.llama_tokenizer.pad_token_id

        with torch.no_grad():
            outputs = self.llama_model.generate(
                inputs_embeds=prefix_embeds,
                attention_mask=prefix_atts,
                do_sample=False,
                num_beams=self.scst_num_samples,
                max_new_tokens=self.max_txt_len,
                repetition_penalty=1.0,
                length_penalty=1.0,
                num_return_sequences=self.scst_num_samples,
                return_dict_in_generate=True,
                output_scores=False,
                pad_token_id=pad_token_id,
                eos_token_id=self.llama_tokenizer.eos_token_id,
            )
            generated_ids = outputs.sequences

        generated_ids = generated_ids.masked_fill(generated_ids.lt(0), pad_token_id)
        if self.llama_tokenizer.eos_token_id is None:
            generated_mask = generated_ids.ne(pad_token_id).long()
        else:
            eos_seen = generated_ids.eq(self.llama_tokenizer.eos_token_id).cumsum(dim=1)
            generated_mask = eos_seen.le(1).long()

        repeated_prefix_embeds = prefix_embeds.repeat_interleave(self.scst_num_samples, dim=0)
        repeated_prefix_atts = prefix_atts.repeat_interleave(self.scst_num_samples, dim=0)
        generated_embeds = self.llama_model.get_input_embeddings()(generated_ids)
        score_inputs_embeds = torch.cat([repeated_prefix_embeds, generated_embeds], dim=1)
        score_attention_mask = torch.cat([repeated_prefix_atts, generated_mask], dim=1)

        score_outputs = self.llama_model(
            inputs_embeds=score_inputs_embeds,
            attention_mask=score_attention_mask,
            return_dict=True,
        )
        prefix_ignore = torch.full(
            repeated_prefix_atts.shape,
            -100,
            dtype=generated_ids.dtype,
            device=generated_ids.device,
        )
        labels = torch.cat([prefix_ignore, generated_ids], dim=1)
        prefix_token_mask = torch.zeros_like(repeated_prefix_atts)
        token_mask = torch.cat([prefix_token_mask, generated_mask], dim=1)
        shift_logits = score_outputs.logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        labels_mask = shift_labels.ne(-100) & token_mask[:, 1:].bool()
        safe_labels = shift_labels.masked_fill(~labels_mask, 0)
        token_log_probs = F.log_softmax(shift_logits, dim=-1)
        selected_log_probs = token_log_probs.gather(
            dim=-1,
            index=safe_labels.unsqueeze(-1),
        ).squeeze(-1)

        selected_log_probs = selected_log_probs.masked_fill(~labels_mask, 0.0)
        output_length = labels_mask.sum(dim=1).clamp(min=1)
        sequences_scores = selected_log_probs.sum(dim=1) / output_length.float()
        sequences_scores = sequences_scores.view(batch_size, self.scst_num_samples)

        with torch.no_grad():
            caps_gen = self.llama_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            caps_gen = [t.strip() for t in caps_gen]

            if gt_refs is not None and len(gt_refs) == batch_size:
                caps_gt_repeated = []
                for sample_refs in gt_refs:
                    for _ in range(self.scst_num_samples):
                        caps_gt_repeated.append(sample_refs)
            else:
                gt_ids = llama_output_caption_ids if llama_output_caption_ids is not None else output_caption_ids
                gt_tokens = gt_ids.clone().masked_fill(gt_ids.lt(0), pad_token_id)
                if llama_output_caption_mask is not None:
                    gt_tokens = gt_tokens.masked_fill(llama_output_caption_mask.eq(0), pad_token_id)
                caps_gt = self.llama_tokenizer.batch_decode(gt_tokens, skip_special_tokens=True)
                caps_gt_repeated = [[c] for c in itertools.chain.from_iterable(
                    [c] * self.scst_num_samples for c in caps_gt
                )]

            caps_gt_tok, caps_gen_tok = tokenize(caps_gt_repeated, caps_gen)
            reward = self._get_cider_scorer().compute_score(caps_gt_tok, caps_gen_tok)[1].astype(np.float32)
            reward = torch.from_numpy(reward).to(prefix_embeds.device).view(batch_size, self.scst_num_samples)

            reward_baseline = torch.mean(reward, dim=-1, keepdim=True)
            advantage = reward - reward_baseline

        loss = -(sequences_scores) * advantage.detach()
        return loss.mean()

    def generate_caption_ids(self, visual_output, video_mask, num_beams=None, max_length=None):
        if num_beams is None:
            num_beams = max(1, getattr(self, "eval_beam_size", getattr(self, "beam_size", 1)))
        if max_length is None:
            max_length = getattr(self, "max_txt_len", 32)

        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            prefix_embeds, prefix_atts = self._build_llama_prefix_inputs(
                visual_output, video_mask
            )
            outputs = self.llama_model.generate(
                inputs_embeds=prefix_embeds,
                attention_mask=prefix_atts,
                do_sample=False,
                num_beams=num_beams,
                max_new_tokens=max_length,
                repetition_penalty=1.2,
                length_penalty=1.0,
                pad_token_id=self.llama_tokenizer.pad_token_id,
                eos_token_id=self.llama_tokenizer.eos_token_id,
            )

        return outputs

    def generate_caption_text(self, visual_output, video_mask, num_beams=None, max_length=None):
        output_ids = self.generate_caption_ids(
            visual_output, video_mask, num_beams=num_beams, max_length=max_length
        )
        captions = self.llama_tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        return [caption.strip() for caption in captions]

    def _mean_pooling_for_similarity(self, sequence_output, visual_output, attention_mask, video_mask,):
        attention_mask_un = attention_mask.to(dtype=torch.float).unsqueeze(-1)
        attention_mask_un[:, 0, :] = 0.
        sequence_output = sequence_output * attention_mask_un
        text_out = torch.sum(sequence_output, dim=1) / torch.sum(attention_mask_un, dim=1, dtype=torch.float)

        video_mask_un = video_mask.to(dtype=torch.float).unsqueeze(-1)
        visual_output = visual_output * video_mask_un
        video_mask_un_sum = torch.sum(video_mask_un, dim=1, dtype=torch.float)
        video_mask_un_sum = torch.clamp(video_mask_un_sum, min=1.0)
        video_out = torch.sum(visual_output, dim=1) / video_mask_un_sum

        return text_out, video_out

    def _cross_similarity(self, sequence_output, visual_output, attention_mask, video_mask):
        b_text, _, _ = sequence_output.size()
        b_visual, s_visual, h_visual = visual_output.size()

        retrieve_logits_list = []
        step_size = 5

        split_size = [step_size] * (b_text // step_size)
        release_size = b_text - sum(split_size)
        if release_size > 0:
            split_size += [release_size]

        sequence_output_splits = torch.split(sequence_output, split_size, dim=0)
        for i in range(len(split_size)):
            sequence_output_row = sequence_output_splits[i]

            step_truth = sequence_output_row.size(0)
            visual_output_r = visual_output.unsqueeze(0).repeat(step_truth, 1, 1, 1)
            visual_output_r = visual_output_r.view(-1, s_visual, h_visual)
            video_mask_r = video_mask.unsqueeze(0).repeat(step_truth, 1, 1)
            video_mask_r = video_mask_r.view(-1, s_visual)

            _, pooled_output = self._get_cross_output(visual_output_r, video_mask_r)
            retrieve_logits_row = self.similarity_dense(pooled_output).squeeze(-1).view(step_truth, b_visual)

            retrieve_logits_list.append(retrieve_logits_row)
        retrieve_logits = torch.cat(retrieve_logits_list, dim=0)
        return retrieve_logits

    def get_similarity_logits(self, sequence_output, visual_output, attention_mask, video_mask, shaped=False, _pretrain_joint=False):
        if shaped is False:
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])

        if (self._stage_two and _pretrain_joint is False) or self.train_sim_after_cross:
            retrieve_logits = self._cross_similarity(sequence_output, visual_output, attention_mask, video_mask)
        else:
            text_out, video_out = self._mean_pooling_for_similarity(sequence_output, visual_output, attention_mask, video_mask)
            if self.task_config.use_mil is False:
                text_out = F.normalize(text_out, dim=-1)
                video_out = F.normalize(video_out, dim=-1)
            retrieve_logits = torch.matmul(text_out, video_out.t())

        return retrieve_logits

    def _get_decoder_score(self, visual_output, video_mask, input_caption_ids, decoder_mask, shaped=False):

        if shaped is False:
            video_mask = video_mask.view(-1, video_mask.shape[-1])

            input_caption_ids = input_caption_ids.view(-1, input_caption_ids.shape[-1])
            decoder_mask = decoder_mask.view(-1, decoder_mask.shape[-1])

        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            prefix_embeds, prefix_atts = self._build_llama_prefix_inputs(
                visual_output, video_mask
            )

            pad_token_id = self.llama_tokenizer.pad_token_id
            decoder_input_ids = input_caption_ids.clone().masked_fill(input_caption_ids.lt(0), pad_token_id)
            if decoder_mask is not None:
                decoder_att_mask = decoder_mask.long()
            else:
                decoder_att_mask = decoder_input_ids.ne(pad_token_id).long()

            caption_embeds = self.llama_model.get_input_embeddings()(decoder_input_ids)
            inputs_embeds = torch.cat([prefix_embeds, caption_embeds], dim=1)
            attention_mask = torch.cat([prefix_atts, decoder_att_mask], dim=1)
            outputs = self.llama_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
            )
            decoder_scores = outputs.logits[:, -decoder_input_ids.size(1):, :]

        return decoder_scores

    def decoder_caption(self, sequence_output, visual_output, input_ids, attention_mask, video_mask, input_caption_ids, decoder_mask,
                        shaped=False, get_logits=False):
        if shaped is False:
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])

            input_caption_ids = input_caption_ids.view(-1, input_caption_ids.shape[-1])
            decoder_mask = decoder_mask.view(-1, decoder_mask.shape[-1])

        decoder_scores = self._get_decoder_score(visual_output,
                             video_mask,
                             input_caption_ids, decoder_mask, shaped=True)

        if get_logits:
            return decoder_scores

        _, decoder_scores_result = torch.max(decoder_scores, -1)

        return decoder_scores_result
