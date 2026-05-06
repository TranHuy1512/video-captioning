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

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss


from modules.until_module import PreTrainedModel, LayerNorm, CrossEn, MILNCELoss, MaxMarginRankingLoss
from modules.module_bert import BertModel, BertConfig, BertOnlyMLMHead
from modules.module_visual import VisualModel, VisualConfig, VisualOnlyMLMHead
from modules.module_cross import CrossModel, CrossConfig
from modules.module_decoder import DecoderConfig, DecoderModel
from modules.blip2 import Blip2Base
from modules.beam import Beam
from modules.tokenization import BertTokenizer

logger = logging.getLogger(__name__)


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
            model = cls.init_preweight(model, state_dict, task_config=task_config)

        return model

    @staticmethod
    def _filter_init_model_state_dict(state_dict, task_config=None):
        allowed_prefixes = (
            "bert.",
            "visual.",
            "cross.",
            "decoder.",
            "Qformer.",
            "query_tokens",
            "qformer_visual_proj.",
            "similarity_dense.",
            "cls.",
            "cls_visual.",
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

        self.tokenizer = BertTokenizer.from_pretrained(
            task_config.bert_model,
            do_lower_case=getattr(task_config, "do_lower_case", False),
        )

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
        bert_position_embeddings_weight = self.bert.embeddings.position_embeddings.weight
        # <=== End of Text Encoder

        # Video Encoder ===>
        visual_config = update_attr("visual_config", visual_config, "num_hidden_layers",
                                    self.task_config, "visual_num_hidden_layers")
        self.visual = VisualModel(visual_config)
        self.freeze_vit = getattr(self.task_config, "freeze_vit", False)
        if self.freeze_vit:
            for param in self.visual.parameters():
                param.requires_grad = False
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
            # for layer in self.Qformer.bert.encoder.layer:
            #     layer.output = None
            #     layer.intermediate = None
            # <=== End of Cross Encoder

            if self.train_sim_after_cross is False:
                # Decoder ===>
                self.eval_beam_size = getattr(
                    self.task_config,
                    "eval_beam_size",
                    getattr(self.task_config, "beam_size", 5),
                )
                self.beam_size = self.eval_beam_size

                decoder_config = update_attr(
                    "decoder_config",
                    decoder_config,
                    "num_decoder_layers",
                    self.task_config,
                    "decoder_num_hidden_layers",
                )
                self.decoder = DecoderModel(
                    decoder_config,
                    bert_word_embeddings_weight,
                    bert_position_embeddings_weight,
                )
                if self.Qformer.config.hidden_size != decoder_config.hidden_size:
                    self.decoder_cross_proj = nn.Linear(self.Qformer.config.hidden_size, decoder_config.hidden_size)
                    show_log(
                        task_config,
                        "Add decoder cross projection: {} -> {}.".format(
                            self.Qformer.config.hidden_size, decoder_config.hidden_size
                        )
                    )
                else:
                    self.decoder_cross_proj = nn.Identity()
                # <=== End of Decoder

            if self.task_config.do_pretrain:
                self.cls = BertOnlyMLMHead(bert_config, bert_word_embeddings_weight)
                self.cls_visual = VisualOnlyMLMHead(visual_config, visual_word_embeddings_weight)
                self.alm_loss_fct = CrossEntropyLoss(ignore_index=-1)
                
            self.similarity_dense = nn.Linear(bert_config.hidden_size, 1)
            # Decoder labels are padded with 0 ([PAD]) in dataloaders, so ignore 0 for CE loss.
            self.decoder_loss_fct = CrossEntropyLoss(ignore_index=0)

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
        
        self._init_weights_except_pretrained_submodules()

    def _init_weights_except_pretrained_submodules(self):
        skip_roots = {"Qformer", "qformer_visual_proj", "query_tokens", "normalize_video"}

        def init_module(module):
            for name, child in module._modules.items():
                if child is None or name in skip_roots:
                    continue
                init_module(child)
            self.init_weights(module)

        init_module(self)

    def forward(self, input_ids, token_type_ids, attention_mask, video, video_mask=None,
                pairs_masked_text=None, pairs_token_labels=None, masked_video=None, video_labels_index=None,
                input_caption_ids=None, decoder_mask=None, output_caption_ids=None):

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
                        decoder_scores = self._get_decoder_score(
                            visual_output_alm,
                            video_mask,
                            input_caption_ids,
                            decoder_mask,
                            shaped=True,
                        )
                    elif self.task_config.task_type == "caption":
                        decoder_scores = self._get_decoder_score(
                            visual_output,
                            video_mask,
                            input_caption_ids,
                            decoder_mask,
                            shaped=True,
                        )
                    else:
                        raise NotImplementedError

                    output_caption_ids = output_caption_ids.view(-1, output_caption_ids.shape[-1])
                    decoder_loss = self.decoder_loss_fct(
                        decoder_scores.view(-1, self.bert_config.vocab_size),
                        output_caption_ids.view(-1),
                    )
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
                decoder_scores = self._get_decoder_score(
                    visual_output,
                    video_mask,
                    input_caption_ids,
                    decoder_mask,
                    shaped=True,
                )
                output_caption_ids = output_caption_ids.view(-1, output_caption_ids.shape[-1])
                decoder_loss = self.decoder_loss_fct(
                    decoder_scores.view(-1, self.bert_config.vocab_size),
                    output_caption_ids.view(-1),
                )
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
        query_mask = torch.ones(
            cross_output.size()[:-1],
            dtype=torch.long,
            device=cross_output.device,
        )

        return cross_output, pooled_output, query_mask


    def generate_caption_ids(self, visual_output, video_mask, num_beams=None, max_length=None):
        if num_beams is None:
            num_beams = max(1, getattr(self, "eval_beam_size", getattr(self, "beam_size", 1)))
        if max_length is None:
            max_length = getattr(self.task_config, "max_words", 32)

        device = visual_output.device
        n_inst, len_v, v_h = visual_output.size()
        n_bm = num_beams

        def get_inst_idx_to_tensor_position_map(inst_idx_list):
            return {inst_idx: tensor_position for tensor_position, inst_idx in enumerate(inst_idx_list)}

        def collect_active_part(beamed_tensor, curr_active_inst_idx, n_prev_active_inst, n_bm):
            _, *d_hs = beamed_tensor.size()
            n_curr_active_inst = len(curr_active_inst_idx)
            new_shape = (n_curr_active_inst * n_bm, *d_hs)

            beamed_tensor = beamed_tensor.view(n_prev_active_inst, -1)
            beamed_tensor = beamed_tensor.index_select(0, curr_active_inst_idx)
            beamed_tensor = beamed_tensor.view(*new_shape)
            return beamed_tensor

        def collate_active_info(visual_output_rpt, video_mask_rpt, inst_idx_to_position_map, active_inst_idx_list, n_bm):
            n_prev_active_inst = len(inst_idx_to_position_map)
            active_inst_idx = [inst_idx_to_position_map[k] for k in active_inst_idx_list]
            active_inst_idx = torch.LongTensor(active_inst_idx).to(device)

            visual_output_rpt = collect_active_part(visual_output_rpt, active_inst_idx, n_prev_active_inst, n_bm)
            video_mask_rpt = collect_active_part(video_mask_rpt, active_inst_idx, n_prev_active_inst, n_bm)
            active_inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)
            return visual_output_rpt, video_mask_rpt, active_inst_idx_to_position_map

        def prepare_beam_dec_seq(inst_dec_beams, len_dec_seq):
            dec_partial_seq = [b.get_current_state() for b in inst_dec_beams if not b.done]
            dec_partial_seq = torch.stack(dec_partial_seq).to(device)
            dec_partial_seq = dec_partial_seq.view(-1, len_dec_seq)
            return dec_partial_seq

        def collect_active_inst_idx_list(inst_beams, word_prob, inst_idx_to_position_map):
            active_inst_idx_list = []
            for inst_idx, inst_position in inst_idx_to_position_map.items():
                is_inst_complete = inst_beams[inst_idx].advance(word_prob[inst_position])
                if not is_inst_complete:
                    active_inst_idx_list.append(inst_idx)
            return active_inst_idx_list

        def collect_hypothesis(inst_dec_beams):
            results = []
            for inst_idx in range(len(inst_dec_beams)):
                _, tail_idxs = inst_dec_beams[inst_idx].sort_scores()
                results.append(inst_dec_beams[inst_idx].get_hypothesis(tail_idxs[0]))
            return results

        visual_output_rpt = visual_output.repeat(1, n_bm, 1).view(n_inst * n_bm, len_v, v_h)
        video_mask_rpt = video_mask.repeat(1, n_bm).view(n_inst * n_bm, len_v)

        inst_dec_beams = [Beam(n_bm, device=device, tokenizer=self.tokenizer) for _ in range(n_inst)]
        active_inst_idx_list = list(range(n_inst))
        inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)

        for len_dec_seq in range(1, max_length + 1):
            dec_seq = prepare_beam_dec_seq(inst_dec_beams, len_dec_seq)
            next_decoder_mask = torch.ones(dec_seq.size(), dtype=torch.uint8, device=device)

            dec_output = self.decoder_caption(
                visual_output_rpt,
                video_mask_rpt,
                dec_seq,
                next_decoder_mask,
                shaped=True,
                get_logits=True,
            )
            dec_output = dec_output[:, -1, :]
            word_prob = torch.nn.functional.log_softmax(dec_output, dim=1)
            word_prob = word_prob.view(len(active_inst_idx_list), n_bm, -1)

            active_inst_idx_list = collect_active_inst_idx_list(inst_dec_beams, word_prob, inst_idx_to_position_map)
            if not active_inst_idx_list:
                break

            visual_output_rpt, video_mask_rpt, inst_idx_to_position_map = collate_active_info(
                visual_output_rpt,
                video_mask_rpt,
                inst_idx_to_position_map,
                active_inst_idx_list,
                n_bm,
            )

        hypotheses = collect_hypothesis(inst_dec_beams)
        pad_id = self.tokenizer.vocab.get("[PAD]", 0)
        output_ids = torch.full((n_inst, max_length), pad_id, dtype=torch.long, device=device)
        for idx, hyp in enumerate(hypotheses):
            if not hyp:
                continue
            length = min(len(hyp), max_length)
            output_ids[idx, :length] = torch.tensor(hyp[:length], device=device)

        return output_ids

    def generate_caption_text(self, visual_output, video_mask, num_beams=None, max_length=None):
        output_ids = self.generate_caption_ids(
            visual_output, video_mask, num_beams=num_beams, max_length=max_length
        )
        captions = []
        for token_ids in output_ids.tolist():
            captions.append(" ".join(self.tokenizer.convert_ids_to_tokens(token_ids)).strip())
        return captions

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

            _, pooled_output, _ = self._get_cross_output(
                visual_output_r,
                video_mask_r,
                num_query_token=self.num_query_token,
            )
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

        decoder_input_ids = input_caption_ids.clone().masked_fill(input_caption_ids.lt(0), 0)
        decoder_att_mask = decoder_mask.long() if decoder_mask is not None else decoder_input_ids.ne(0).long()

        cross_output, _, query_mask = self._get_cross_output(
            visual_output,
            video_mask,
            num_query_token=self.num_query_token,
        )
        cross_output = self.decoder_cross_proj(cross_output)
        decoder_scores = self.decoder(
            decoder_input_ids,
            encoder_outs=cross_output,
            answer_mask=decoder_att_mask,
            encoder_mask=query_mask,
        )

        return decoder_scores

    def decoder_caption(self, visual_output, video_mask, input_caption_ids, decoder_mask,
                        shaped=False, get_logits=False):
        if shaped is False:
            video_mask = video_mask.view(-1, video_mask.shape[-1])
            input_caption_ids = input_caption_ids.view(-1, input_caption_ids.shape[-1])
            decoder_mask = decoder_mask.view(-1, decoder_mask.shape[-1])

        decoder_scores = self._get_decoder_score(
            visual_output,
            video_mask,
            input_caption_ids,
            decoder_mask,
            shaped=True,
        )

        if get_logits:
            return decoder_scores

        _, decoder_scores_result = torch.max(decoder_scores, -1)
        return decoder_scores_result
