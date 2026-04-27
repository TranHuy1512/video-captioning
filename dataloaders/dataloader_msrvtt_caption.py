from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
from torch.utils.data import Dataset
import torch
import numpy as np
import pickle
import pandas as pd
from collections import defaultdict
import json
import random


class NumpyCompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        return super(NumpyCompatUnpickler, self).find_class(module, name)


def load_pickle_with_numpy_compat(reader):
    try:
        return pickle.load(reader)
    except ModuleNotFoundError as error:
        if error.name is None or not error.name.startswith("numpy._core"):
            raise
        reader.seek(0)
        return NumpyCompatUnpickler(reader).load()


class MSRVTT_Caption_DataLoader(Dataset):
    """MSRVTT train dataset loader."""
    def __init__(
            self,
            csv_path,
            json_path,
            features_path,
            tokenizer,
            max_words=30,
            feature_framerate=1.0,
            max_frames=100,
            split_type="",
            t5_tokenizer=None,
            max_txt_len=32,
            scst=False,
    ):
        self.csv = pd.read_csv(csv_path)
        self.data = json.load(open(json_path, 'r'))
        self.features_path = features_path
        self.features_are_file_folder = os.path.isdir(features_path)
        self.feature_file_index = self._build_feature_file_index(features_path) if self.features_are_file_folder else {}
        if self.features_are_file_folder:
            self.feature_dict = None
        else:
            with open(features_path, 'rb') as reader:
                self.feature_dict = load_pickle_with_numpy_compat(reader)
        self.feature_framerate = feature_framerate
        self.max_words = max_words
        self.max_frames = max_frames
        self.tokenizer = tokenizer
        self.t5_tokenizer = t5_tokenizer
        self.max_txt_len = max_txt_len

        self.scst = scst
        self.feature_size = self._infer_feature_size()

        assert split_type in ["train", "val", "test"]
        # Train: video0 : video6512 (6513)
        # Val: video6513 : video7009 (497)
        # Test: video7010 : video9999 (2990)
        video_ids = [self.data['videos'][idx]['video_id'] for idx in range(len(self.data['videos']))]
        split_dict = {"train": video_ids[:6513], "val": video_ids[6513:6513 + 497], "test": video_ids[6513 + 497:]}
        choiced_video_ids = split_dict[split_type]

        self.sample_len = 0
        self.sentences_dict = {}
        self.video_sentences_dict = defaultdict(list)
        if split_type == "train" and scst:
            # SCST mode: one sample per video (6,513); caption is randomly
            # picked in __getitem__ from video_sentences_dict each time.
            for itm in self.data['sentences']:
                if itm['video_id'] in choiced_video_ids:
                    self.video_sentences_dict[itm['video_id']].append(itm['caption'])
            for vid in choiced_video_ids:
                self.sentences_dict[len(self.sentences_dict)] = (vid, None)
        elif split_type == "train":  # XE mode: expand all sentences
            for itm in self.data['sentences']:
                if itm['video_id'] in choiced_video_ids:
                    self.sentences_dict[len(self.sentences_dict)] = (itm['video_id'], itm['caption'])
                    self.video_sentences_dict[itm['video_id']].append(itm['caption'])
        elif split_type == "val" or split_type == "test":
            for itm in self.data['sentences']:
                if itm['video_id'] in choiced_video_ids:
                    self.video_sentences_dict[itm['video_id']].append(itm['caption'])
            for vid in choiced_video_ids:
                self.sentences_dict[len(self.sentences_dict)] = (vid, self.video_sentences_dict[vid][0])
        else:
            raise NotImplementedError

        self.sample_len = len(self.sentences_dict)

    def __len__(self):
        return self.sample_len

    def _build_feature_file_index(self, features_path):
        feature_file_index = defaultdict(dict)
        feature_extensions = {".pickle", ".pkl", ".pt"}
        for root, _, files in os.walk(features_path):
            for file_name in files:
                video_id, extension = os.path.splitext(file_name)
                if extension in feature_extensions:
                    feature_file_index[video_id][extension] = os.path.join(root, file_name)
        return feature_file_index

    def _pt_feature_path(self, video_id):
        indexed_file = self.feature_file_index.get(video_id, {}).get(".pt")
        if indexed_file is not None:
            return indexed_file
        return os.path.join(self.features_path, "{}.pt".format(video_id))

    def _pickle_feature_path(self, video_id):
        for extension in (".pickle", ".pkl"):
            indexed_file = self.feature_file_index.get(video_id, {}).get(extension)
            if indexed_file is not None:
                return indexed_file
            direct_file = os.path.join(self.features_path, "{}{}".format(video_id, extension))
            if os.path.exists(direct_file):
                return direct_file
        return os.path.join(self.features_path, "{}.pickle".format(video_id))

    def _tensor_from_pt_object(self, obj, video_id):
        if torch.is_tensor(obj):
            return obj
        if isinstance(obj, dict):
            for key in ("features", "feature", "video", "video_features", "embeddings"):
                value = obj.get(key)
                if torch.is_tensor(value):
                    return value
        raise TypeError("Unsupported .pt feature format for {}.".format(video_id))

    def _array_from_pickle_object(self, obj, video_id):
        if torch.is_tensor(obj):
            return obj.float().numpy()
        if isinstance(obj, np.ndarray):
            return obj.astype(np.float32, copy=False)
        if isinstance(obj, dict):
            if video_id in obj:
                return self._array_from_pickle_object(obj[video_id], video_id)
            for key in ("features", "feature", "video", "video_features", "embeddings"):
                if key in obj:
                    return self._array_from_pickle_object(obj[key], video_id)
        raise TypeError("Unsupported .pickle feature format for {}.".format(video_id))

    def _load_pt_feature(self, video_id):
        feature_file = self._pt_feature_path(video_id)
        if not os.path.exists(feature_file):
            raise FileNotFoundError("Missing feature file: {}".format(feature_file))
        feature = self._tensor_from_pt_object(
            torch.load(feature_file, map_location="cpu"),
            video_id,
        )
        if feature.dim() == 3 and feature.size(0) == 1:
            feature = feature.squeeze(0)
        if feature.dim() != 2:
            raise ValueError(
                "Expected .pt feature for {} to have shape (T, D), got {}.".format(
                    video_id, tuple(feature.shape)
                )
            )
        return feature.float().numpy()

    def _load_pickle_feature_file(self, video_id):
        feature_file = self._pickle_feature_path(video_id)
        if not os.path.exists(feature_file):
            raise FileNotFoundError("Missing feature file: {}".format(feature_file))
        with open(feature_file, "rb") as reader:
            feature = self._array_from_pickle_object(load_pickle_with_numpy_compat(reader), video_id)
        if feature.ndim == 3 and feature.shape[0] == 1:
            feature = feature.squeeze(0)
        if feature.ndim != 2:
            raise ValueError(
                "Expected .pickle feature for {} to have shape (T, D), got {}.".format(
                    video_id, tuple(feature.shape)
                )
            )
        return feature

    def _load_feature(self, video_id):
        if self.features_are_file_folder:
            pickle_feature_file = self._pickle_feature_path(video_id)
            if os.path.exists(pickle_feature_file):
                return self._load_pickle_feature_file(video_id)
            return self._load_pt_feature(video_id)
        return self.feature_dict[video_id]

    def _infer_feature_size(self):
        if not self.features_are_file_folder:
            return self.feature_dict[self.csv['video_id'].values[0]].shape[-1]

        candidate_video_ids = list(self.csv['video_id'].values)
        candidate_video_ids.extend(
            video.get('video_id') for video in self.data.get('videos', []) if video.get('video_id')
        )
        for video_id in candidate_video_ids:
            pickle_feature_file = self._pickle_feature_path(video_id)
            if os.path.exists(pickle_feature_file):
                return self._load_pickle_feature_file(video_id).shape[-1]
            pt_feature_file = self._pt_feature_path(video_id)
            if os.path.exists(pt_feature_file):
                return self._load_pt_feature(video_id).shape[-1]

        raise FileNotFoundError(
            "Could not infer feature size: no .pickle, .pkl, or .pt files found in {}".format(self.features_path)
        )

    def _get_text(self, video_id, caption=None):
        k = 1
        choice_video_ids = [video_id]
        pairs_text = np.zeros((k, self.max_words), dtype=np.int64)
        pairs_mask = np.zeros((k, self.max_words), dtype=np.int64)
        pairs_segment = np.zeros((k, self.max_words), dtype=np.int64)
        pairs_masked_text = np.zeros((k, self.max_words), dtype=np.int64)
        pairs_token_labels = np.zeros((k, self.max_words), dtype=np.int64)

        pairs_input_caption_ids = np.zeros((k, self.max_words), dtype=np.int64)
        pairs_output_caption_ids = np.zeros((k, self.max_words), dtype=np.int64)
        pairs_decoder_mask = np.zeros((k, self.max_words), dtype=np.int64)

        # T5-tokenized ground truth for SCST training
        t5_max_len = self.max_txt_len if self.t5_tokenizer is not None else self.max_words
        pairs_t5_output_caption_ids = np.zeros((k, t5_max_len), dtype=np.int64)

        for i, video_id in enumerate(choice_video_ids):
            words = []
            words = ["[CLS]"] + words
            total_length_with_CLS = self.max_words - 1
            if len(words) > total_length_with_CLS:
                words = words[:total_length_with_CLS]
            words = words + ["[SEP]"]

            # Mask Language Model <-----
            token_labels = []
            masked_tokens = words.copy()
            for token_id, token in enumerate(masked_tokens):
                if token_id == 0 or token_id == len(masked_tokens) - 1:
                    token_labels.append(-1)
                    continue
                prob = random.random()
                # mask token with 15% probability
                if prob < 0.15:
                    prob /= 0.15
                    # 80% randomly change token to mask token
                    if prob < 0.8:
                        masked_tokens[token_id] = "[MASK]"
                    # 10% randomly change token to random token
                    elif prob < 0.9:
                        masked_tokens[token_id] = random.choice(list(self.tokenizer.vocab.items()))[0]
                    # -> rest 10% randomly keep current token
                    # append current token to output (we will predict these later)
                    try:
                        token_labels.append(self.tokenizer.vocab[token])
                    except KeyError:
                        # For unknown words (should not occur with BPE vocab)
                        token_labels.append(self.tokenizer.vocab["[UNK]"])
                        # print("Cannot find token '{}' in vocab. Using [UNK] insetad".format(token))
                else:
                    # no masking token (will be ignored by loss function later)
                    token_labels.append(-1)
            # -----> Mask Language Model

            input_ids = self.tokenizer.convert_tokens_to_ids(words)
            input_mask = [1] * len(input_ids)
            segment_ids = [0] * len(input_ids)
            masked_token_ids = self.tokenizer.convert_tokens_to_ids(masked_tokens)
            while len(input_ids) < self.max_words:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
                masked_token_ids.append(0)
                token_labels.append(-1)
            assert len(input_ids) == self.max_words
            assert len(input_mask) == self.max_words
            assert len(segment_ids) == self.max_words
            assert len(masked_token_ids) == self.max_words
            assert len(token_labels) == self.max_words

            pairs_text[i] = np.array(input_ids)
            pairs_mask[i] = np.array(input_mask)
            pairs_segment[i] = np.array(segment_ids)
            pairs_masked_text[i] = np.array(masked_token_ids)
            pairs_token_labels[i] = np.array(token_labels)

            # For generate captions
            if caption is not None:
                caption_words = self.tokenizer.tokenize(caption)
            else:
                caption_words = self._get_single_text(video_id)
            if len(caption_words) > total_length_with_CLS:
                caption_words = caption_words[:total_length_with_CLS]
            input_caption_words = ["[CLS]"] + caption_words
            output_caption_words = caption_words + ["[SEP]"]

            # For generate captions
            input_caption_ids = self.tokenizer.convert_tokens_to_ids(input_caption_words)
            output_caption_ids = self.tokenizer.convert_tokens_to_ids(output_caption_words)
            decoder_mask = [1] * len(input_caption_ids)
            while len(input_caption_ids) < self.max_words:
                input_caption_ids.append(0)
                output_caption_ids.append(0)
                decoder_mask.append(0)
            assert len(input_caption_ids) == self.max_words
            assert len(output_caption_ids) == self.max_words
            assert len(decoder_mask) == self.max_words

            pairs_input_caption_ids[i] = np.array(input_caption_ids)
            pairs_output_caption_ids[i] = np.array(output_caption_ids)
            pairs_decoder_mask[i] = np.array(decoder_mask)

            # T5-tokenize the raw caption text for SCST ground truth
            if self.t5_tokenizer is not None:
                raw_caption = caption if caption is not None else ""
                t5_tokens = self.t5_tokenizer(
                    raw_caption,
                    padding="max_length",
                    truncation=True,
                    max_length=t5_max_len,
                    return_tensors="np",
                )
                pairs_t5_output_caption_ids[i] = t5_tokens.input_ids[0]

        return pairs_text, pairs_mask, pairs_segment, pairs_masked_text, pairs_token_labels, \
               pairs_input_caption_ids, pairs_decoder_mask, pairs_output_caption_ids, choice_video_ids, \
               pairs_t5_output_caption_ids

    # def _get_single_text(self, video_id):
    #     rind = random.randint(0, len(self.sentences[video_id]) - 1)
    #     caption = self.sentences[video_id][rind]
    #     words = self.tokenizer.tokenize(caption)
    #     return words
    def _get_single_text(self, video_id):
        captions = self.video_sentences_dict[video_id]
        rind = random.randint(0, len(captions) - 1)
        caption = captions[rind]
        return self.tokenizer.tokenize(caption)

    def _get_video(self, choice_video_ids):
        video_mask = np.zeros((len(choice_video_ids), self.max_frames), dtype=np.int64)
        max_video_length = [0] * len(choice_video_ids)

        video = np.zeros((len(choice_video_ids), self.max_frames, self.feature_size), dtype=np.float32)
        for i, video_id in enumerate(choice_video_ids):
            video_slice = self._load_feature(video_id)

            if self.max_frames < video_slice.shape[0]:
                # Preserve information from the full sequence instead of
                # taking only the first max_frames entries.
                # This is important when feature files contain long token/frame
                # sequences (e.g., fused features with T >> max_frames).
                chunks = np.array_split(video_slice, self.max_frames, axis=0)
                video_slice = np.stack([chunk.mean(axis=0) for chunk in chunks], axis=0)

            slice_shape = video_slice.shape
            max_video_length[i] = max_video_length[i] if max_video_length[i] > slice_shape[0] else slice_shape[0]
            if len(video_slice) < 1:
                print("video_id: {}".format(video_id))
            else:
                video[i][:slice_shape[0]] = video_slice

        for i, v_length in enumerate(max_video_length):
            video_mask[i][:v_length] = [1] * v_length

        # Mask Frame Model <-----
        video_labels_index = [[] for _ in range(len(choice_video_ids))]
        masked_video = video.copy()
        for i, video_pair_ in enumerate(masked_video):
            for j, _ in enumerate(video_pair_):
                if j < max_video_length[i]:
                    prob = random.random()
                    # mask token with 15% probability
                    if prob < 0.15:
                        masked_video[i][j] = [0.] * video.shape[-1]
                        video_labels_index[i].append(j)
                    else:
                        video_labels_index[i].append(-1)
                else:
                    video_labels_index[i].append(-1)
        video_labels_index = np.array(video_labels_index, dtype=np.int64)
        # -----> Mask Frame Model

        return video, video_mask, masked_video, video_labels_index

    def get_all_refs_for_video(self, video_id):
        """Return all reference captions for a given video_id."""
        return self.video_sentences_dict.get(video_id, [])

    def __getitem__(self, idx):
        video_id, caption = self.sentences_dict[idx]
        # SCST mode: randomly pick one caption per video each access
        if caption is None and self.scst:
            captions = self.video_sentences_dict[video_id]
            caption = random.choice(captions)

        pairs_text, pairs_mask, pairs_segment, \
        pairs_masked_text, pairs_token_labels, \
        pairs_input_caption_ids, pairs_decoder_mask, \
        pairs_output_caption_ids, choice_video_ids, \
        pairs_t5_output_caption_ids = self._get_text(video_id, caption)

        video, video_mask, masked_video, video_labels_index = self._get_video(choice_video_ids)

        return pairs_text, pairs_mask, pairs_segment, video, video_mask, \
               pairs_masked_text, pairs_token_labels, masked_video, video_labels_index, \
               pairs_input_caption_ids, pairs_decoder_mask, pairs_output_caption_ids, \
               pairs_t5_output_caption_ids, idx
