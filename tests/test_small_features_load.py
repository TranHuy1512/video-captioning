import os
import sys
import unittest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from dataloaders.dataloader_msrvtt_caption import MSRVTT_Caption_DataLoader


class DummyTokenizer:
    vocab = {"[UNK]": 0}

    def tokenize(self, text):
        return text.split()

    def convert_tokens_to_ids(self, tokens):
        return [0 for _ in tokens]


class TestSmallFeaturesLoad(unittest.TestCase):
    def test_small_features_with_small_dataset_split(self):
        features_path = os.environ.get(
            "SMALL_FEATURES_PATH",
            "/Users/mac/Documents/NCKH/checkpoint_zip/small_features.pickle",
        )
        if not os.path.exists(features_path):
            self.skipTest("Missing small_features.pickle; set SMALL_FEATURES_PATH to run.")

        csv_path = os.path.join(REPO_ROOT, "data", "msrvtt", "MSRVTT_train.9k.csv")
        json_path = os.path.join(REPO_ROOT, "data", "msrvtt", "MSRVTT_data.json")

        loader = MSRVTT_Caption_DataLoader(
            csv_path=csv_path,
            json_path=json_path,
            features_path=features_path,
            tokenizer=DummyTokenizer(),
            max_words=20,
            max_frames=100,
            split_type="train",
            use_small_dataset=True,
        )

        self.assertGreater(len(loader), 0)


if __name__ == "__main__":
    unittest.main()
