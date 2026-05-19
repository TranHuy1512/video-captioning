import re


def normalize_msrvtt_feature_dict(feature_dict, expected_video_ids):
    """Normalize MSRVTT feature dict keys to match video ids like 'video123'."""
    if not isinstance(feature_dict, dict):
        return feature_dict
    if expected_video_ids is None:
        return feature_dict

    try:
        expected_len = len(expected_video_ids)
    except TypeError:
        expected_len = 0
    if expected_len == 0:
        return feature_dict

    if any(key in expected_video_ids for key in feature_dict.keys()):
        return feature_dict

    expected_set = set(expected_video_ids)

    def normalize_key(key):
        key_str = str(key)
        if key_str.startswith("video"):
            return key_str
        if re.fullmatch(r"\d+", key_str):
            return "video" + key_str
        return key_str

    normalized = {}
    for key, value in feature_dict.items():
        normalized_key = normalize_key(key)
        if normalized_key not in normalized:
            normalized[normalized_key] = value

    if any(key in expected_set for key in normalized.keys()):
        return normalized

    return feature_dict
