"""Corpus-level diversity / mode-collapse diagnostics for generated captions.

Pure stdlib (no torch / pycocoevalcap dependency) so it runs anywhere,
including a laptop with no GPU. Point it at a `hyp.txt` (one caption per
line, same format `inference/caption_generator.py` writes) to check whether
the model is falling back to a small set of generic templates instead of
describing each video specifically.

Usage:
    python -m utils.diversity_utils --hyp evidents/hyp.txt
    python -m utils.diversity_utils --hyp new/hyp.txt --compare evidents/hyp.txt
"""

import argparse
import json
import math
import random
import re
from collections import Counter
from typing import Dict, List, Sequence


def load_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def ngrams(tokens: Sequence[str], n: int) -> List[tuple]:
    if len(tokens) < n:
        return []
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def ngram_set(tokens: Sequence[str], n: int) -> set:
    return set(ngrams(tokens, n))


def batch_diversity_overlap(
    cand_tokens: Sequence[str],
    other_tokens_list: Sequence[Sequence[str]],
    orders: Sequence[int] = (1, 2),
) -> float:
    """Max, over `other_tokens_list`, of the average n-gram overlap ratio
    (fraction of the candidate's own n-grams that also appear in one other
    caption), averaged over the given n-gram orders (default: unigram+bigram).

    Used both by the offline diagnostic (`self_bleu`/duplicate stats above)
    and, imported directly, by the SCST reward in `modules/modeling.py` to
    penalize a generated caption that is basically a duplicate of another
    video's generated caption in the same training batch.

    Deliberately NOT IDF-weighted: CIDEr already down-weights common
    n-grams, which is exactly why per-sample CIDEr reward fails to notice a
    candidate reusing a generic template (e.g. "a man is talking") that also
    happens to be common in the reference corpus -- this penalty must catch
    that reuse regardless of how common the shared phrase is.

    All inputs are pre-tokenized token lists (not raw strings) so this stays
    usable both on already-PTBTokenizer-tokenized text (training) and on
    tokens produced by `tokenize()` above (offline diagnostics).
    """
    if not other_tokens_list:
        return 0.0
    cand_sets = {n: ngram_set(cand_tokens, n) for n in orders}
    if all(len(s) == 0 for s in cand_sets.values()):
        return 0.0
    best = 0.0
    for other_tokens in other_tokens_list:
        ratios = []
        for n in orders:
            cand_ngrams = cand_sets[n]
            if not cand_ngrams:
                continue
            other_ngrams = ngram_set(other_tokens, n)
            ratios.append(len(cand_ngrams & other_ngrams) / len(cand_ngrams))
        if ratios:
            best = max(best, sum(ratios) / len(ratios))
    return best


def duplicate_stats(captions: Sequence[str], top_k: int = 15) -> Dict:
    counts = Counter(captions)
    total = len(captions)
    unique = len(counts)
    top = counts.most_common(top_k)
    return {
        "total": total,
        "unique": unique,
        "unique_ratio": unique / total if total else 0.0,
        "duplicate_ratio": 1 - (unique / total if total else 0.0),
        "top_templates": [
            {"caption": cap, "count": cnt, "pct_of_total": cnt / total}
            for cap, cnt in top
        ],
    }


def opening_ngram_stats(captions: Sequence[str], prefix_len: int = 4, top_k: int = 15) -> Dict:
    """How often the same OPENING n-gram is reused across different captions.

    Catches the "a man in a black shirt is ..." pattern: the tail of the
    sentence may vary, but the model always reaches for the same fixed
    opening template regardless of what the video actually shows.
    """
    total = len(captions)
    openers = Counter()
    for cap in captions:
        toks = tokenize(cap)
        if len(toks) >= prefix_len:
            openers[" ".join(toks[:prefix_len])] += 1
    top = openers.most_common(top_k)
    return {
        "prefix_len": prefix_len,
        "distinct_openers": len(openers),
        "top_openers": [
            {"prefix": p, "count": cnt, "pct_of_total": cnt / total}
            for p, cnt in top
        ],
    }


def distinct_n(captions: Sequence[str], n: int) -> float:
    """Corpus-level distinct-n: unique n-grams / total n-grams across ALL captions.

    Standard diversity metric for text generation (Li et al., 2016).
    Low value = model reuses the same n-grams over and over (generic/bland).
    """
    all_ngrams = []
    for cap in captions:
        all_ngrams.extend(ngrams(tokenize(cap), n))
    if not all_ngrams:
        return 0.0
    return len(set(all_ngrams)) / len(all_ngrams)


def _bleu_score(candidate_tokens: Sequence[str], reference_token_lists: Sequence[Sequence[str]], max_n: int = 4) -> float:
    """Lightweight smoothed BLEU-n, no external deps. Used only for self-BLEU
    (relative diversity comparison across runs), not as a reported eval metric."""
    if not candidate_tokens:
        return 0.0

    precisions = []
    for n in range(1, max_n + 1):
        cand_ngrams = Counter(ngrams(candidate_tokens, n))
        if not cand_ngrams:
            precisions.append(0.0)
            continue
        max_ref_counts = Counter()
        for ref_tokens in reference_token_lists:
            ref_ngrams = Counter(ngrams(ref_tokens, n))
            for g, c in ref_ngrams.items():
                if c > max_ref_counts[g]:
                    max_ref_counts[g] = c
        clipped = sum(min(c, max_ref_counts[g]) for g, c in cand_ngrams.items())
        total = sum(cand_ngrams.values())
        # +1 smoothing to avoid zero precision killing the geometric mean on short captions
        precisions.append((clipped + 1) / (total + 1))

    log_avg = sum(math.log(p) for p in precisions) / max_n
    cand_len = len(candidate_tokens)
    closest_ref_len = min((len(r) for r in reference_token_lists), key=lambda l: (abs(l - cand_len), l)) if reference_token_lists else cand_len
    bp = 1.0 if cand_len >= closest_ref_len else math.exp(1 - closest_ref_len / max(cand_len, 1))
    return bp * math.exp(log_avg)


def self_bleu(captions: Sequence[str], sample_size: int = 200, seed: int = 42) -> float:
    """Average BLEU-4 of each sampled caption against all OTHER sampled captions
    (used as references). Lower = more diverse corpus; higher = mode collapse
    (captions keep matching each other, i.e. the model is repeating itself).

    This is the standard "Self-BLEU" diversity diagnostic (Zhu et al., 2018),
    reimplemented here without extra dependencies.
    """
    if len(captions) < 2:
        return 0.0
    rng = random.Random(seed)
    sample = captions if len(captions) <= sample_size else rng.sample(list(captions), sample_size)
    tokenized = [tokenize(c) for c in sample]

    scores = []
    for i, cand in enumerate(tokenized):
        refs = tokenized[:i] + tokenized[i + 1:]
        scores.append(_bleu_score(cand, refs))
    return sum(scores) / len(scores)


def basic_stats(captions: Sequence[str]) -> Dict:
    lengths = [len(tokenize(c)) for c in captions]
    vocab = set()
    for c in captions:
        vocab.update(tokenize(c))
    return {
        "vocab_size": len(vocab),
        "avg_len_tokens": sum(lengths) / len(lengths) if lengths else 0.0,
    }


def build_report(captions: Sequence[str], top_k: int = 15, self_bleu_sample: int = 200) -> Dict:
    return {
        "duplicates": duplicate_stats(captions, top_k=top_k),
        "opening_templates": opening_ngram_stats(captions, top_k=top_k),
        "distinct_1": distinct_n(captions, 1),
        "distinct_2": distinct_n(captions, 2),
        "distinct_3": distinct_n(captions, 3),
        "self_bleu_4": self_bleu(captions, sample_size=self_bleu_sample),
        "basic": basic_stats(captions),
    }


def print_report(name: str, report: Dict) -> None:
    dup = report["duplicates"]
    opn = report["opening_templates"]
    basic = report["basic"]
    print(f"\n=== {name} ===")
    print(f"Total captions        : {dup['total']}")
    print(f"Unique captions       : {dup['unique']}  ({dup['unique_ratio']:.1%} unique, "
          f"{dup['duplicate_ratio']:.1%} are exact duplicates of another line)")
    print(f"Vocab size            : {basic['vocab_size']}")
    print(f"Avg length (tokens)   : {basic['avg_len_tokens']:.2f}")
    print(f"Distinct-1 / 2 / 3    : {report['distinct_1']:.3f} / {report['distinct_2']:.3f} / {report['distinct_3']:.3f}"
          f"  (closer to 1.0 = more diverse n-grams, closer to 0 = repeats itself)")
    print(f"Self-BLEU-4 (sampled) : {report['self_bleu_4']:.3f}"
          f"  (higher = captions are more similar to each other = more mode collapse)")

    print(f"\nTop {len(dup['top_templates'])} most-repeated exact captions:")
    for row in dup["top_templates"]:
        print(f"  {row['count']:>4}x ({row['pct_of_total']:>5.1%})  {row['caption']}")

    print(f"\nTop {len(opn['top_openers'])} most-repeated opening {opn['prefix_len']}-grams "
          f"({opn['distinct_openers']} distinct openers total):")
    for row in opn["top_openers"]:
        print(f"  {row['count']:>4}x ({row['pct_of_total']:>5.1%})  \"{row['prefix']} ...\"")


def print_delta(name_a: str, report_a: Dict, name_b: str, report_b: Dict) -> None:
    print(f"\n=== Delta: {name_a}  ->  {name_b} ===")
    a, b = report_a["duplicates"], report_b["duplicates"]
    print(f"Unique ratio     : {a['unique_ratio']:.1%} -> {b['unique_ratio']:.1%}  "
          f"({'better' if b['unique_ratio'] > a['unique_ratio'] else 'worse' if b['unique_ratio'] < a['unique_ratio'] else 'same'})")
    for n in (1, 2, 3):
        ka = report_a[f"distinct_{n}"]
        kb = report_b[f"distinct_{n}"]
        print(f"Distinct-{n}       : {ka:.3f} -> {kb:.3f}  "
              f"({'better' if kb > ka else 'worse' if kb < ka else 'same'})")
    sa, sb = report_a["self_bleu_4"], report_b["self_bleu_4"]
    print(f"Self-BLEU-4      : {sa:.3f} -> {sb:.3f}  "
          f"({'better (more diverse)' if sb < sa else 'worse (more mode collapse)' if sb > sa else 'same'})")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hyp", required=True, help="Path to hyp.txt (one generated caption per line).")
    parser.add_argument("--compare", default=None, help="Optional second hyp.txt to compare against (e.g. an earlier checkpoint).")
    parser.add_argument("--top-k", type=int, default=15, help="How many top repeated templates/openers to print.")
    parser.add_argument("--self-bleu-sample", type=int, default=200, help="Sample size for self-BLEU (O(n^2), keep this modest).")
    parser.add_argument("--json-out", default=None, help="Optional path to dump the full report(s) as JSON.")
    args = parser.parse_args()

    captions = load_lines(args.hyp)
    report = build_report(captions, top_k=args.top_k, self_bleu_sample=args.self_bleu_sample)
    print_report(args.hyp, report)

    all_reports = {args.hyp: report}

    if args.compare:
        other_captions = load_lines(args.compare)
        other_report = build_report(other_captions, top_k=args.top_k, self_bleu_sample=args.self_bleu_sample)
        print_report(args.compare, other_report)
        print_delta(args.compare, other_report, args.hyp, report)
        all_reports[args.compare] = other_report

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(all_reports, f, indent=2, ensure_ascii=False)
        print(f"\nFull report written to {args.json_out}")


if __name__ == "__main__":
    main()
