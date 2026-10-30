"""Validation metrics for PICO. Copied from verl/custom/new_metrics.py (compute_exact_pass_at_k only)."""

from collections import defaultdict


def compute_exact_pass_at_k(data_sources, sample_inputs, scores, success_value: float = 1.0):
    """Exact pass@k per data source: the share of QUESTIONS with at least one hit.

    verl's own metric is ``best@k/mean`` (metric_utils.bootstrap_metric), which draws k
    responses WITH REPLACEMENT 1000 times and averages the per-draw max. That answers
    "if I sampled k, how likely is a hit" and is systematically BELOW this: a question
    solved 2 of 8 times scores 1.0 here but ~0.90 there, since a resample of 8 misses
    both hits about 10% of the time. Both are useful -- this one is the number usually
    reported as pass@k in papers, so it is worth having next to the estimator rather
    than instead of it.

    Grouping is by (data_source, prompt text), matching process_validation_metrics, so a
    val set that repeats each question k times groups exactly like rollout.val_kwargs.n=k.

    k is not an argument: it is whatever each question actually has, so a val set with an
    uneven number of samples per question is reported honestly rather than silently
    truncated. Questions are bucketed by their own count, and the metric name carries it.

    Args:
        data_sources: per-response data source, length == len(scores).
        sample_inputs: per-response decoded prompt, length == len(scores).
        scores: per-response scalar score.
        success_value: a response counts as a hit when its score reaches this.

    Returns:
        ``{data_source: {"pass@k": value, "n_questions@k": count}}``, one entry per
        distinct k found within that data source.
    """
    src2prompt2hits = defaultdict(lambda: defaultdict(list))
    for src, prompt, score in zip(data_sources, sample_inputs, scores):
        try:
            hit = float(score) >= success_value
        except (TypeError, ValueError):
            hit = False
        src2prompt2hits[src][prompt].append(hit)

    out = {}
    for src, prompt2hits in src2prompt2hits.items():
        # Bucket by sample count so questions with different k are never averaged
        # together -- mixing them would produce a pass@k that belongs to no k at all.
        k2solved = defaultdict(list)
        for hits in prompt2hits.values():
            k2solved[len(hits)].append(any(hits))
        metrics = {}
        for k, solved in sorted(k2solved.items()):
            metrics[f"pass@{k}"] = sum(solved) / len(solved)
            metrics[f"n_questions@{k}"] = len(solved)
        out[src] = metrics
    return out
