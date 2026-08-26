import torch


def _calculate_and_get_stats(tensor: torch.Tensor, mask: torch.Tensor, name_prefix: str, is_log: bool = False):
    """
    (内部辅助函数) 计算并统计给定张量在 mask 下的 min, max, mean 以及分布情况。
    """
    stats = {}
    mask = mask.bool()

    # 如果掩码中没有任何 True 值，则返回默认值
    if not torch.any(mask):
        stats[f"{name_prefix}_min"] = torch.tensor(0.0, device=tensor.device)
        stats[f"{name_prefix}_max"] = torch.tensor(0.0, device=tensor.device)
        stats[f"{name_prefix}_mean"] = torch.tensor(0.0, device=tensor.device)
        if is_log:
            stats[f"{name_prefix}_dist_neg_inf_-0.7"] = torch.tensor(0, device=tensor.device)
            stats[f"{name_prefix}_dist_-0.7_-0.2"] = torch.tensor(0, device=tensor.device)
            stats[f"{name_prefix}_dist_-0.2_0.2"] = torch.tensor(0, device=tensor.device)
            stats[f"{name_prefix}_dist_0.2_0.7"] = torch.tensor(0, device=tensor.device)
            stats[f"{name_prefix}_dist_0.7_inf"] = torch.tensor(0, device=tensor.device)
        else:
            stats[f"{name_prefix}_dist_0_0.5"] = torch.tensor(0, device=tensor.device)
            stats[f"{name_prefix}_dist_0.5_1.0"] = torch.tensor(0, device=tensor.device)
            stats[f"{name_prefix}_dist_1.0_1.5"] = torch.tensor(0, device=tensor.device)
            stats[f"{name_prefix}_dist_1.5_2.0"] = torch.tensor(0, device=tensor.device)
            stats[f"{name_prefix}_dist_2.0_inf"] = torch.tensor(0, device=tensor.device)
        return stats

    masked_tensor = torch.masked_select(tensor, mask)
    
    if masked_tensor.numel() == 0:
        # 递归调用以返回零值字典
        return _calculate_and_get_stats(tensor, torch.zeros_like(mask, dtype=torch.bool), name_prefix, is_log)

    stats[f"{name_prefix}_min"] = torch.min(masked_tensor)
    stats[f"{name_prefix}_max"] = torch.max(masked_tensor)
    stats[f"{name_prefix}_mean"] = torch.mean(masked_tensor)

    if is_log:
        # log_ratio 的分布统计 (以0为中心)
        stats[f"{name_prefix}_dist_neg_inf_-0.7"] = torch.sum(masked_tensor < -0.7)
        stats[f"{name_prefix}_dist_-0.7_-0.2"] = torch.sum((masked_tensor >= -0.7) & (masked_tensor < -0.2))
        stats[f"{name_prefix}_dist_-0.2_0.2"] = torch.sum((masked_tensor >= -0.2) & (masked_tensor <= 0.2))
        stats[f"{name_prefix}_dist_0.2_0.7"] = torch.sum((masked_tensor > 0.2) & (masked_tensor <= 0.7))
        stats[f"{name_prefix}_dist_0.7_inf"] = torch.sum(masked_tensor > 0.7)
    else:
        # ratio 的分布统计 (以1为中心)
        stats[f"{name_prefix}_dist_0_0.5"] = torch.sum(masked_tensor < 0.5)
        stats[f"{name_prefix}_dist_0.5_1.0"] = torch.sum((masked_tensor >= 0.5) & (masked_tensor < 1.0))
        stats[f"{name_prefix}_dist_1.0_1.5"] = torch.sum((masked_tensor >= 1.0) & (masked_tensor < 1.5))
        stats[f"{name_prefix}_dist_1.5_2.0"] = torch.sum((masked_tensor >= 1.5) & (masked_tensor < 2.0))
        stats[f"{name_prefix}_dist_2.0_inf"] = torch.sum(masked_tensor >= 2.0)

    return stats


def get_ratio_stats(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    target_probs: torch.Tensor,
    off_policy_strategy: str,
    prefix_mask: torch.Tensor,
    response_mask: torch.Tensor,
    off_policy_reshape: str,
):
    """
    计算 on-policy 和 off-policy 的 ratio 和 log_ratio，并对它们进行统计。

    Args:
        old_log_prob (torch.Tensor): 旧策略的 log 概率。
        log_prob (torch.Tensor): 当前策略的 log 概率。
        target_probs (torch.Tensor): 目标概率（如果可用）。
        off_policy_strategy (str): 离策略损失的类型。
        prefix_mask (torch.Tensor): 前缀部分的掩码。
        response_mask (torch.Tensor): 回复部分的掩码。
        off_policy_reshape (str): 离策略 ratio 的重塑方法。

    Returns:
        dict: 包含 on/off ratio 和 log_ratio 统计信息的字典。
    """
    with torch.no_grad():
        all_stats = {}
        on_mask = (~prefix_mask) * response_mask
        off_mask = prefix_mask * response_mask
        all_stats["on_policy_token_count"] = torch.sum(on_mask.float())
        all_stats["off_policy_token_count"] = torch.sum(off_mask.float())
        # 1. 计算和统计 on_ratio 和 on_log_ratio
        on_log_ratio = log_prob - old_log_prob
        on_ratio = torch.exp(on_log_ratio)
        all_stats.update(_calculate_and_get_stats(on_ratio, on_mask, "on_ratio", is_log=False))
        all_stats.update(_calculate_and_get_stats(on_log_ratio, on_mask, "on_log_ratio", is_log=True))

        # 2. 计算和统计 off_ratio 和 off_log_ratio
        if target_probs is None:
            off_log_ratio = log_prob
            if off_policy_strategy == "rlpluss":
                omiga_prob = 0.5 * (torch.exp(old_log_prob) + 1)
                off_ratio = 2 * torch.exp(log_prob) / (torch.exp(old_log_prob) + omiga_prob)
                off_log_ratio = torch.log(off_ratio)
            elif off_policy_strategy == "luffy":
                off_ratio = torch.exp(off_log_ratio)
                if off_policy_reshape == "p_div_p_0.1":
                    off_ratio = off_ratio / (off_ratio + 0.1)
                off_log_ratio = torch.log(off_ratio)
        else:
            off_log_ratio = log_prob - torch.log(target_probs + 1e-6)
            off_ratio = torch.exp(off_log_ratio)

        all_stats.update(_calculate_and_get_stats(off_ratio, off_mask, "off_ratio", is_log=False))
        all_stats.update(_calculate_and_get_stats(off_log_ratio, off_mask, "off_log_ratio", is_log=True))

        return all_stats


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
    from collections import defaultdict

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
