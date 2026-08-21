"""The per-token density joint decoding reports as the importance ratio's denominator.

``joint_generate`` hands the trainer a ``log_q`` per generated token, and
``_summarize_replace_normal_step`` feeds it straight into ``off_old_log_probs`` /
``target_probs``. Nothing downstream can tell a wrong density from a right one -- the
loss just gets a different gradient -- so the value is pinned here against hand
computation instead.

The claim under test is that ONE expression covers every branch:

    log q(v) = scores[v]/T - logsumexp(scores/T)

because each decode mode differs only in what it puts in ``scores``. The tests take the
branches one at a time and check the expression collapses to the right thing in each:

    constrained step  -> log p_student(v) - log Z_t   (the mu_t of the method)
    fallback step     -> the full fallback model's log-prob
    k = vocab, no floors -> plain student sampling, unchanged

The probability vectors are the same ones the Z_t work was hand-verified against
(Z_t = 1.000 / 0.550 / 0.450 / 0.300 / 0.450 / 0.000), so a regression in either
quantity shows up against numbers that were checked once by hand.
"""

import pytest
import torch
import torch.nn.functional as F

from verl.custom.joint_decode_core import agree_select, fuse_logits, sample_next, sampled_log_prob

# Student and teacher rank the vocabulary differently on purpose: the student's
# favourite (index 0) is the teacher's least-liked of the plausible tokens, so a
# teacher floor bites exactly where it should be visible.
P_STUDENT = torch.tensor([[0.40, 0.30, 0.15, 0.10, 0.05]])
P_TEACHER = torch.tensor([[0.10, 0.35, 0.30, 0.20, 0.05]])
LOGITS_STUDENT = P_STUDENT.log()
LOGITS_TEACHER = P_TEACHER.log()

# (student_top_k, teacher_top_k, student_min_prob, teacher_min_prob, expected Z_t).
# The Z_t column is hand-computed: sum of the student's probabilities over the tokens
# that survive both top-k's and both floors.
#   full        -- everything eligible                         -> 1.00
#   t-floor     -> drops idx 0 (p_b=0.10) and idx 4 (p_b=0.05) -> 0.30+0.15+0.10 = 0.55
#   +s-topk 3   -> also drops idx 3                            -> 0.30+0.15 = 0.45
#   narrow      -> student's top-2 {0,1}, teacher floor kills 0-> 0.30
#   t-topk 2    -> teacher's top-2 {1,2}                       -> 0.30+0.15 = 0.45
#   disjoint    -- top-1 each: student {0} vs teacher {1}      -> empty, fallback
CONFIGS = [
    pytest.param(5, 5, 0.0, 0.00, 1.00, id="full"),
    pytest.param(5, 5, 0.0, 0.15, 0.55, id="teacher-floor"),
    pytest.param(3, 5, 0.0, 0.15, 0.45, id="student-topk-3"),
    pytest.param(2, 3, 0.0, 0.15, 0.30, id="narrow-both"),
    pytest.param(5, 2, 0.0, 0.00, 0.45, id="teacher-topk-2"),
    pytest.param(1, 1, 0.0, 0.00, 0.00, id="disjoint-fallback"),
]


def _select(student_top_k, teacher_top_k, student_min_prob, teacher_min_prob,
            temperature=1.0, top_p=1.0, fallback="teacher"):
    torch.manual_seed(0)
    return agree_select(
        LOGITS_STUDENT, LOGITS_TEACHER,
        student_top_k=student_top_k, teacher_top_k=teacher_top_k,
        student_min_prob=student_min_prob, teacher_min_prob=teacher_min_prob,
        temperature=temperature, top_p=top_p, fallback=fallback,
    )


@pytest.mark.parametrize("s_k,t_k,s_min,t_min,want_z", CONFIGS)
def test_keep_ratio_matches_hand_computation(s_k, t_k, s_min, t_min, want_z):
    """Z_t is the sum of student mass the teacher permitted -- the mu_t denominator."""
    _, _, _, z_t, _ = _select(s_k, t_k, s_min, t_min)
    assert z_t.item() == pytest.approx(want_z, abs=1e-6)


@pytest.mark.parametrize("s_k,t_k,s_min,t_min,want_z", CONFIGS)
def test_density_is_student_logprob_minus_log_z(s_k, t_k, s_min, t_min, want_z):
    """At T=1 the density must be exactly log p_student(v) - log Z_t.

    This is the equation from the method write-up, so it is checked literally rather
    than by re-deriving it the way the implementation does. A fallback step has no Z_t
    (nothing survived), so it is expected to report the teacher's full log-prob
    instead -- covered here as the same assertion with a different target.
    """
    chosen, fell_back, _, z_t, log_q = _select(s_k, t_k, s_min, t_min)
    token = chosen.item()

    if fell_back.item():
        assert want_z == 0.0, "a row with surviving mass should not have fallen back"
        expected = F.log_softmax(LOGITS_TEACHER, dim=-1)[0, token]
    else:
        expected = LOGITS_STUDENT[0, token] - torch.tensor(want_z).log()

    assert log_q.item() == pytest.approx(expected.item(), abs=1e-6)


def test_density_normalises_over_the_eligible_set():
    """exp(log_q) must sum to 1 across the tokens that could have been drawn.

    The per-token test above only pins the ONE token that was sampled. If the
    normaliser were wrong by a constant -- the exact failure mode of forgetting Z_t --
    every token would shift together and a single-token check could still pass under a
    lucky draw. Summing over the whole support catches that.
    """
    for s_k, t_k, s_min, t_min in [(5, 5, 0.0, 0.15), (3, 5, 0.0, 0.15), (2, 3, 0.0, 0.15)]:
        _, fell_back, log_pa, z_t, _ = _select(s_k, t_k, s_min, t_min)
        assert not fell_back.item()

        # Ask sampled_log_prob for EVERY token's density under the same masked scores
        # sampling saw, then sum over the support. Going through the function under test
        # rather than re-deriving p_a/Z_t is the point: it is the normaliser that is
        # being checked, not the algebra.
        masked = log_pa.masked_fill(~_eligible_mask(s_k, t_k, s_min, t_min), float("-inf"))
        vocab = log_pa.size(-1)
        every_token = torch.arange(vocab).view(-1, 1)
        densities = sampled_log_prob(masked.expand(vocab, -1), every_token, 1.0)

        finite = densities[torch.isfinite(densities)]
        assert finite.exp().sum().item() == pytest.approx(1.0, abs=1e-6)
        # And the surviving support is exactly Z_t's worth of student mass.
        assert log_pa.exp()[_eligible_mask(s_k, t_k, s_min, t_min)].sum().item() == pytest.approx(
            z_t.item(), abs=1e-6
        )


def _eligible_mask(student_top_k, teacher_top_k, student_min_prob, teacher_min_prob):
    """Recompute eligibility here, independently of agree_select's internals."""
    log_pa = F.log_softmax(LOGITS_STUDENT, dim=-1)
    log_pb = F.log_softmax(LOGITS_TEACHER, dim=-1)
    in_a = torch.zeros_like(log_pa, dtype=torch.bool)
    in_b = torch.zeros_like(log_pb, dtype=torch.bool)
    in_a.scatter_(-1, log_pa.topk(student_top_k, dim=-1).indices, True)
    in_b.scatter_(-1, log_pb.topk(teacher_top_k, dim=-1).indices, True)
    floor_a = torch.log(torch.tensor(student_min_prob)) if student_min_prob else float("-inf")
    floor_b = torch.log(torch.tensor(teacher_min_prob)) if teacher_min_prob else float("-inf")
    return in_a & in_b & (log_pa >= floor_a) & (log_pb >= floor_b)


@pytest.mark.parametrize("temperature", [0.6, 2.0])
def test_temperature_is_applied(temperature):
    """The density must be temperature-scaled, matching compute_log_prob's numerator.

    ``compute_log_prob`` divides logits by ``rollout.temperature``
    (fsdp_workers_new.py:772 -> new_dp_actor.py:122). A denominator computed at T=1
    against that numerator would leave a systematic, T-dependent bias in off_ratio
    that nothing else in the pipeline would flag.

    Two assertions, because only their conjunction is evidence: the value matches
    log_softmax(masked_scores / T), AND it differs from the T=1 form. Without the
    second, an implementation that silently ignored T could still pass whenever the
    two happened to be close.
    """
    chosen, fell_back, log_pa, z_t, log_q = _select(5, 5, 0.0, 0.15, temperature=temperature)
    assert not fell_back.item()
    token = chosen.item()

    scaled = log_pa.masked_fill(~_eligible_mask(5, 5, 0.0, 0.15), float("-inf")) / temperature
    expected = (scaled - torch.logsumexp(scaled, dim=-1, keepdim=True))[0, token]
    assert log_q.item() == pytest.approx(expected.item(), abs=1e-6)

    unscaled = (log_pa[0, token] - z_t.log()).item()
    assert abs(log_q.item() - unscaled) > 1e-3, "temperature was ignored"


@pytest.mark.parametrize("temperature", [1.0, 0.6])
def test_wide_k_reduces_to_plain_student(temperature):
    """k = vocab with no floors must be indistinguishable from student-only sampling.

    This is the degenerate case that makes the whole path auditable: it is the
    configuration the GPU ``off_ratio ~= 1`` check uses, where the numerator and the
    denominator are the same model under the same prompt, so any drift is a bug rather
    than a property of the method.
    """
    chosen, fell_back, _, z_t, log_q = _select(5, 5, 0.0, 0.0, temperature=temperature)
    assert not fell_back.item()
    assert z_t.item() == pytest.approx(1.0, abs=1e-6)

    # _select reseeds, so this is the density of the token that was actually drawn.
    plain = F.log_softmax(LOGITS_STUDENT / temperature, dim=-1)
    assert log_q.item() == pytest.approx(plain[0, chosen.item()].item(), abs=1e-6)


@pytest.mark.parametrize("fallback,logits", [("teacher", LOGITS_TEACHER), ("student", LOGITS_STUDENT)])
def test_fallback_uses_the_full_fallback_distribution(fallback, logits):
    """With an empty intersection, the density is the fallback model's -- no -inf, no NaN.

    Z_t is 0 here, so an implementation that wrote ``log p_a - log Z_t`` unconditionally
    would emit -inf, and ``exp(-inf) = 0`` would make off_ratio infinite at the loss.
    Reading the density off the same ``scores`` tensor sampling used avoids the special
    case entirely; this test is what proves it.
    """
    chosen, fell_back, _, z_t, log_q = _select(1, 1, 0.0, 0.0, fallback=fallback)
    assert fell_back.item()
    assert z_t.item() == 0.0

    expected = F.log_softmax(logits, dim=-1)[0, chosen.item()]
    assert log_q.item() == pytest.approx(expected.item(), abs=1e-6)
    assert torch.isfinite(log_q).all()


def test_greedy_is_a_point_mass():
    """T=0 samples deterministically, so its log-density is 0, not -inf or NaN.

    Greedy decoding is how the degenerate-case fingerprint checks are run
    (smoke_dual_vllm.py:21-24), so it must not produce a density that poisons the loss
    if someone trains with it.
    """
    _, _, _, _, log_q = _select(5, 5, 0.0, 0.15, temperature=0.0)
    assert log_q.item() == 0.0
    assert torch.isfinite(log_q).all()


@pytest.mark.parametrize("mode", ["linear", "contrastive", "max"])
def test_fuse_modes_also_report_a_density(mode):
    """The blend modes get a density from the same expression, on the fused scores.

    They are not the method's main path, but they share ``joint_generate``, so a
    missing density here would surface as a silently wrong denominator rather than an
    error.
    """
    scores = fuse_logits(LOGITS_STUDENT, LOGITS_TEACHER, mode, 0.5)
    torch.manual_seed(0)
    chosen = sample_next(scores, 0.8, 1.0, -1)
    log_q = sampled_log_prob(scores, chosen, 0.8)

    scaled = scores / 0.8
    expected = (scaled - torch.logsumexp(scaled, dim=-1, keepdim=True))[0, chosen.item()]
    assert log_q.item() == pytest.approx(expected.item(), abs=1e-6)
    assert log_q.item() < 0.0


def test_density_is_batched_per_row():
    """Rows must be independent: joint_generate calls this on a whole batch at once.

    A reduction that accidentally spanned the batch dimension would still return
    plausible negative numbers, so check two rows with different eligible sets get
    their own normalisers.
    """
    logits_a = torch.cat([LOGITS_STUDENT, LOGITS_TEACHER], dim=0)
    logits_b = torch.cat([LOGITS_TEACHER, LOGITS_STUDENT], dim=0)
    torch.manual_seed(0)
    chosen, _, _, z_t, log_q = agree_select(
        logits_a, logits_b,
        student_top_k=5, teacher_top_k=5, student_min_prob=0.0, teacher_min_prob=0.15,
        temperature=1.0, top_p=1.0, fallback="teacher",
    )
    assert log_q.shape == (2,)
    assert z_t.shape == (2,)
    for row in range(2):
        expected = logits_a[row, chosen[row].item()] - z_t[row].log()
        assert log_q[row].item() == pytest.approx(expected.item(), abs=1e-6)
