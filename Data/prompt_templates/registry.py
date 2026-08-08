"""Registry of prompt templates: name -> file, pinned hash, family, provenance note.

Two invariants are enforced at import time, both aimed at the failure that made this
package necessary -- two templates sharing a name while differing in text:

1. **No name may map to the same bytes as another.** Duplicate templates under two
   names would make provenance ambiguous.
2. **Every name pins a sha256 prefix**, checked against the file on every load (see
   ``load`` in ``__init__.py``). A template that has been used to generate data is
   frozen; changing it means adding a new version, not editing the file.

Naming is ``<role>_<lineage>_<version>``. The lineage segment is mandatory and the
name ``default`` is banned outright: "default" is precisely what let two different
templates hide behind one identifier.

FILTER_PHRASES holds the leakage-filter vocabulary per family, and the families are
split so that each consumer's current list is reproduced EXACTLY rather than as a
superset:

* the RL trainer used ``rephrase_main + experience``  (19+4 = 23 keywords, 19 phrases)
* the offline filter used ``rephrase_main + teacher`` (19+21 = 40 keywords, 43 phrases)

so neither behaviour changes on migration. The lists live here rather than being
derived from template text because many phrases match no template literally -- they
are paraphrase-robust guesses about what a model might echo back, plus residue from
template revisions. A test asserts the literally-present subset.
"""

from __future__ import annotations

from typing import NamedTuple


class TemplateSpec(NamedTuple):
    """One registered template."""

    filename: str
    sha256: str  # first 12 hex chars of sha256(text); text = file minus one trailing \n
    family: str  # groups templates sharing leakage-filter vocabulary
    deprecated: bool  # kept only to reproduce old artefacts; hidden from listings
    note: str  # where it came from and what distinguishes it


TEMPLATES: dict[str, TemplateSpec] = {
    "rephrase_shared_gold_v1": TemplateSpec(
        filename="rephrase_shared_gold_v1.txt",
        sha256="5c010c95e990",
        family="rephrase_main",
        deprecated=False,
        note=(
            """Gold-standard re-authoring. Byte-identical in both forks (was DEFAULT_TEMPLATE1 in
        each), hence lineage "shared"."""
        ),
    ),
    "rephrase_main_v1": TemplateSpec(
        filename="rephrase_main_v1.txt",
        sha256="f8b9892ee505",
        family="rephrase_main",
        deprecated=False,
        note=(
            """Was Data/prepare_summarize_prompts.py:DEFAULT_TEMPLATE. Says "## Reference
        Reasoning Draft:" and treats the draft as noisy private guidance."""
        ),
    ),
    "rephrase_main_v2": TemplateSpec(
        filename="rephrase_main_v2.txt",
        sha256="443e9504e201",
        family="rephrase_main",
        deprecated=False,
        note=(
            """Was Data/prepare_summarize_prompts.py:DEFAULT_TEMPLATE2. Near-duplicate of v1;
        kept because prepare_rephraser_sft.py defaulted to it (old key "template2")."""
        ),
    ),
    "rephrase_inferapi_v1": TemplateSpec(
        filename="rephrase_inferapi_v1.txt",
        sha256="943c8a0f88f9",
        family="rephrase_main",
        deprecated=False,
        note=(
            """Was Inferapi:DEFAULT_TEMPLATE -- the same NAME as rephrase_main_v1 but different
        text (similarity 0.199), which is the collision this registry exists to prevent.
        Says "## Partial Reasoning Draft:". Rendered the existing 512-row SFT parquet."""
        ),
    ),
    "teacher_continue_v1": TemplateSpec(
        filename="teacher_continue_v1.txt",
        sha256="e5292936fbf2",
        family="teacher",
        deprecated=False,
        note=(
            """Continue a partial reasoning under expert guidance. Was
        Inferapi:TEACHER_TEMPLATE_DEFAULT with the doubled-brace bug FIXED
        (\\boxed{{}} -> \\boxed{}). Use this for all new runs."""
        ),
    ),
    "teacher_continue_v1_boxedbug": TemplateSpec(
        filename="teacher_continue_v1_boxedbug.txt",
        sha256="fb4c9efec18b",
        family="teacher",
        deprecated=True,
        note=(
            """teacher_continue_v1 exactly as shipped, preserving the literal \\boxed{{}}. Kept
        ONLY to reproduce the 16-row batch generated before the fix -- the doubled braces
        really did reach the API. Do not use for new data."""
        ),
    ),
    "teacher_repair_v1": TemplateSpec(
        filename="teacher_repair_v1.txt",
        sha256="692dcdd25166",
        family="teacher",
        deprecated=False,
        note=(
            """Repair the target model's incorrect attempt using expert guidance. Was
        Inferapi:TEACHER_TEMPLATE_DEFAULT1; its brace escaping was already correct."""
        ),
    ),
}


FILTER_PHRASES: dict[str, dict[str, list[str]]] = {
    # Shared rephrase vocabulary: meta-narration about the draft, and the template's
    # own instruction wording echoed back.
    "rephrase_main": {
        "keywords": [
            "the draft",
            "this draft",
            "reasoning draft",
            "let's summarize the reasoning",
            "partial reasoning draft",
            "[reasoning draft]",
            "[partial reasoning draft]",
            "the provided draft",
            "the given draft",
            "the reference reasoning",
            "reference reasoning",
            "provided reasoning",
            "based on the reasoning above",
            "given reasoning",
            "based on the draft",
            "according to the draft",
            "as stated in the draft",
            "as shown in the draft",
            "from the draft",
        ],
        "instruction_phrases": [
            "gold standard",
            "strict requirements",
            "total de-reference",
            "invisible integration",
            "re-author",
            "your task is",
            "do not mention",
            "do not quote",
            "do not summarize",
            "do not copy",
            "output only",
            "the reader must",
            "the reader should",
            "commentary on the draft",
            "single coherent solution",
            "self-contained solution",
            "no meta-talk",
            "conversational fillers",
            "final output should read",
        ],
    },
    # Separate lineage (SE / "experience"-style prompts). Present in the RL trainer's
    # list, absent from the offline filter -- kept apart so unions stay exact.
    "experience": {
        "keywords": [
            "the experience",
            "refer to experience",
            "based on the experience",
            "according to the experience",
        ],
        "instruction_phrases": [
            # (none)
        ],
    },
    # Teacher templates add XML scaffolding tags and correction-process wording.
    "teacher": {
        "keywords": [
            "<target_model_attempt>",
            "</target_model_attempt>",
            "<expert_reasoning_guidance>",
            "</expert_reasoning_guidance>",
            "<problem>",
            "</problem>",
            "target model's attempt",
            "target model attempt",
            "the target attempt",
            "provided target attempt",
            "given target attempt",
            "the original attempt",
            "expert reasoning guidance",
            "the expert guidance",
            "provided expert guidance",
            "given expert guidance",
            "according to the expert guidance",
            "based on the expert guidance",
            "use the expert guidance",
            "the correction process",
            "mention the correction process",
        ],
        "instruction_phrases": [
            "natural completion of the target model's own attempt",
            "longest usable initial prefix",
            "copied verbatim",
            "do not paraphrase",
            "do not shorten",
            "do not reorder",
            "do not polish",
            "do not reformat",
            "small harmless mistakes",
            "false starts, and redundancy",
            "first substantive error",
            "minimum necessary repair",
            "reconstruct only the remaining part",
            "same narrative person",
            "markdown structure",
            "do not switch from 'i' to 'we'",
            "textbook voice",
            "do not compress valid",
            "for mathematical content, not for wording",
            "do not merely replace the answer",
            "output only the completed solution",
            "mention the attempt, expert guidance",
            "exactly one \\boxed",
            "education-level solution",
        ],
    },
}


def _check_invariants() -> None:
    """Enforce the registry invariants at import time."""
    seen: dict[str, str] = {}
    for name, spec in TEMPLATES.items():
        if spec.sha256 in seen:
            raise ValueError(f"templates {seen[spec.sha256]!r} and {name!r} have identical bytes ({spec.sha256}); use one name per distinct template")
        seen[spec.sha256] = name
        if spec.family not in FILTER_PHRASES:
            raise ValueError(f"template {name!r} declares family {spec.family!r} with no FILTER_PHRASES entry")
    if "default" in TEMPLATES:
        raise ValueError("'default' is banned as a template name; use <role>_<lineage>_<version>")


_check_invariants()
