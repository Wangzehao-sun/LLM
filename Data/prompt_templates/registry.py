"""Registry of prompt templates: name -> file, pinned hash, provenance note.

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

This registry covers template TEXT only. Output-side leakage filters stay with their
consumers, because the RL trainer and the offline post-processor screen different
things and are expected to use different vocabularies.
"""

from __future__ import annotations

from typing import NamedTuple


class TemplateSpec(NamedTuple):
    """One registered template."""

    filename: str
    sha256: str  # first 12 hex chars of sha256(text); text = file minus one trailing \n
    deprecated: bool  # kept only to reproduce old artefacts; hidden from listings
    note: str  # where it came from and what distinguishes it


TEMPLATES: dict[str, TemplateSpec] = {
    "rephrase_shared_gold_v1": TemplateSpec(
        filename="rephrase_shared_gold_v1.txt",
        sha256="5c010c95e990",
        deprecated=False,
        note=(
            """Gold-standard re-authoring. Byte-identical in both forks (was DEFAULT_TEMPLATE1 in
        each), hence lineage "shared"."""
        ),
    ),
    "rephrase_main_v1": TemplateSpec(
        filename="rephrase_main_v1.txt",
        sha256="f8b9892ee505",
        deprecated=False,
        note=(
            """Was Data/prepare_summarize_prompts.py:DEFAULT_TEMPLATE. Says "## Reference
        Reasoning Draft:" and treats the draft as noisy private guidance."""
        ),
    ),
    "rephrase_main_v2": TemplateSpec(
        filename="rephrase_main_v2.txt",
        sha256="443e9504e201",
        deprecated=False,
        note=(
            """Was Data/prepare_summarize_prompts.py:DEFAULT_TEMPLATE2. Near-duplicate of v1;
        kept because prepare_rephraser_sft.py defaulted to it (old key "template2")."""
        ),
    ),
    "rephrase_inferapi_v1": TemplateSpec(
        filename="rephrase_inferapi_v1.txt",
        sha256="943c8a0f88f9",
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
        deprecated=False,
        note=(
            """Repair the target model's incorrect attempt using expert guidance. Was
        Inferapi:TEACHER_TEMPLATE_DEFAULT1; its brace escaping was already correct."""
        ),
    ),
}


def _check_invariants() -> None:
    """Enforce the registry invariants at import time."""
    seen: dict[str, str] = {}
    for name, spec in TEMPLATES.items():
        if spec.sha256 in seen:
            raise ValueError(f"templates {seen[spec.sha256]!r} and {name!r} have identical bytes ({spec.sha256}); use one name per distinct template")
        seen[spec.sha256] = name
    if "default" in TEMPLATES:
        raise ValueError("'default' is banned as a template name; use <role>_<lineage>_<version>")


_check_invariants()
