"""Registry of prompt templates: name -> file, plus a note on where it came from.

Naming is ``<role>_<lineage>_<version>``. The lineage segment is mandatory and the
name ``default`` is banned outright: two different templates both called
``DEFAULT_TEMPLATE`` -- one in this repo, one in the offline API repo, with
completely different text -- is what made this package necessary. An explicit name
per template is the whole mechanism; git carries the history of any edit.

This registry covers template TEXT only. Output-side leakage filters stay with
their consumers, because the RL trainer and the offline post-processor screen
different things and are expected to use different vocabularies.
"""

from __future__ import annotations

from typing import NamedTuple


class TemplateSpec(NamedTuple):
    """One registered template."""

    filename: str
    note: str  # where it came from and what distinguishes it


TEMPLATES: dict[str, TemplateSpec] = {
    "rephrase_shared_gold_v1": TemplateSpec(
        filename="rephrase_shared_gold_v1.txt",
        note=(
            """Gold-standard re-authoring. Byte-identical in both forks (was DEFAULT_TEMPLATE1
        in each), hence lineage "shared"."""
        ),
    ),
    "rephrase_main_v1": TemplateSpec(
        filename="rephrase_main_v1.txt",
        note=(
            """Was Data/prepare_summarize_prompts.py:DEFAULT_TEMPLATE. Says "## Reference
        Reasoning Draft:" and treats the draft as noisy private guidance."""
        ),
    ),
    "rephrase_main_v2": TemplateSpec(
        filename="rephrase_main_v2.txt",
        note=(
            """Was Data/prepare_summarize_prompts.py:DEFAULT_TEMPLATE2. Near-duplicate of v1;
        kept because prepare_rephraser_sft.py defaulted to it (old key "template2")."""
        ),
    ),
    "rephrase_inferapi_v1": TemplateSpec(
        filename="rephrase_inferapi_v1.txt",
        note=(
            """Was Inferapi:DEFAULT_TEMPLATE -- the same NAME as rephrase_main_v1 but different
        text (similarity 0.199), which is the collision this package exists to prevent.
        Says "## Partial Reasoning Draft:". Rendered the existing 512-row SFT parquet."""
        ),
    ),
    "teacher_continue_v1": TemplateSpec(
        filename="teacher_continue_v1.txt",
        note=(
            """Continue a partial reasoning under expert guidance. Was
        Inferapi:TEACHER_TEMPLATE_DEFAULT with the doubled-brace bug FIXED
        (\\boxed{{}} -> \\boxed{}). Use this for all new runs."""
        ),
    ),
    "teacher_repair_v1": TemplateSpec(
        filename="teacher_repair_v1.txt",
        note=(
            """Repair the target model's incorrect attempt using expert guidance. Was
        Inferapi:TEACHER_TEMPLATE_DEFAULT1; its brace escaping was already correct."""
        ),
    ),
}

if "default" in TEMPLATES:
    raise ValueError("'default' is banned as a template name; use <role>_<lineage>_<version>")
