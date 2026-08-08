"""Single source of truth for rephrase / teacher prompt templates.

Why this exists
---------------
The same prompt has to be rendered at three different points in the pipeline --
offline SFT-data construction (strong-model API), SFT training, and evaluation.
When each of those carried its own copy of the template text, they drifted:

* ``DEFAULT_TEMPLATE`` existed in BOTH ``Data/prepare_summarize_prompts.py`` and
  ``Inferapi/rephrase_rollout/prepare_summarize_prompts.py`` under the same name
  with COMPLETELY different text (similarity 0.199; one says
  ``## Reference Reasoning Draft:``, the other ``## Partial Reasoning Draft:``).
  The already-generated 512-row SFT parquet was rendered with the Inferapi
  variant, which does not exist in this repo at all.
* A literal ``\\boxed{{}}`` slipped into a template that is rendered by
  replacement rather than ``str.format``, so the doubled braces reached the API
  verbatim in a shipped batch.

Scope: this package owns TEMPLATE TEXT only. The leakage-filter phrase lists in
the RL trainer and in the offline post-processor are deliberately NOT here --
they screen different things (on-policy rollouts under a rephrase prompt vs a
strong model's output under a teacher prompt), so they are expected to differ and
each stays with its consumer.

Every template now lives in exactly one ``.txt`` file next to this module, is
addressed by an explicit name, and is verified against a pinned sha256 on load.
An accuracy difference measured across the pipeline can therefore be attributed
to the model rather than to prompt drift.

Design notes
------------
**Why .txt and not YAML/Python.** The doubled-brace bug was purely an artifact of
Python string escaping, and YAML would reintroduce an escaping layer of its own
(multi-line prose needs block scalars, where one stray indent silently changes
the bytes). A ``.txt`` file has no escaping layer at all -- ``\\boxed{}`` is just
those characters -- diffs line-by-line in git, and loads with
``Path.read_text()`` and no third-party dependency.

**Why one render path.** Templates are stored in LITERAL form (``\\boxed{}``,
never ``\\boxed{{}}``) and rendered by a single regex substitution over
``{question}`` / ``{prefix}`` / ``{style_example_N}``. This was verified to be
byte-identical to the ``str.format`` path it replaces, for every template and for
all 512 rows of an already-generated SFT parquet. There is deliberately no
``is_teacher`` switch: one path cannot disagree with itself.

**Pure stdlib.** No pandas / numpy / yaml, so the offline API repo can vendor
this module and use it standalone.

Usage::

    import prompt_templates as pt

    text, name, sha = pt.resolve("teacher_continue_v1")   # name or file path
    messages = pt.build_messages(system_msg, question, draft, text)
    meta = pt.provenance(name, sha)                       # record with the output
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from .registry import TEMPLATES, TemplateSpec  # noqa: F401

__all__ = [
    "LOADER_VERSION",
    "PLACEHOLDER_RE",
    "TEMPLATES",
    "build_messages",
    "load",
    "provenance",
    "render",
    "resolve",
    "sha256_of",
    "template_hash",
    "template_names",
]

# Bump when the rendering contract changes in a way that alters output bytes.
# Recorded into provenance so an artefact can be traced to how it was rendered.
LOADER_VERSION = 1

TEMPLATE_DIR = Path(__file__).resolve().parent

# The only placeholders any template may contain. Everything else -- including
# literal ``\boxed{}`` -- passes through untouched, which is the whole reason this
# is a regex substitution rather than str.format.
PLACEHOLDER_RE = re.compile(r"\{(question|prefix|style_example_(\d+))\}")


def sha256_of(text: str) -> str:
    """Full sha256 hex digest of ``text`` (utf-8). The registry pins prefixes of
    these; provenance records the prefix too, so both are derived from one place."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def template_names(include_deprecated: bool = False) -> list[str]:
    """Registered template names, sorted. Deprecated ones are hidden by default so
    they never get picked up by accident -- they exist to reproduce old batches."""
    return sorted(name for name, spec in TEMPLATES.items() if include_deprecated or not spec.deprecated)


def load(name: str) -> str:
    """Return one template's text, verified against its pinned hash.

    Exactly one trailing newline is stripped: the files are stored as POSIX text
    (so editors and git hooks leave them alone), but no template is meant to end
    with a newline. ``.strip()`` is deliberately NOT used -- it would also eat
    leading whitespace, silently changing the prompt.

    The hash check is what makes "same name, different text" impossible: an edited
    template fails loudly here instead of quietly producing different prompts in
    one stage of the pipeline.
    """
    spec = TEMPLATES.get(name)
    if spec is None:
        raise KeyError(f"unknown template {name!r}; registered: {', '.join(template_names(True))}")
    path = TEMPLATE_DIR / spec.filename
    if not path.exists():
        raise FileNotFoundError(f"template file missing for {name!r}: {path}")
    text = path.read_text(encoding="utf-8").removesuffix("\n")
    actual = sha256_of(text)
    if not actual.startswith(spec.sha256):
        raise ValueError(
            f"template {name!r} does not match its pinned hash.\n"
            f"  file:     {path}\n"
            f"  expected: {spec.sha256}...\n"
            f"  actual:   {actual[: len(spec.sha256)]}...\n"
            f"Templates are frozen once used to generate data. To change one, add a "
            f"new version (e.g. {name.rsplit('_v', 1)[0]}_v2) instead of editing this file."
        )
    return text


def template_hash(name: str) -> str:
    """Pinned short hash for ``name`` (no file read)."""
    return TEMPLATES[name].sha256


def resolve(spec: str) -> tuple[str, str, str]:
    """Resolve a template spec to ``(text, name, sha256_short)``.

    ``spec`` is either a registered name or a path to a ``.txt`` file. The path
    form keeps ad-hoc template experiments possible; such a template still gets a
    name (``custom:<filename>``) and its own hash, so an artefact rendered with it
    is still traceable -- it is just not frozen.
    """
    if spec in TEMPLATES:
        return load(spec), spec, TEMPLATES[spec].sha256

    path = Path(spec)
    if not path.exists():
        raise SystemExit(f"--template {spec!r} is neither a registered template ({', '.join(template_names())}) nor an existing file")
    text = path.read_text(encoding="utf-8").removesuffix("\n")
    missing = [p for p in ("{question}", "{prefix}") if p not in text]
    if missing:
        raise SystemExit(f"custom template {path} is missing required placeholder(s): {', '.join(missing)}")
    return text, f"custom:{path.name}", sha256_of(text)[:12]


def render(
    text: str,
    question: str,
    prefix: str,
    style_examples: Sequence[str] | None = None,
) -> str:
    """Fill a template's placeholders. The ONLY render path in the pipeline.

    ``{question}`` and ``{prefix}`` are always substituted. ``{style_example_N}``
    is substituted from ``style_examples`` (1-indexed) when supplied, and left
    UNTOUCHED otherwise -- the RL trainer fills it online with the student's own
    incorrect attempt for the same question, so offline rendering must preserve it.

    Literal braces (``\\boxed{}``) pass through, which is why this is a regex
    substitution: ``str.format`` would raise on them.
    """

    def _sub(m: re.Match) -> str:
        kind = m.group(1)
        if kind == "question":
            return question
        if kind == "prefix":
            return prefix
        idx = int(m.group(2))
        if style_examples is not None and 1 <= idx <= len(style_examples):
            return style_examples[idx - 1]
        return m.group(0)  # leave for the trainer to fill online

    return PLACEHOLDER_RE.sub(_sub, text)


def build_messages(
    system_msg: dict[str, str] | None,
    question: str,
    prefix: str,
    text: str,
    style_examples: Sequence[str] | None = None,
) -> list[dict[str, str]]:
    """Build the ``[system?, user]`` chat messages every consumer starts from.

    The three consumers diverge only AFTER this call: the API sends these messages
    as-is, SFT appends an assistant target, evaluation appends a generation prompt.
    Keeping the shared part in one function is what makes those three paths
    comparable.
    """
    messages: list[dict[str, str]] = []
    if system_msg is not None:
        messages.append(dict(system_msg))
    messages.append({"role": "user", "content": render(text, question, prefix, style_examples)})
    return messages


def provenance(
    name: str,
    sha: str | None = None,
    tokenizer=None,
) -> dict[str, object]:
    """Metadata to record alongside anything rendered from a template.

    Prompt alignment is not just identical visible text -- the same messages under
    a different chat template tokenize differently, so SFT and evaluation can agree
    on text and still disagree on tokens. When a tokenizer is passed, its name and
    the hash of its ``chat_template`` are recorded so that mismatch is detectable.

    Downstream steps compare ``prompt_id`` / ``prompt_sha256`` and refuse to run on
    a mismatch, which is where this stops being decoration.
    """
    meta: dict[str, object] = {
        "prompt_id": name,
        "prompt_sha256": sha if sha is not None else TEMPLATES[name].sha256,
        "loader_version": LOADER_VERSION,
        "rendered_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if tokenizer is not None:
        chat_template = getattr(tokenizer, "chat_template", None)
        meta["tokenizer_name"] = getattr(tokenizer, "name_or_path", None) or str(tokenizer)
        meta["chat_template_sha256"] = sha256_of(chat_template)[:12] if chat_template else None
    return meta
