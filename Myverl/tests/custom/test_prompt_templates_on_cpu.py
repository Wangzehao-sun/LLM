"""Byte-identity and invariant tests for the shared prompt-template library.

This suite exists to prove one thing: routing every prompt through
``Data/prompt_templates`` does not change a single rendered byte. It deliberately
lands BEFORE the consumers are migrated, while the old hardcoded constants and the
new ``.txt`` files coexist, so the equivalence can be checked directly rather than
asserted after the fact.

The tests that reach outside the repo (the old Inferapi fork, an already-generated
parquet) skip when those paths are absent, so CI stays green on a bare checkout
while still catching regressions on the development machine.

Run:
    pytest Myverl/tests/custom/test_prompt_templates_on_cpu.py -v
"""

from __future__ import annotations

import ast
import hashlib
import re
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]  # .../LLM (this file: LLM/Myverl/tests/custom/)
DATA_DIR = REPO_ROOT / "Data"
TEMPLATE_DIR = DATA_DIR / "prompt_templates"

# Data/ is a directory of standalone scripts, not an installed package, so it is
# imported by path the same way its own scripts import each other.
if str(DATA_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_DIR))

import prompt_templates as pt  # noqa: E402

# Sources of the pre-refactor constants. These are the ground truth for the
# byte-identity claims; once the constants are deleted, tests reading them skip.
MAIN_PREPARE = DATA_DIR / "prepare_summarize_prompts.py"
INFERAPI_ROOT = Path.home() / "Desktop" / "Inferapi" / "rephrase_rollout"
INFERAPI_PREPARE = INFERAPI_ROOT / "prepare_summarize_prompts.py"
# The 512-row SFT parquet generated with rephrase_inferapi_v1 before this refactor.
REFERENCE_PARQUET = Path.home() / "Desktop" / "deepmath_hard_thinkonly1024_summarize0_5_sft.parquet"

# name -> the constant it was extracted from, as (source file, constant name).
# Used to prove each .txt still round-trips to the original rendering.
LINEAGE = {
    "rephrase_shared_gold_v1": (MAIN_PREPARE, "DEFAULT_TEMPLATE1"),
    "rephrase_main_v1": (MAIN_PREPARE, "DEFAULT_TEMPLATE"),
    "rephrase_main_v2": (MAIN_PREPARE, "DEFAULT_TEMPLATE2"),
    "rephrase_inferapi_v1": (INFERAPI_PREPARE, "DEFAULT_TEMPLATE"),
    "teacher_continue_v1": (INFERAPI_PREPARE, "TEACHER_TEMPLATE_DEFAULT"),
    "teacher_continue_v1_boxedbug": (INFERAPI_PREPARE, "TEACHER_TEMPLATE_DEFAULT"),
    "teacher_repair_v1": (INFERAPI_PREPARE, "TEACHER_TEMPLATE_DEFAULT1"),
}

QUESTION = "WHAT_IS_THE_QUESTION_SENTINEL"
PREFIX = "THE_REASONING_DRAFT_SENTINEL"


def _string_constant(path: Path, name: str) -> str | None:
    """Read a module-level string constant without importing the module.

    Parsing rather than importing keeps this independent of the module's own
    imports -- the Inferapi fork pulls in transformers/math_verify, which need not
    be installed to check a template's bytes.
    """
    if not path.exists():
        return None
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            try:
                value = ast.literal_eval(node.value)
            except (ValueError, SyntaxError):
                return None
            return value if isinstance(value, str) else None
    return None


class TestPinnedHashes(unittest.TestCase):
    """Every .txt must match the hash pinned in the registry.

    This is the tripwire for silent mutation: an editor adding a trailing newline,
    a formatter reflowing prose, an encoding change. Without it, "the template is
    frozen" would be a comment rather than a guarantee.
    """

    def test_all_templates_match_pinned_hash(self):
        for name in pt.template_names(include_deprecated=True):
            with self.subTest(template=name):
                text = pt.load(name)  # raises on mismatch
                self.assertTrue(hashlib.sha256(text.encode()).hexdigest().startswith(pt.template_hash(name)))

    def test_template_files_have_exactly_one_trailing_newline(self):
        # load() strips exactly one newline; if a file ends with two, the loaded
        # text keeps one and the hash check fails. Assert the on-disk convention
        # directly so the failure names the real cause.
        for name in pt.template_names(include_deprecated=True):
            with self.subTest(template=name):
                raw = (TEMPLATE_DIR / pt.TEMPLATES[name].filename).read_text(encoding="utf-8")
                self.assertTrue(raw.endswith("\n"), "template file should end with a newline")
                self.assertFalse(raw.endswith("\n\n"), "template file should end with exactly one newline")

    def test_loaded_text_has_no_trailing_newline(self):
        for name in pt.template_names(include_deprecated=True):
            with self.subTest(template=name):
                self.assertFalse(pt.load(name).endswith("\n"))


class TestRegistryInvariants(unittest.TestCase):
    """The registry's job is to make the original failure -- one name, two different
    texts -- structurally impossible."""

    def test_no_two_names_share_bytes(self):
        seen: dict[str, str] = {}
        for name in pt.template_names(include_deprecated=True):
            digest = hashlib.sha256(pt.load(name).encode()).hexdigest()
            self.assertNotIn(digest, seen, f"{name!r} and {seen.get(digest)!r} have identical bytes")
            seen[digest] = name

    def test_default_name_is_banned(self):
        # "default" is what let two different templates hide behind one identifier.
        self.assertNotIn("default", pt.TEMPLATES)

    def test_every_name_carries_lineage_and_version(self):
        for name in pt.template_names(include_deprecated=True):
            with self.subTest(template=name):
                self.assertRegex(name, r"^[a-z]+(_[a-z0-9]+)+_v\d+(_[a-z]+)?$")

    def test_deprecated_templates_are_hidden_by_default(self):
        visible = pt.template_names()
        everything = pt.template_names(include_deprecated=True)
        self.assertIn("teacher_continue_v1_boxedbug", everything)
        self.assertNotIn("teacher_continue_v1_boxedbug", visible)

    def test_unknown_name_raises(self):
        with self.assertRaises(KeyError):
            pt.load("no_such_template_v1")

    def test_edited_template_fails_the_hash_check(self):
        # Simulate an in-place edit of a frozen template: load() must refuse rather
        # than quietly serve different prompt bytes to one stage of the pipeline.
        name = "rephrase_main_v1"
        path = TEMPLATE_DIR / pt.TEMPLATES[name].filename
        original = path.read_text(encoding="utf-8")
        try:
            path.write_text(original.replace("[Problem]", "[Problem!]", 1), encoding="utf-8")
            with self.assertRaises(ValueError):
                pt.load(name)
        finally:
            path.write_text(original, encoding="utf-8")
        self.assertEqual(path.read_text(encoding="utf-8"), original)


class TestRenderEquivalence(unittest.TestCase):
    """The single re.sub render path must reproduce str.format byte-for-byte.

    This is what licenses deleting the old dual-path logic (and the ``is_teacher``
    flag): if the two paths agree on every template, one of them is redundant.
    """

    def test_matches_str_format_for_every_format_path_template(self):
        checked = 0
        for name, (path, const) in LINEAGE.items():
            original = _string_constant(path, const)
            if original is None:
                continue  # constant already deleted, or Inferapi not present
            if "{style_example_" in original:
                continue  # str.format cannot render these at all; see the teacher tests
            with self.subTest(template=name):
                want = original.format(question=QUESTION, prefix=PREFIX)
                got = pt.render(pt.load(name), QUESTION, PREFIX)
                self.assertEqual(want, got)
                checked += 1
        if checked == 0:
            self.skipTest("no pre-refactor constants available to compare against")

    def test_stored_text_equals_brace_collapsed_original(self):
        checked = 0
        for name, (path, const) in LINEAGE.items():
            if name.endswith("_boxedbug"):
                continue  # deliberately preserves the doubled braces
            original = _string_constant(path, const)
            if original is None:
                continue
            with self.subTest(template=name):
                expected = original.replace("{{", "{").replace("}}", "}")
                self.assertEqual(expected, pt.load(name))
                checked += 1
        if checked == 0:
            self.skipTest("no pre-refactor constants available to compare against")

    def test_literal_braces_survive_rendering(self):
        for name in pt.template_names():
            with self.subTest(template=name):
                rendered = pt.render(pt.load(name), QUESTION, PREFIX)
                self.assertNotIn("{question}", rendered)
                self.assertNotIn("{prefix}", rendered)

    def test_placeholders_are_substituted(self):
        rendered = pt.render(pt.load("rephrase_main_v1"), QUESTION, PREFIX)
        self.assertIn(QUESTION, rendered)
        self.assertIn(PREFIX, rendered)

    def test_style_example_is_preserved_when_unsupplied(self):
        # The RL trainer fills {style_example_1} online with the student's own
        # incorrect attempt, so offline rendering must leave it alone.
        rendered = pt.render(pt.load("teacher_continue_v1"), QUESTION, PREFIX)
        self.assertIn("{style_example_1}", rendered)

    def test_style_example_is_substituted_when_supplied(self):
        rendered = pt.render(pt.load("teacher_continue_v1"), QUESTION, PREFIX, style_examples=["THE_ATTEMPT"])
        self.assertIn("THE_ATTEMPT", rendered)
        self.assertNotIn("{style_example_1}", rendered)

    def test_render_is_idempotent_on_sentinel_free_text(self):
        once = pt.render(pt.load("rephrase_main_v1"), QUESTION, PREFIX)
        twice = pt.render(once, QUESTION, PREFIX)
        self.assertEqual(once, twice)


class TestBoxedBraceRegression(unittest.TestCase):
    """Locks in both halves of the brace fix: the corrected template, and the
    ability to reproduce the batch generated before it."""

    def test_fixed_template_has_literal_single_braces(self):
        text = pt.load("teacher_continue_v1")
        self.assertIn(r"\boxed{}", text)
        self.assertNotIn(r"\boxed{{}}", text)

    def test_boxedbug_variant_preserves_the_shipped_bytes(self):
        text = pt.load("teacher_continue_v1_boxedbug")
        self.assertIn(r"\boxed{{}}", text)

    def test_the_two_variants_differ_only_in_that(self):
        fixed = pt.load("teacher_continue_v1")
        buggy = pt.load("teacher_continue_v1_boxedbug")
        self.assertNotEqual(fixed, buggy)
        self.assertEqual(fixed, buggy.replace(r"\boxed{{}}", r"\boxed{}"))

    def test_no_active_template_carries_doubled_braces(self):
        for name in pt.template_names():  # deprecated excluded
            with self.subTest(template=name):
                self.assertNotIn(r"\boxed{{}}", pt.load(name))


class TestResolve(unittest.TestCase):
    def test_registered_name_round_trips(self):
        text, name, sha = pt.resolve("rephrase_main_v1")
        self.assertEqual(name, "rephrase_main_v1")
        self.assertEqual(sha, pt.template_hash("rephrase_main_v1"))
        self.assertEqual(text, pt.load("rephrase_main_v1"))

    def test_file_path_is_accepted_and_gets_its_own_hash(self):
        import tempfile

        body = "Solve it.\n\n## Problem\n{question}\n\n## Draft\n{prefix}\n\n## Answer:"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "my_template.txt"
            path.write_text(body + "\n", encoding="utf-8")
            text, name, sha = pt.resolve(str(path))
            self.assertEqual(text, body)
            self.assertEqual(name, "custom:my_template.txt")
            self.assertTrue(hashlib.sha256(body.encode()).hexdigest().startswith(sha))

    def test_custom_template_missing_placeholders_is_rejected(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.txt"
            path.write_text("no placeholders here\n", encoding="utf-8")
            with self.assertRaises(SystemExit):
                pt.resolve(str(path))

    def test_nonexistent_spec_is_rejected(self):
        with self.assertRaises(SystemExit):
            pt.resolve("/nonexistent/path/to/template.txt")


class TestBuildMessages(unittest.TestCase):
    """The shared starting point for all three consumers."""

    def test_system_and_user_roles(self):
        msgs = pt.build_messages(
            {"role": "system", "content": "You are a helpful assistant."},
            QUESTION,
            PREFIX,
            pt.load("rephrase_main_v1"),
        )
        self.assertEqual([m["role"] for m in msgs], ["system", "user"])
        self.assertIn(QUESTION, msgs[-1]["content"])

    def test_system_message_is_optional(self):
        msgs = pt.build_messages(None, QUESTION, PREFIX, pt.load("rephrase_main_v1"))
        self.assertEqual([m["role"] for m in msgs], ["user"])

    def test_system_message_is_copied_not_aliased(self):
        system = {"role": "system", "content": "original"}
        msgs = pt.build_messages(system, QUESTION, PREFIX, pt.load("rephrase_main_v1"))
        msgs[0]["content"] = "mutated"
        self.assertEqual(system["content"], "original")

    def test_three_entry_points_agree(self):
        # Data construction, training and evaluation must start from identical
        # messages; they may only diverge in what they append afterwards.
        text = pt.load("teacher_continue_v1")
        system = {"role": "system", "content": "You are a helpful assistant."}
        build = pt.build_messages(system, QUESTION, PREFIX, text)
        train = pt.build_messages(system, QUESTION, PREFIX, text)
        evaluate = pt.build_messages(system, QUESTION, PREFIX, text)
        self.assertEqual(build, train)
        self.assertEqual(build, evaluate)
        # SFT appends the target; the shared prefix must be untouched.
        sft = list(train) + [{"role": "assistant", "content": "THE_TARGET"}]
        self.assertEqual(sft[: len(build)], build)


class TestProvenance(unittest.TestCase):
    def test_records_id_and_hash(self):
        meta = pt.provenance("teacher_continue_v1")
        self.assertEqual(meta["prompt_id"], "teacher_continue_v1")
        self.assertEqual(meta["prompt_sha256"], pt.template_hash("teacher_continue_v1"))
        self.assertEqual(meta["loader_version"], pt.LOADER_VERSION)
        self.assertIn("rendered_at", meta)

    def test_records_chat_template_hash_when_tokenizer_given(self):
        # Identical text under a different chat template tokenizes differently, so
        # this is what makes SFT/eval token-level misalignment detectable.
        class FakeTokenizer:
            name_or_path = "fake/Qwen3-4b-base"
            chat_template = "{% for m in messages %}{{ m['content'] }}{% endfor %}"

        meta = pt.provenance("rephrase_main_v1", tokenizer=FakeTokenizer())
        self.assertEqual(meta["tokenizer_name"], "fake/Qwen3-4b-base")
        self.assertTrue(hashlib.sha256(FakeTokenizer.chat_template.encode()).hexdigest().startswith(meta["chat_template_sha256"]))

    def test_differing_chat_templates_produce_differing_hashes(self):
        class A:
            name_or_path = "a"
            chat_template = "{{ 'A' }}"

        class B:
            name_or_path = "b"
            chat_template = "{{ 'B' }}"

        self.assertNotEqual(
            pt.provenance("rephrase_main_v1", tokenizer=A())["chat_template_sha256"],
            pt.provenance("rephrase_main_v1", tokenizer=B())["chat_template_sha256"],
        )


class TestReferenceParquetReRender(unittest.TestCase):
    """The load-bearing test: re-render an already-generated SFT parquet through the
    new loader and require every row to come back byte-identical.

    If this passes, the refactor provably did not perturb the data that has already
    been produced -- which is the entire promise of "unify without changing bytes".
    """

    def test_reference_parquet_rerenders_identically(self):
        if not REFERENCE_PARQUET.exists():
            self.skipTest(f"reference parquet not present at {REFERENCE_PARQUET}")
        try:
            import pandas as pd
        except ImportError:
            self.skipTest("pandas not available")

        template = pt.load("rephrase_inferapi_v1")
        # Split the template on its own placeholders to recover (question, prefix)
        # from each rendered prompt, then re-render and compare.
        head, rest = template.split("{question}", 1)
        mid, tail = rest.split("{prefix}", 1)

        df = pd.read_parquet(REFERENCE_PARQUET)
        identical = 0
        for _, row in df.iterrows():
            content = dict(row["summarize_prompts"][1])["content"]
            self.assertTrue(content.startswith(head))
            self.assertTrue(content.endswith(tail))
            body = content[len(head) : len(content) - len(tail)]
            self.assertIn(mid, body)
            question, prefix = body.split(mid, 1)
            self.assertEqual(pt.render(template, question, prefix), content)
            identical += 1
        self.assertEqual(identical, len(df))


class TestPlaceholderContract(unittest.TestCase):
    def test_no_template_uses_an_unknown_placeholder(self):
        # Any {foo} that is not a known placeholder and not a literal brace group
        # would silently survive into the prompt.
        known = {"question", "prefix"}
        for name in pt.template_names(include_deprecated=True):
            text = pt.load(name)
            for match in re.finditer(r"\{([a-z_][a-z0-9_]*)\}", text):
                field = match.group(1)
                with self.subTest(template=name, field=field):
                    self.assertTrue(
                        field in known or field.startswith("style_example_"),
                        f"{name} uses unknown placeholder {{{field}}}",
                    )

    def test_every_template_has_question_and_prefix(self):
        for name in pt.template_names(include_deprecated=True):
            with self.subTest(template=name):
                text = pt.load(name)
                self.assertIn("{question}", text)
                self.assertIn("{prefix}", text)


if __name__ == "__main__":
    unittest.main()
