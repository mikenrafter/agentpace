"""One model and kind normalizer for learn, calibration and estimate-run.

SYNTHETIC values only. Each test names the bug it guards.
"""

from __future__ import annotations

import unittest

from agentpace import local_runs as lr


class ModelName(unittest.TestCase):
    def test_model_ids_map_to_one_family_name(self):
        # why: learn stored `neon` while estimate-run queried `gemini-4-neon`, so learned history
        #      never reached the estimate; every reader must map an id to the same family.
        # catches: two normalizers, or a family missing from the shared one.
        cases = {
            "claude-opus-5-5": "opus",
            "claude-sonnet-5-5": "sonnet",
            "claude-haiku-5-5": "haiku",
            "claude-fable-5-2": "fable",
            "gemini-4-neon": "neon",
            "gpt-6.1-sol": "sol",
            "gpt-5.3-codex": "codex",
            "codex-auto-review": "codex",
            "gpt-4o": "codex",
            "": "unknown",
            None: "unknown",
        }
        self.assertEqual({raw: lr.model_name(raw) for raw in cases}, cases)

    def test_family_words_match_whole_words_only(self):
        # why: substring matching made `console-1` a Sol run and would do the same to any id
        #      that contains a family's letters.
        # catches: `name in value` instead of a token match.
        self.assertEqual(lr.model_name("console-1"), "console")
        self.assertEqual(lr.model_name("resolver-2"), "resolver")
        self.assertEqual(lr.model_name("opusculum"), "opusculum")

    def test_normalizing_a_normalized_name_changes_nothing(self):
        # why: stored rows already hold a family name; estimate-run normalizes them again on load,
        #      so the mapping must be idempotent or old rows drift away from new queries.
        # catches: a normalizer that maps `neon` or `antigravity-default` somewhere else.
        for raw in ("claude-fable-5-2", "gemini-4-neon", "gpt-5.3-codex", "antigravity-default", "unknown", "o3"):
            once = lr.model_name(raw)
            self.assertEqual(lr.model_name(once), once, raw)

    def test_model_word_in_a_label_only_names_a_known_family(self):
        # why: a vigil label like `sol-parser-run` names its model, `checkin5-widgets` names none;
        #      reading the first word of every label as a model invents models.
        # catches: running the id normalizer over free-form labels.
        self.assertEqual(lr.model_word("sol-parser-run"), "sol")
        self.assertEqual(lr.model_word("fable-review-r1"), "fable")
        self.assertIsNone(lr.model_word("checkin5-widgets"))
        self.assertIsNone(lr.model_word("console-check"))


class KindName(unittest.TestCase):
    def test_abbreviated_kinds_map_to_the_keyword_kind(self):
        # why: vigil meta says `impl`, keywords say `implementation`; one kind split into two
        #      calibration groups with no shared history.
        # catches: storing an explicit kind without normalizing it.
        self.assertEqual(lr.kind_name("impl"), "implementation")
        self.assertEqual(lr.kind_name("Implementation"), "implementation")
        self.assertEqual(lr.kind_name("test"), "tests")
        self.assertEqual(lr.kind_name("review"), "review")

    def test_an_unknown_explicit_kind_is_kept_not_guessed(self):
        # why: an explicit kind the owner wrote (`migration`) is evidence; replacing it with a
        #      keyword guess or `other` loses it.
        # catches: running task_kind over explicit values.
        self.assertEqual(lr.kind_name("migration"), "migration")
        self.assertEqual(lr.kind_name(""), "other")

    def test_make_row_normalizes_explicit_kind_and_model(self):
        # why: every reader builds rows through make_row; normalizing there is what makes learn,
        #      the replay and estimate-run queries agree.
        # catches: normalizing in one reader only.
        row = lr.make_row(start=0, end=10, model="gemini-4-neon", kind="impl")
        self.assertEqual((row["model"], row["kind"]), ("neon", "implementation"))

    def test_stored_rows_are_renormalized_on_load(self):
        # why: rows replayed before this normalizer hold `claude-fable-5-2` or `impl`; a query
        #      for `fable` must still weight them as the same model and kind.
        # catches: estimate-run comparing a normalized query with raw stored labels.
        rows = lr.normalize_rows([{"model": "claude-fable-5-2", "kind": "impl"}, {"model": "opus", "kind": "review"}])
        self.assertEqual([(r["model"], r["kind"]) for r in rows], [("fable", "implementation"), ("opus", "review")])


if __name__ == "__main__":
    unittest.main()
