"""Hostile-input tests: the loader must reject or neutralise, never crash or hang.

Run with ``python3 -m unittest discover -s tests``.
"""

import contextlib
import copy
import io
import json
import os
import tempfile
import time
import unittest

from loopwatch import attribute, score_account
from loopwatch.cli import main
from loopwatch.model import Account, InputError, load_account

_EXAMPLE = os.path.join(os.path.dirname(__file__), "..", "examples", "bot_account.json")
with open(_EXAMPLE, encoding="utf-8") as _fh:
    BASE = json.load(_fh)


def _mutated(fn):
    data = copy.deepcopy(BASE)
    fn(data)
    return data


class RejectsBadFields(unittest.TestCase):
    def assertRejected(self, fn):
        with self.assertRaises(InputError):
            Account.from_dict(_mutated(fn))

    def test_negative_likes(self):
        self.assertRejected(lambda d: d["posts"][0].update(likes=-1))

    def test_string_likes(self):
        self.assertRejected(lambda d: d["posts"][0].update(likes="3"))

    def test_boolean_reposts(self):
        self.assertRejected(lambda d: d["posts"][0].update(reposts=True))

    def test_infinite_likes(self):
        self.assertRejected(lambda d: d["posts"][0].update(likes=float("inf")))

    def test_non_boolean_is_reply(self):
        self.assertRejected(lambda d: d["posts"][0].update(is_reply="false"))

    def test_non_string_text(self):
        self.assertRejected(lambda d: d["posts"][0].update(text={"a": 1}))

    def test_non_object_account(self):
        self.assertRejected(lambda d: d.update(account=[1]))

    def test_non_string_handle(self):
        self.assertRejected(lambda d: d.update(handle=["x"]))

    def test_far_future_timestamp(self):
        self.assertRejected(lambda d: d["posts"][0].update(timestamp="9999-12-31T23:59:59Z"))

    def test_timezone_overflow_timestamp(self):
        self.assertRejected(lambda d: d["posts"][0].update(timestamp="0001-01-01T00:00:00+05:00"))


class HandleIsSafeToPrint(unittest.TestCase):
    def test_control_and_bidi_characters_removed(self):
        acct = Account.from_dict(
            _mutated(lambda d: d.update(handle="x\x1b[2J\x1b]0;t\x07‮evil\nline"))
        )
        self.assertEqual(acct.handle, "x[2J]0;tevilline")

    def test_handle_truncated(self):
        acct = Account.from_dict(_mutated(lambda d: d.update(handle="a" * 10_000)))
        self.assertEqual(len(acct.handle), 64)

    def test_empty_after_sanitising_falls_back(self):
        acct = Account.from_dict(_mutated(lambda d: d.update(handle="\x1b\x07")))
        self.assertEqual(acct.handle, "unknown")


class LoaderRejectsBadFiles(unittest.TestCase):
    def _load_text(self, text):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "in.json")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(text)
            return load_account(path)

    def test_nan_literal(self):
        text = json.dumps(BASE).replace('"likes": 0', '"likes": NaN', 1)
        with self.assertRaises(InputError):
            self._load_text(text)

    def test_deep_nesting(self):
        with self.assertRaises(InputError):
            self._load_text("[" * 100_000)

    def test_huge_integer_literal(self):
        text = json.dumps(BASE).replace('"likes": 0', '"likes": ' + "9" * 10_000, 1)
        with self.assertRaises(InputError):
            self._load_text(text)

    def test_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(InputError):
                load_account(tmp)

    def test_invalid_utf8(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "in.json")
            with open(path, "wb") as fh:
                fh.write(b'{"posts": "\xff"}')
            with self.assertRaises(InputError):
                load_account(path)

    def test_cli_reports_input_errors_without_traceback(self):
        with tempfile.TemporaryDirectory() as tmp:
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                rc = main(["score", tmp])
        self.assertEqual(rc, 2)
        self.assertIn("input error", err.getvalue())


class NoPathologicalRuntime(unittest.TestCase):
    def test_newline_heavy_post_is_linear(self):
        acct = Account.from_dict(
            _mutated(lambda d: d["posts"][0].update(text="a" + "\n" * 200_000 + "a"))
        )
        start = time.monotonic()
        attribute(acct)
        score_account(acct)
        self.assertLess(time.monotonic() - start, 10)

    def test_list_markers_still_detected(self):
        from loopwatch.attribution import _LIST_RE

        for text in ("- a", "intro\n  * b", "x\n1. one", "x\n2) two", "\n\n\t• dot"):
            self.assertTrue(_LIST_RE.search(text), text)
        for text in ("no list here", "a - b", "3.14 is pi"):
            self.assertFalse(_LIST_RE.search(text), text)


if __name__ == "__main__":
    unittest.main()
