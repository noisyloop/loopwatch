"""Command-line interface for loopwatch.

    python3 -m loopwatch score account.json [--json] [--window N]
    python3 -m loopwatch attribute account.json [--json]
"""

from __future__ import annotations

import argparse
import json
import sys

from .model import load_account, InputError
from .scoring import score_account, DEFAULT_WINDOW_DAYS
from .attribution import attribute
from . import report
from . import __version__


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="loopwatch",
        description=(
            "Detect LLM-assisted engagement bots from a JSON dump of one "
            "account's posts. Outputs are leads for a human analyst, not "
            "verdicts — see the README for limitations."
        ),
    )
    parser.add_argument("--version", action="version", version=f"loopwatch {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_score = sub.add_parser("score", help="run all signals and the combined score")
    p_score.add_argument("path", help="path to an account JSON file")
    p_score.add_argument("--json", action="store_true", help="emit full results as JSON")
    p_score.add_argument(
        "--window",
        type=int,
        default=DEFAULT_WINDOW_DAYS,
        metavar="DAYS",
        help=f"rolling window in days for drift/growth signals (default {DEFAULT_WINDOW_DAYS})",
    )

    p_attr = sub.add_parser("attribute", help="run model attribution only")
    p_attr.add_argument("path", help="path to an account JSON file")
    p_attr.add_argument("--json", action="store_true", help="emit attribution as JSON")

    return parser


def _cmd_score(args) -> int:
    acct = load_account(args.path)
    if args.window < 1:
        print("error: --window must be >= 1", file=sys.stderr)
        return 2
    result = score_account(acct, window_days=args.window)
    if args.json:
        out = result.as_dict()
        out["attribution"] = attribute(acct)["distribution"]
        print(json.dumps(out, indent=2))
    else:
        print(report.render_score(result, acct))
    return 0


def _cmd_attribute(args) -> int:
    acct = load_account(args.path)
    if args.json:
        print(json.dumps(attribute(acct), indent=2))
    else:
        print(report.render_attribution(acct))
    return 0


def main(argv=None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "score":
            return _cmd_score(args)
        if args.command == "attribute":
            return _cmd_attribute(args)
    except InputError as exc:
        print(f"input error: {exc}", file=sys.stderr)
        return 2
    except FileNotFoundError:
        print(f"error: no such file: {args.path}", file=sys.stderr)
        return 2
    parser.error("unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
