#!/usr/bin/env python3
"""Regression tests for the stat number parser (assemble._best_stat_figure /
_money_figure / _spelled_to_number).

Guards the documentary-credibility invariants:
  • magnitude words parse to the CORRECT full value (50 million = 50,000,000)
  • money figures format as money ($-prefixed compact); non-money stay bare
  • dates / ordinals / small counts / percents never become magnitude callouts
  • a BARE magnitude word ("the million-dollar question") never conjures a
    phantom 1,000,000 figure

Run:  python tools/test_stat_parser.py     (exit 0 = all pass)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vidlore import assemble as A          # noqa: E402

# (text, expected_best_stat_figure, expected_money_figure)
CASES = [
    # ---- MONEY: correct value + money formatting ----
    ("thirty million dollars a day",            "30,000,000",        "$30M"),
    ("$50 million",                             "50,000,000",        "$50M"),
    ("four hundred and twenty million dollars", "420,000,000",       "$420M"),
    ("$1.2 billion",                            "1,200,000,000",     "$1,200M"),
    ("two trillion dollars",                    "2,000,000,000,000", "$2T"),
    # ---- NON-MONEY MAGNITUDE: correct bare value, NO money ----
    ("50 million views",                        "50,000,000",        ""),
    ("2.5 billion people",                      "2,500,000,000",     ""),
    ("one hundred thousand soldiers",           "100,000",           ""),
    ("3 million users",                         "3,000,000",         ""),
    ("twenty thousand documents",               "20,000",            ""),
    # ---- DATES / QUANTITIES: must never trigger ----
    ("in 1982",                                 "", ""),
    ("the year 2024",                           "", ""),
    ("for 100 years",                           "", ""),
    ("scene 12",                                "", ""),
    ("chapter three",                           "", ""),
    ("three days later",                        "", ""),
    ("47 percent",                              "", ""),
    ("the first time",                          "", ""),
    # ---- PHANTOM PROBES: a bare magnitude word is NOT a stat ----
    ("the million-dollar question",             "", ""),
    ("a billion reasons to worry",              "", ""),
    ("worth millions",                          "", ""),
    ("a million reasons",                       "", ""),
    ("millions of people fled",                 "", ""),
]


def main() -> int:
    fails = 0
    for text, exp_best, exp_money in CASES:
        got_best = A._best_stat_figure(text)
        got_money = A._money_figure(text)
        ok = (got_best == exp_best) and (got_money == exp_money)
        mark = "PASS" if ok else "FAIL"
        if not ok:
            fails += 1
        print(f"  [{mark}] best={got_best!r:<18} money={got_money!r:<9} "
              f"<- {text}")
        if not ok:
            print(f"         expected best={exp_best!r} money={exp_money!r}")
    print(f"\n{len(CASES) - fails}/{len(CASES)} passed"
          + ("" if fails else "  — ALL GREEN"))
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
