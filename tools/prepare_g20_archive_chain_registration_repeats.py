#!/usr/bin/env python3
"""Handle comma-separated zero repeats, then run archive-chain preparation."""

from __future__ import annotations

from tools import prepare_g20_archive_chain_registration as prepare


_dict_reader = prepare.csv.DictReader


def _first_repeat_reader(*args, **kwargs):
    for row in _dict_reader(*args, **kwargs):
        for key, value in row.items():
            if value and "," in value and "summary.json" in value:
                row[key] = value.split(",", 1)[0]
        yield row


def main() -> int:
    prepare.csv.DictReader = _first_repeat_reader
    return prepare.main()


if __name__ == "__main__":
    raise SystemExit(main())
