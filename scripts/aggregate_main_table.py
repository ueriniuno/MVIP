#!/usr/bin/env python3
"""Aggregate results/per_sample_metrics.csv into the main table."""

from __future__ import annotations

import argparse
import csv
import math
from collections import OrderedDict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "results/per_sample_metrics.csv"
DEFAULT_EXPECTED = ROOT / "results/main_table.csv"
IDENTITY_COLUMNS = {"method", "source", "video_id", "chunk_id"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--check", action="store_true", help="Compare aggregates with the released table.")
    parser.add_argument("--expected", type=Path, default=DEFAULT_EXPECTED)
    return parser.parse_args()


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Missing CSV header: {path}")
        return reader.fieldnames, list(reader)


def aggregate(fieldnames: list[str], rows: list[dict[str, str]]) -> tuple[list[str], list[dict[str, str]]]:
    metric_columns = [name for name in fieldnames if name not in IDENTITY_COLUMNS]
    grouped: OrderedDict[str, list[dict[str, str]]] = OrderedDict()
    for row in rows:
        grouped.setdefault(row["method"], []).append(row)

    output_fields = ["method", "n", "videos", *metric_columns]
    output_rows: list[dict[str, str]] = []
    for method, method_rows in grouped.items():
        output: dict[str, str] = {
            "method": method,
            "n": str(len(method_rows)),
            "videos": str(len({(row["source"], row["video_id"]) for row in method_rows})),
        }
        for metric in metric_columns:
            values = [float(row[metric]) for row in method_rows if row.get(metric, "").strip()]
            output[metric] = repr(math.fsum(values) / len(values)) if values else ""
        output_rows.append(output)
    return output_fields, output_rows


def write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def check(expected_path: Path, actual_rows: list[dict[str, str]], tolerance: float = 1e-12) -> None:
    expected_fields, expected_rows = read_rows(expected_path)
    actual_by_method = {row["method"]: row for row in actual_rows}
    errors: list[str] = []
    for expected in expected_rows:
        method = expected["method"]
        actual = actual_by_method.get(method)
        if actual is None:
            errors.append(f"missing method: {method}")
            continue
        for field in expected_fields:
            left = expected.get(field, "").strip()
            right = actual.get(field, "").strip()
            if not left and not right:
                continue
            if field in {"method", "n", "videos"}:
                if left != right:
                    errors.append(f"{method}/{field}: expected {left}, got {right}")
                continue
            if not left or not right or not math.isclose(float(left), float(right), abs_tol=tolerance, rel_tol=tolerance):
                errors.append(f"{method}/{field}: expected {left}, got {right}")
    if errors:
        raise SystemExit("Aggregate mismatch:\n" + "\n".join(errors))
    print(f"Verified {len(expected_rows)} methods against {expected_path}")


def main() -> None:
    args = parse_args()
    fields, rows = read_rows(args.input)
    output_fields, output_rows = aggregate(fields, rows)
    if args.output is not None:
        write_rows(args.output, output_fields, output_rows)
        print(f"Wrote {args.output}")
    if args.check:
        check(args.expected, output_rows)
    if args.output is None and not args.check:
        for row in output_rows:
            print(row)


if __name__ == "__main__":
    main()
