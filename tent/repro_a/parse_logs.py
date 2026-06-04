#!/usr/bin/env python3
"""Parse upstream cifar10c.py logs into a tidy CSV (Student A, "Reproduced").

cifar10c.py logs one line per (corruption, severity) in the form
    [..] [cifar10c.py:   50]: error % [gaussian_noise5]: 24.80%
and dumps the full YACS config near the top of each log. We walk a tree of
such logs and emit one row per (arch, method, seed, severity, corruption).

The mean error per (arch, method, seed, severity) is NOT logged upstream; it is
computed downstream in make_tables.py as the average of the 15 per-corruption
errors (this matches the README "mean" column).

This script is pure text processing -- no torch -- so it runs locally on macOS.
Only the standard library is used.

Usage (from tent/):
    python repro_a/parse_logs.py                       # reads ./output/A
    python repro_a/parse_logs.py --root ./output/A --out ./output/A/results.csv
"""
import argparse
import csv
import os
import re
import sys

# "error % [gaussian_noise5]: 24.80%"  ->  ("gaussian_noise", "5", "24.80")
ERROR_RE = re.compile(r"error % \[([a-z_]+)(\d)\]:\s*([\d.]+)%")

# YACS config dump fallbacks (used only when the path does not encode the field)
ARCH_RE = re.compile(r"\bARCH:\s*(\S+)")
ADAPT_RE = re.compile(r"\bADAPTATION:\s*(\S+)")
SEED_RE = re.compile(r"\bRNG_SEED:\s*(\d+)")


def fields_from_path(log_path, root):
    """Recover (arch, method, seed) from output/A/<arch>/<method>/seed<seed>/."""
    rel = os.path.relpath(log_path, root)
    parts = rel.split(os.sep)
    arch = method = seed = None
    if len(parts) >= 4:  # arch / method / seed<seed> / logfile.txt
        arch, method, seed_dir = parts[0], parts[1], parts[2]
        m = re.fullmatch(r"seed(\d+)", seed_dir)
        seed = m.group(1) if m else None
    return arch, method, seed


def fields_from_content(text):
    """Recover (arch, method, seed) from the logged YACS config dump."""
    arch = ARCH_RE.search(text)
    method = ADAPT_RE.search(text)
    seed = SEED_RE.search(text)
    return (
        arch.group(1) if arch else None,
        method.group(1) if method else None,
        seed.group(1) if seed else None,
    )


def parse_log(log_path, root):
    with open(log_path, "r", errors="replace") as f:
        text = f.read()

    rows = ERROR_RE.findall(text)
    if not rows:
        return []  # not an evaluation log (e.g. an empty/failed run)

    arch, method, seed = fields_from_path(log_path, root)
    c_arch, c_method, c_seed = fields_from_content(text)
    arch = arch or c_arch
    method = method or c_method
    seed = seed or c_seed

    if not all([arch, method, seed]):
        print(f"[parse_logs] WARN: could not identify arch/method/seed for "
              f"{log_path} (arch={arch}, method={method}, seed={seed})",
              file=sys.stderr)

    out = []
    for corruption, severity, err in rows:
        out.append({
            "arch": arch,
            "method": method,
            "seed": seed,
            "severity": int(severity),
            "corruption": corruption,
            "error": float(err),
        })
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="./output/A",
                    help="directory tree of cifar10c.py logs (default ./output/A)")
    ap.add_argument("--out", default="./output/A/results.csv",
                    help="output CSV path (default ./output/A/results.csv)")
    args = ap.parse_args()

    if not os.path.isdir(args.root):
        sys.exit(f"[parse_logs] root not found: {args.root}")

    all_rows = []
    for dirpath, _, filenames in os.walk(args.root):
        for fn in filenames:
            if fn.endswith(".txt"):
                all_rows.extend(parse_log(os.path.join(dirpath, fn), args.root))

    if not all_rows:
        sys.exit(f"[parse_logs] no error lines found under {args.root}")

    # Deduplicate (a re-run can leave multiple timestamped logs in one dir):
    # keep the last value seen for a given (arch, method, seed, severity, corruption).
    keyed = {}
    for r in all_rows:
        keyed[(r["arch"], r["method"], r["seed"], r["severity"], r["corruption"])] = r
    rows = sorted(keyed.values(),
                  key=lambda r: (str(r["arch"]), str(r["method"]), str(r["seed"]),
                                 r["severity"], r["corruption"]))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "arch", "method", "seed", "severity", "corruption", "error"])
        w.writeheader()
        w.writerows(rows)

    print(f"[parse_logs] wrote {len(rows)} rows to {args.out}")
    # Row count depends on scope, so we report it rather than assert a fixed
    # number: Student A's reduced reproduction is 4 runs x 15 corruptions
    # (severity 5) = 60 rows; Student B's ablation sweeps add more.


if __name__ == "__main__":
    main()
