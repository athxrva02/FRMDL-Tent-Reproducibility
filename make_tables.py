#!/usr/bin/env python3
"""Turn results.csv into the reproduction deliverables.

Reads the tidy CSV from parse_logs.py and writes, into the output dir:
  - table_sev5_<arch>.md   README-format table (mean + 15 corruptions, sev 5)
  - severity_trend.md/.png mean error vs severity (1-5), per method, per arch
  - variance.md            seed-1 vs seed-2 sev-5 mean per config + gap
  - deviation_report.md    reproduced sev-5 cells vs README targets, flagged

Pure analysis (pandas + matplotlib, no torch) -- runs locally on macOS.

Usage (from tent/):
    python repro_a/make_tables.py
    python repro_a/make_tables.py --csv ./output/A/results.csv --out ./output/A
"""
import argparse
import os
import sys

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Canonical corruption order and the short labels the README uses in its header.
CORRUPTION_ORDER = [
    "gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur",
    "glass_blur", "motion_blur", "zoom_blur", "snow", "frost", "fog",
    "brightness", "contrast", "elastic_transform", "pixelate",
    "jpeg_compression",
]
SHORT_LABELS = {
    "gaussian_noise": "gauss_noise", "shot_noise": "shot_noise",
    "impulse_noise": "impulse_noise", "defocus_blur": "defocus_blur",
    "glass_blur": "glass_blur", "motion_blur": "motion_blur",
    "zoom_blur": "zoom_blur", "snow": "snow", "frost": "frost", "fog": "fog",
    "brightness": "brightness", "contrast": "contrast",
    "elastic_transform": "elastic_trans", "pixelate": "pixelate",
    "jpeg_compression": "jpeg",
}
ARCH_PRETTY = {
    "Standard": "WRN-28-10 (Standard)",
    "Hendrycks2020AugMix_WRN": "WRN-40-2 (Hendrycks2020AugMix_WRN)",
}
METHOD_ORDER = ["source", "norm", "tent"]

# README example targets: severity-5 error % in CORRUPTION_ORDER, plus mean.
README_TARGETS = {
    ("Standard", "source"): [72.3, 65.7, 72.9, 46.9, 54.3, 34.8, 42.0, 25.1,
                             41.3, 26.0, 9.3, 46.7, 26.6, 58.5, 30.3],
    ("Standard", "norm"):   [28.1, 26.1, 36.3, 12.8, 35.3, 14.2, 12.1, 17.3,
                             17.4, 15.3, 8.4, 12.6, 23.8, 19.7, 27.3],
    ("Standard", "tent"):   [24.8, 23.5, 33.0, 12.0, 31.8, 13.7, 10.8, 15.9,
                             16.2, 13.7, 7.9, 12.1, 22.0, 17.3, 24.2],
    ("Hendrycks2020AugMix_WRN", "source"): [28.8, 23.0, 26.2, 9.5, 20.6, 10.6,
                                            9.3, 14.2, 15.3, 17.5, 7.6, 20.9,
                                            14.7, 41.3, 14.7],
    ("Hendrycks2020AugMix_WRN", "norm"):   [18.5, 16.2, 22.3, 9.0, 21.9, 10.5,
                                            9.7, 12.8, 13.3, 15.0, 7.6, 11.9,
                                            16.3, 15.0, 17.5],
    ("Hendrycks2020AugMix_WRN", "tent"):   [15.7, 13.2, 18.8, 7.9, 18.1, 9.0,
                                            8.0, 10.4, 10.8, 12.4, 6.7, 10.0,
                                            14.0, 11.4, 14.8],
}
README_MEAN = {  # the README's printed mean (rounded)
    ("Standard", "source"): 43.5, ("Standard", "norm"): 20.4,
    ("Standard", "tent"): 18.6,
    ("Hendrycks2020AugMix_WRN", "source"): 18.3,
    ("Hendrycks2020AugMix_WRN", "norm"): 14.5,
    ("Hendrycks2020AugMix_WRN", "tent"): 12.1,
}


def sev5_row(df, arch, method, seed):
    """Per-corruption error list (in CORRUPTION_ORDER) for one config at sev 5."""
    sub = df[(df.arch == arch) & (df.method == method) & (df.seed == seed)
             & (df.severity == 5)]
    by_corr = sub.set_index("corruption")["error"]
    return [by_corr.get(c, float("nan")) for c in CORRUPTION_ORDER]


def fmt(x):
    return "—" if pd.isna(x) else f"{x:.1f}"


def write_sev5_table(df, arch, seed, out_dir):
    archs_present = [m for m in METHOD_ORDER
                     if not df[(df.arch == arch) & (df.method == m)
                               & (df.seed == seed)].empty]
    if not archs_present:
        return None
    header = ["", "mean"] + [SHORT_LABELS[c] for c in CORRUPTION_ORDER]
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * len(header)) + "|"]
    for method in archs_present:
        vals = sev5_row(df, arch, method, seed)
        mean = pd.Series(vals).mean()
        cells = [method, fmt(mean)] + [fmt(v) for v in vals]
        lines.append("| " + " | ".join(cells) + " |")

    path = os.path.join(out_dir, f"table_sev5_{arch}.md")
    with open(path, "w") as f:
        f.write(f"### {ARCH_PRETTY.get(arch, arch)} — severity 5 (seed {seed})\n\n"
                f"Error (%) across corruption types at the most severe level "
                f"(level 5).\n\n")
        f.write("\n".join(lines) + "\n")
    return path


def write_severity_trend(df, seed, out_dir):
    # Markdown table: mean error per (arch, method, severity)
    grp = (df[df.seed == seed]
           .groupby(["arch", "method", "severity"])["error"].mean()
           .reset_index())
    lines = ["### Severity 1–5 trend (mean error %, seed {})\n".format(seed),
             "| arch | method | sev1 | sev2 | sev3 | sev4 | sev5 |",
             "|---|---|---|---|---|---|---|"]
    archs = [a for a in ARCH_PRETTY if a in grp.arch.unique()]
    for arch in archs:
        for method in METHOD_ORDER:
            row = [fmt(grp[(grp.arch == arch) & (grp.method == method)
                          & (grp.severity == s)]["error"].mean())
                   for s in [1, 2, 3, 4, 5]]
            if all(v == "—" for v in row):
                continue
            lines.append(f"| {arch} | {method} | " + " | ".join(row) + " |")
    md_path = os.path.join(out_dir, "severity_trend.md")
    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    # Plot: one subplot per arch, one line per method.
    if archs:
        fig, axes = plt.subplots(1, len(archs), figsize=(6 * len(archs), 4.5),
                                 squeeze=False)
        for ax, arch in zip(axes[0], archs):
            for method in METHOD_ORDER:
                sub = grp[(grp.arch == arch) & (grp.method == method)
                          ].sort_values("severity")
                if sub.empty:
                    continue
                ax.plot(sub.severity, sub.error, marker="o", label=method)
            ax.set_title(ARCH_PRETTY.get(arch, arch))
            ax.set_xlabel("corruption severity")
            ax.set_ylabel("mean error (%)")
            ax.set_xticks([1, 2, 3, 4, 5])
            ax.grid(True, alpha=0.3)
            ax.legend()
        fig.suptitle(f"CIFAR-10-C error vs severity (seed {seed})")
        fig.tight_layout()
        png_path = os.path.join(out_dir, "severity_trend.png")
        fig.savefig(png_path, dpi=150)
        plt.close(fig)
        return md_path, png_path
    return md_path, None


def write_variance(df, out_dir):
    seeds = sorted(df.seed.unique())
    grp = (df[df.severity == 5]
           .groupby(["arch", "method", "seed"])["error"].mean())
    lines = ["### Seed variance — severity-5 mean error (%)\n",
             "| arch | method | " + " | ".join(f"seed {s}" for s in seeds)
             + " | |Δ| |",
             "|---|---|" + "|".join(["---"] * (len(seeds) + 1)) + "|"]
    for arch in [a for a in ARCH_PRETTY if a in df.arch.unique()]:
        for method in METHOD_ORDER:
            vals = [grp.get((arch, method, s), float("nan")) for s in seeds]
            if all(pd.isna(v) for v in vals):
                continue
            gap = (max(v for v in vals if not pd.isna(v))
                   - min(v for v in vals if not pd.isna(v))) \
                if sum(not pd.isna(v) for v in vals) >= 2 else float("nan")
            cells = [arch, method] + [fmt(v) for v in vals] + [fmt(gap)]
            lines.append("| " + " | ".join(cells) + " |")
    path = os.path.join(out_dir, "variance.md")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path


def write_deviation_report(df, seed, out_dir):
    lines = ["# Deviation report — reproduced vs README example (severity 5, "
             f"seed {seed})\n",
             "Flags: ⚠️ |Δ| > 1pp (note in write-up) · ❗ |Δ| > 2pp "
             "(investigate cuDNN/torch version; upstream pinned torch 1.8.1).\n"]
    n_note = n_inv = 0
    for arch in ARCH_PRETTY:
        for method in METHOD_ORDER:
            if (arch, method) not in README_TARGETS:
                continue
            repro = sev5_row(df, arch, method, seed)
            if all(pd.isna(v) for v in repro):
                continue
            target = README_TARGETS[(arch, method)]
            repro_mean = pd.Series(repro).mean()
            tgt_mean = README_MEAN[(arch, method)]
            dmean = repro_mean - tgt_mean
            flag = "❗" if abs(dmean) > 2 else ("⚠️" if abs(dmean) > 1 else "✅")
            lines.append(f"\n## {ARCH_PRETTY[arch]} · {method}\n")
            lines.append(f"- **mean**: reproduced {fmt(repro_mean)} vs README "
                         f"{tgt_mean} → Δ {dmean:+.1f}pp {flag}")
            # Per-corruption offenders
            offenders = []
            for c, r, t in zip(CORRUPTION_ORDER, repro, target):
                if pd.isna(r):
                    continue
                d = r - t
                if abs(d) > 1:
                    mark = "❗" if abs(d) > 2 else "⚠️"
                    offenders.append(f"{SHORT_LABELS[c]} {d:+.1f}pp {mark}")
                    if abs(d) > 2:
                        n_inv += 1
                    else:
                        n_note += 1
            if offenders:
                lines.append("- per-corruption Δ>1pp: " + ", ".join(offenders))
            else:
                lines.append("- all per-corruption cells within 1pp ✅")
    summary = (f"\n---\n**Summary:** {n_note} cell(s) flagged ⚠️ (>1pp), "
               f"{n_inv} cell(s) flagged ❗ (>2pp).\n")
    lines.append(summary)
    path = os.path.join(out_dir, "deviation_report.md")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default="./output/A/results.csv")
    ap.add_argument("--out", default="./output/A")
    ap.add_argument("--seed", type=str, default="1",
                    help="seed used for the headline tables/report (default 1)")
    args = ap.parse_args()

    if not os.path.isfile(args.csv):
        sys.exit(f"[make_tables] CSV not found: {args.csv} "
                 f"(run parse_logs.py first)")
    df = pd.read_csv(args.csv, dtype={"seed": str})
    os.makedirs(args.out, exist_ok=True)

    if args.seed not in set(df.seed.unique()):
        avail = ", ".join(sorted(df.seed.unique()))
        print(f"[make_tables] seed {args.seed} not in results (have: {avail}); "
              f"using first available.")
        args.seed = sorted(df.seed.unique())[0]

    written = []
    for arch in ARCH_PRETTY:
        if arch in df.arch.unique():
            p = write_sev5_table(df, arch, args.seed, args.out)
            if p:
                written.append(p)
    md, png = write_severity_trend(df, args.seed, args.out)
    written.append(md)
    if png:
        written.append(png)
    written.append(write_variance(df, args.out))
    written.append(write_deviation_report(df, args.seed, args.out))

    print("[make_tables] wrote:")
    for p in written:
        print("  " + p)


if __name__ == "__main__":
    main()
