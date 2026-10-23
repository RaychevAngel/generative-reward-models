#!/usr/bin/env python3
"""Rebuild vector empirical figures from retained, read-only paper CSVs.

No model execution, network access, additional smoothing, new curve fitting,
inferred coordinates, or interpolation of missing snapshots. Historical KL-axis
preprocessing is not fully reconstructed; exports are not assumed raw measured
KL. Requires only ReportLab and the stdlib.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

from reportlab.lib.colors import HexColor
from reportlab.pdfgen import canvas


BLUE = "#1765A3"
ORANGE = "#C75D16"
INK = "#202D3A"
MUTED = "#526172"
GRID = "#DFE5EB"
MODES = {"ans": ("Answer-Only", BLUE), "think": ("Thinking", ORANGE)}
CONFIGS = [
    ("0.6b", "4b"), ("1.7b", "4b"), ("4b", "4b"), ("8b", "4b"),
    ("4b", "0.6b"), ("4b", "1.7b"), ("4b", "8b"), ("4b", "14b"),
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ratings(path: Path) -> dict[str, float]:
    rows = read_csv(path)
    result = {row["player"]: float(row["rating"]) for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicate player IDs in {path}")
    return result


def trajectory(table: dict[str, float], prefix: str) -> list[tuple[int, float]]:
    return sorted(
        (int(player.rsplit("_", 1)[1]), value)
        for player, value in table.items()
        if player.rsplit("_", 1)[0] == prefix
    )


def series(points, color, dashed=False, markers=True):
    return {"points": points, "color": color, "dashed": dashed, "markers": markers}


def label(c, x, y, text, size=9, bold=False, color=INK, align="left"):
    c.setFillColor(HexColor(color))
    c.setFont("Helvetica-Bold" if bold else "Helvetica", size)
    getattr(c, {"left": "drawString", "center": "drawCentredString", "right": "drawRightString"}[align])(x, y, text)


def legend(c, items, x, y, step=143, size=8.7):
    for index, (text, color, dashed) in enumerate(items):
        sx = x + index * step
        c.setStrokeColor(HexColor(color))
        c.setLineWidth(1.6)
        c.setDash([4, 2] if dashed else [])
        c.line(sx, y + 2, sx + 19, y + 2)
        c.setDash([])
        label(c, sx + 25, y - 1, text, size=size)


def chart(c, box, title, curves, xlim, ylim, xticks, yticks,
          xlabel="Policy training step", ylabel="Gold Elo", logx=False,
          mark=None):
    x, y, width, height = box
    left, right, bottom, top = 44, 10, 31, 25
    px, py = x + left, y + bottom
    pw, ph = width - left - right, height - bottom - top
    tx = (lambda value: math.log(value)) if logx else (lambda value: value)
    lo, hi = tx(xlim[0]), tx(xlim[1])
    xx = lambda value: px + (tx(value) - lo) / (hi - lo) * pw
    yy = lambda value: py + (value - ylim[0]) / (ylim[1] - ylim[0]) * ph
    label(c, x + left, y + height - 13, title, 10, True)
    c.setLineWidth(0.45)
    c.setStrokeColor(HexColor(GRID))
    for value in yticks:
        if ylim[0] <= value <= ylim[1]:
            c.line(px, yy(value), px + pw, yy(value))
            label(c, px - 6, yy(value) - 3, f"{value:g}", 8.3, color=MUTED, align="right")
    for value in xticks:
        if xlim[0] <= value <= xlim[1]:
            c.line(xx(value), py, xx(value), py - 3)
            label(c, xx(value), py - 13, f"{value:g}", 8.3, color=MUTED, align="center")
    c.setStrokeColor(HexColor(MUTED))
    c.line(px, py, px + pw, py)
    c.line(px, py, px, py + ph)
    label(c, px + pw / 2, y + 2, xlabel, 8.5, color=MUTED, align="center")
    c.saveState()
    c.translate(x + 9, py + ph / 2)
    c.rotate(90)
    label(c, 0, 0, ylabel, 8.5, color=MUTED, align="center")
    c.restoreState()
    c.saveState()
    clip = c.beginPath()
    clip.rect(px - 1, py - 1, pw + 2, ph + 2)
    c.clipPath(clip, stroke=0)
    for curve in curves:
        points = curve["points"]
        c.setStrokeColor(HexColor(curve["color"]))
        c.setFillColor(HexColor(curve["color"]))
        c.setLineWidth(1.15)
        c.setDash([4, 2] if curve["dashed"] else [])
        path = c.beginPath()
        for index, (vx, vy) in enumerate(points):
            if index == 0:
                path.moveTo(xx(vx), yy(vy))
            else:
                path.lineTo(xx(vx), yy(vy))
        if len(points) > 1:
            c.drawPath(path)
        c.setDash([])
        if curve["markers"]:
            for vx, vy in points:
                c.circle(xx(vx), yy(vy), 0.95, stroke=0, fill=1)
    if mark is not None:
        vx, values = mark
        c.setStrokeColor(HexColor(MUTED))
        c.setDash([1, 2])
        c.line(xx(vx), py, xx(vx), py + ph)
        c.setDash([])
        for vy, color in values:
            c.setStrokeColor(HexColor(color))
            c.setLineWidth(1.0)
            c.circle(xx(vx), yy(vy), 3.0, stroke=1, fill=0)
    c.restoreState()


def new_canvas(path, width, height):
    c = canvas.Canvas(str(path), pagesize=(width, height), invariant=1, pageCompression=1)
    c.setTitle("Retained empirical trajectories: Scaling Laws for Generative Reward Models")
    c.setAuthor("Angel Raychev, Nathan Lile, Rafael Rafailov")
    return c


def panel_grid(path, title, configs, global_gold, kl_lookup=None):
    width, height = 468, 342
    c = new_canvas(path, width, height)
    label(c, 5, height - 14, title, 11.2, True)
    legend(c, [(name, color, False) for name, color in MODES.values()], 130, height - 32, 130)
    for index, (policy, judge) in enumerate(configs):
        column, row = index % 2, index // 2
        box = (column * 234, 4 + (1 - row) * 150, 230, 149)
        curves = []
        kl_values = []
        for mode, (_, color) in MODES.items():
            prefix = f"{mode}_{policy}_{judge}"
            values = trajectory(global_gold, prefix)
            if kl_lookup is None:
                points = values
            else:
                points = [
                    (kl_lookup[f"{prefix}_{step}"], value)
                    for step, value in values if f"{prefix}_{step}" in kl_lookup
                ]
                kl_values += [item[0] for item in points]
            curves.append(series(points, color))
        if kl_lookup is None:
            xlim, xticks, xlabel = (0, 480), [0, 100, 200, 300, 400], "Policy training step"
        else:
            minimum, maximum = min(kl_values), max(kl_values)
            span = maximum - minimum
            xlim = (min(-0.025, minimum - span * 0.025), maximum + span * 0.04)
            # Decimal ticks remain readable for low-KL and higher-KL cells.
            raw = maximum / 3
            power = 10 ** math.floor(math.log10(raw))
            delta = next(v * power for v in [1, 2, 5, 10] if v * power >= raw)
            xticks = [delta * n for n in range(4) if delta * n <= xlim[1]]
            xlabel = "Exported KL coordinate"
        chart(c, box, f"Policy {policy.upper()} / judge {judge.upper()}", curves,
              xlim, (-500, 1900), xticks, [-500, 0, 500, 1000, 1500], xlabel=xlabel)
    c.save()


def baseline_figure(path, result_root):
    folder = result_root / "trained_vs_baseline"
    gold = ratings(folder / "gold.csv")
    configs = [
        ("ans_4b_4b", "4b_answer", "Trained Answer-Only", BLUE, False),
        ("think_4b_4b", "4b_thinking", "Trained Thinking", ORANGE, False),
        ("ans_off_4b_4b", "4b_off_answer", "Base Answer-Only", BLUE, True),
        ("think_off_4b_4b", "4b_off_thinking", "Base Thinking", ORANGE, True),
    ]
    c = new_canvas(path, 468, 453)
    label(c, 5, 439, "Trained and off-the-shelf rewarders: observed trajectories", 11.2, True)
    legend(c, [(row[2], row[3], row[4]) for row in configs[:2]], 82, 420, 160)
    legend(c, [(row[2], row[3], row[4]) for row in configs[2:]], 82, 405, 160)
    chart(c, (0, 219, 468, 177), "(a) One common Gold observer", [
        series(trajectory(gold, prefix), color, dashed)
        for prefix, _, _, color, dashed in configs
    ], (0, 480), (-150, 1100), [0, 100, 200, 300, 400, 460], [0, 250, 500, 750, 1000],
          mark=(460, [(gold["think_4b_4b_460"], ORANGE), (gold["think_off_4b_4b_460"], ORANGE)]))
    for index, (prefix, filename, name, color, dashed) in enumerate(configs):
        values = trajectory(ratings(folder / f"{filename}.csv"), prefix)
        low = min(0.0, min(v for _, v in values))
        high = max(v for _, v in values)
        delta = 500 if high > 1500 else 250
        top = math.ceil(high / delta) * delta
        bottom = math.floor(low / delta) * delta
        chart(c, ((index % 2) * 234, (1 - index // 2) * 108, 230, 106),
              f"({chr(98 + index)}) Own proxy: {name}", [series(values, color, dashed)],
              (0, 480), (bottom, top + delta * .12), [0, 200, 400],
              list(range(bottom, top + 1, delta)), ylabel="Own-proxy Elo")
    c.save()
    return gold


def capacity_figure(path, global_gold):
    c = new_canvas(path, 468, 224)
    label(c, 5, 210, "Capacity sweeps in the common Gold arena", 11.2, True)
    legend(c, [(name, color, False) for name, color in MODES.values()], 130, 191, 130)
    summaries = {}
    for policy, judge in CONFIGS:
        for mode in MODES:
            values = trajectory(global_gold, f"{mode}_{policy}_{judge}")
            ordered = sorted((v for _, v in values), reverse=True)
            summaries[(mode, policy, judge)] = {"peak": ordered[0], "top5": sum(ordered[:5]) / 5,
                                               "snapshots": len(values)}
    for index, axis in enumerate(["judge", "policy"]):
        sizes = [0.6, 1.7, 4, 8, 14] if axis == "judge" else [0.6, 1.7, 4, 8]
        curves = []
        for mode, (_, color) in MODES.items():
            points = []
            for size in sizes:
                token = f"{size:g}b"
                key = (mode, "4b", token) if axis == "judge" else (mode, token, "4b")
                points.append((size, summaries[key]["top5"]))
            curves.append(series(points, color))
        chart(c, (index * 234, 5, 230, 173),
              "(a) Fixed 4B policy" if axis == "judge" else "(b) Fixed 4B judge", curves,
              (sizes[0] * .85, sizes[-1] * 1.17), (900, 1800), sizes, [1000, 1200, 1400, 1600, 1800],
              xlabel=f"{axis.capitalize()} parameters (billions)", ylabel="Top-five mean Gold Elo", logx=True)
    c.save()
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    source = root / "paper/figures/results"
    output = root / "arxiv/figures/restored"
    provenance = root / "repro_artifacts/empirical_supplement"
    output.mkdir(parents=True, exist_ok=True)
    provenance.mkdir(parents=True, exist_ok=True)
    global_path = source / "all_params/gold.csv"
    global_gold = ratings(global_path)
    input_paths = {global_path}
    joins = []
    kl_lookup = {}
    for policy, judge in CONFIGS:
        path = source / f"think_vs_ans_{policy}_{judge}/kl_ans_vs_think_{policy}_{judge}.csv"
        input_paths.add(path)
        for source_row, row in enumerate(read_csv(path), 2):
            if row["source"] != "gold":
                continue
            candidates = [
                player for player, rating in global_gold.items()
                if player.rsplit("_", 1)[0] == row["prefix"]
                and abs(rating - float(row["rating"])) < 1e-7
            ]
            if len(candidates) != 1:
                raise ValueError(f"KL join not unique: {path}:{source_row}: {candidates}")
            player = candidates[0]
            if player in kl_lookup:
                raise ValueError(f"Duplicate exported KL coordinate for {player}")
            kl_lookup[player] = float(row["kl"])
            joins.append({"source": str(path.relative_to(root)), "source_row": source_row,
                          "player": player, "step": int(player.rsplit("_", 1)[1]),
                          "kl": float(row["kl"]), "gold_elo": global_gold[player]})
    with (provenance / "exported_kl_joins.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, list(joins[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(joins)
    panel_grid(output / "main-selected-trajectories.pdf",
               "Representative Gold-evaluated policy trajectories",
               [("0.6b", "4b"), ("4b", "4b"), ("8b", "4b"), ("4b", "14b")], global_gold)
    for half, configs in zip("ab", [CONFIGS[:4], CONFIGS[4:]]):
        panel_grid(output / f"gold-trajectories-{half}.pdf",
                   "Gold-evaluated policy training trajectories", configs, global_gold)
        panel_grid(output / f"kl-trajectories-{half}.pdf",
                   "Gold trajectories against exported KL coordinates", configs, global_gold, kl_lookup)
    baseline_gold = baseline_figure(output / "baseline-trajectories.pdf", source)
    input_paths.update((source / "trained_vs_baseline").glob("*.csv"))
    summaries = capacity_figure(output / "capacity-allocation.pdf", global_gold)
    summary_rows = [dict(mode=m, policy=p, judge=j, **values)
                    for (m, p, j), values in sorted(summaries.items())]
    with (provenance / "configuration_summaries.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, list(summary_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary_rows)
    expected_players = {f"{mode}_{policy}_{judge}_{step}"
                        for policy, judge in CONFIGS for mode in MODES
                        for step, _ in trajectory(global_gold, f"{mode}_{policy}_{judge}")}
    manifest = {
        "method": "retained_csv_exports_only_no_model_calls_no_new_smoothing_or_fits",
        "gold_arena": str((source / "all_params/gold.csv").relative_to(root)) + " for all eight matched cells and capacity sweeps",
        "baseline_arena": str((source / "trained_vs_baseline/gold.csv").relative_to(root)) + "; separate from all_params",
        "own_proxy_arenas": "Each baseline own-proxy panel uses its own evaluator CSV; no cross-panel rating comparability claimed",
        "kl_join": {"key": "exact prefix plus unique Gold rating, tolerance 1e-7", "joined_rows": len(joins),
                    "global_snapshots": len(expected_players),
                    "snapshots_without_exported_kl": sorted(expected_players - set(kl_lookup)),
                    "negative_exported_coordinates_retained": sum(value < 0 for value in kl_lookup.values()),
                    "no_additional_coordinate_inference": True,
                    "joined_coordinate_export": "repro_artifacts/empirical_supplement/exported_kl_joins.csv",
                    "historical_preprocessing": "Exact preprocessing of these matched-setting exports is not fully reconstructed; they must not be assumed raw measured KL.",
                    "related_beta_provenance": "The separate beta sweep is known to use historically Savitzky-Golay-smoothed coordinates; this does not establish the exact preprocessing of the matched-setting exports.",
                    "negative_coordinates": "Retained as exported; not interpreted as negative true KL or attributed to a particular estimator."},
        "late_same_step_thinking": {"step": 460,
            "trained_gold": baseline_gold["think_4b_4b_460"],
            "baseline_gold": baseline_gold["think_off_4b_4b_460"],
            "interpretation": "observed crossover, not a significance result or uniformly superior baseline"},
        "inputs": [{"path": str(path.relative_to(root)), "sha256": digest(path),
                    "rows": len(read_csv(path))} for path in sorted(input_paths)],
        "outputs": [{"path": str(path.relative_to(root)), "sha256": digest(path)}
                    for path in sorted(output.glob("*.pdf"))],
        "limitations": ["Snapshots are not independent training repetitions",
                        "No new hypothesis test, model judging or generation",
                        "Top-five summary is selected on Gold, not a held-out deployment rule",
                        "Historical KL-axis preprocessing is not fully reconstructed; exported coordinates are not assumed raw measured KL",
                        "Exported KL coordinates do not establish a calibrated OOD distance to source stories",
                        "Lines connect retained coordinates only; no missing snapshots filled"]}
    (provenance / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"pdfs": len(manifest["outputs"]), "kl_joins": len(joins),
                      "missing_kl": manifest["kl_join"]["snapshots_without_exported_kl"],
                      "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
