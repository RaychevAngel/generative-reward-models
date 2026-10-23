#!/usr/bin/env python3
"""Replot retained main-text data as vector figures, without fitting or inference.

Every point retains its exported coordinates. Lines follow training chronology,
not a new polynomial fit and not an ordering of the x-axis. Gold ratings use each
figure family's retained local arena, not the all-configuration arena. Own-proxy
ratings use separate evaluators and do not share a calibrated scale. See the
generated manifest for input hashes, exact joins, preprocessing and missingness.
The historical beta KL axis was smoothed before CSV export; this builder does
not recover unsmoothed measurements or add further smoothing.

Requires ReportLab and Python's standard library only; no network access.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

from reportlab.lib.colors import HexColor
from reportlab.pdfgen import canvas


INK = "#202D3A"
MUTED = "#526172"
GRID = "#DFE5EB"
COLORS = ["#1765A3", "#C75D16", "#32825D", "#8156A2", "#B24C61"]
INPUTS: set[Path] = set()
JOINS: list[dict] = []


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    INPUTS.add(path)
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def ratings(path):
    records = rows(path)
    result = {r["player"]: float(r["rating"]) for r in records}
    if len(result) != len(records):
        raise ValueError(f"Duplicate player ID: {path.name}")
    if not all(math.isfinite(v) for v in result.values()):
        raise ValueError(f"Nonfinite rating: {path.name}")
    return result


def select(table, prefix):
    result = sorted((int(p.rsplit("_", 1)[1]), p, v) for p, v in table.items()
                    if p.rsplit("_", 1)[0] == prefix)
    if not result:
        raise ValueError(f"No observations for {prefix}")
    return result


def joined(folder, filename, prefix, title, color, dashed=False):
    proxy = ratings(folder / f"{filename}.csv")
    gold = ratings(folder / "gold.csv")
    selected = select(proxy, prefix)
    missing = [p for _, p, _ in selected if p not in gold]
    if missing:
        raise ValueError(f"Missing Gold rows for {prefix}: {missing}")
    points = [(value, gold[player], step) for step, player, value in selected]
    JOINS.append({"family": folder.name, "prefix": prefix,
                  "proxy_file": f"{filename}.csv", "gold_file": "gold.csv",
                  "points": len(points), "first_step": points[0][2],
                  "last_step": points[-1][2], "join": "exact player ID",
                  "missing_gold_players": missing})
    return {"name": title, "color": color, "dashed": dashed, "points": points}


def text(c, x, y, value, size=9, bold=False, color=INK, align="left"):
    c.setFont("Helvetica-Bold" if bold else "Helvetica", size)
    c.setFillColor(HexColor(color))
    getattr(c, {"left": "drawString", "center": "drawCentredString", "right": "drawRightString"}[align])(x, y, str(value))


def nice_bounds(values, count=4):
    low, high = min(values), max(values)
    if low == high:
        low, high = low - 1, high + 1
    span = high - low
    target = span / count
    power = 10 ** math.floor(math.log10(target))
    step = next(v * power for v in [1, 2, 2.5, 5, 10] if v * power >= target)
    pad = span * .045
    first = math.ceil((low - pad) / step) * step
    last = math.floor((high + pad) / step) * step
    ticks = [first + i * step for i in range(round((last - first) / step) + 1)]
    return (low - pad, high + pad), ticks


def tick_label(value):
    if abs(value) < 1e-10:
        return "0"
    return f"{value:g}"


def plot(c, box, curves, title, xlabel="Own-proxy Elo", ylabel="Gold Elo", legend_columns=3):
    x, y, width, height = box
    legend_rows = math.ceil(len(curves) / legend_columns)
    left, right, bottom, top = 55, 10, 43, 32 + legend_rows * 20
    px, py = x + left, y + bottom
    pw, ph = width - left - right, height - bottom - top
    xlim, xticks = nice_bounds([p[0] for curve in curves for p in curve["points"]])
    ylim, yticks = nice_bounds([p[1] for curve in curves for p in curve["points"]])
    xx = lambda value: px + (value - xlim[0]) / (xlim[1] - xlim[0]) * pw
    yy = lambda value: py + (value - ylim[0]) / (ylim[1] - ylim[0]) * ph
    text(c, x + left, y + height - 14, title, 12, True)
    column_width = pw / legend_columns
    for index, curve in enumerate(curves):
        sx = px + (index % legend_columns) * column_width
        sy = y + height - 35 - (index // legend_columns) * 20
        c.setStrokeColor(HexColor(curve["color"]))
        c.setLineWidth(1.5)
        c.setDash([4, 2] if curve["dashed"] else [])
        c.line(sx, sy + 2, sx + 15, sy + 2)
        c.setDash([])
        text(c, sx + 20, sy - 1, curve["name"], 11.5)
    c.setLineWidth(.45)
    c.setStrokeColor(HexColor(GRID))
    for value in yticks:
        c.line(px, yy(value), px + pw, yy(value))
        text(c, px - 7, yy(value) - 4, tick_label(value), 11.5, color=MUTED, align="right")
    for value in xticks:
        c.line(xx(value), py, xx(value), py - 3)
        text(c, xx(value), py - 17, tick_label(value), 11.5, color=MUTED, align="center")
    c.setStrokeColor(HexColor(MUTED))
    c.line(px, py, px + pw, py)
    c.line(px, py, px, py + ph)
    text(c, px + pw / 2, y + 5, xlabel, 12, color=MUTED, align="center")
    c.saveState()
    c.translate(x + 10, py + ph / 2)
    c.rotate(90)
    text(c, 0, 0, ylabel, 12, color=MUTED, align="center")
    c.restoreState()
    c.saveState()
    clipping = c.beginPath()
    clipping.rect(px - 1, py - 1, pw + 2, ph + 2)
    c.clipPath(clipping, stroke=0)
    for curve in curves:
        c.setStrokeColor(HexColor(curve["color"]))
        c.setFillColor(HexColor(curve["color"]))
        c.setLineWidth(1.2)
        c.setDash([4, 2] if curve["dashed"] else [])
        path = c.beginPath()
        for index, (vx, vy, _) in enumerate(curve["points"]):
            if index == 0:
                path.moveTo(xx(vx), yy(vy))
            else:
                path.lineTo(xx(vx), yy(vy))
        c.drawPath(path)
        c.setDash([])
        for vx, vy, _ in curve["points"]:
            c.circle(xx(vx), yy(vy), 1.1, fill=1, stroke=0)
        first, last = curve["points"][0], curve["points"][-1]
        # Hollow square = first retained snapshot; hollow circle = last.
        c.setFillColor(HexColor("#FFFFFF"))
        c.rect(xx(first[0]) - 2.3, yy(first[1]) - 2.3, 4.6, 4.6, fill=1, stroke=1)
        c.circle(xx(last[0]), yy(last[1]), 3, fill=1, stroke=1)
    c.restoreState()


def save(path, curves, title, width=330, height=250, columns=3):
    c = canvas.Canvas(str(path), pagesize=(width, height), invariant=1, pageCompression=1)
    c.setTitle(title)
    c.setAuthor("Angel Raychev, Nathan Lile, Rafael Rafailov")
    plot(c, (0, 12, width, height - 12), curves, title, legend_columns=columns)
    text(c, width / 2, 2, "Training order; square: first, circle: last", 10, color=MUTED, align="center")
    c.save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    source = (args.results_dir or root / "paper/figures/results").resolve()
    output = (args.output_dir or root / "arxiv/figures/replotted").resolve()
    manifest_path = args.manifest or root / "repro_artifacts/main_figure_replots/manifest.json"
    output.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    outputs = []
    plotted_points = []

    def record_points(figure, curves):
        for curve in curves:
            for vx, vy, step in curve["points"]:
                plotted_points.append({"figure": figure, "series": curve["name"],
                                       "step": step, "x": vx, "y": vy})

    def emit(filename, curves, title, **kwargs):
        path = output / f"{filename}.pdf"
        save(path, curves, title, **kwargs)
        record_points(filename, curves)
        outputs.append(path)

    baseline = [
        ("ans_4b_4b", "4b_answer", "Trained Answer-Only", COLORS[0], False),
        ("think_4b_4b", "4b_thinking", "Trained Thinking", COLORS[1], False),
        ("ans_off_4b_4b", "4b_off_answer", "Base Answer-Only", COLORS[0], True),
        ("think_off_4b_4b", "4b_off_thinking", "Base Thinking", COLORS[1], True),
    ]
    emit("trained-vs-baseline", [joined(source / "trained_vs_baseline", file, prefix, name, color, dashed)
                                for prefix, file, name, color, dashed in baseline],
         "4B policy / 4B rewarders", columns=2)

    for mode, mode_title in [("ans", "Answer-Only"), ("think", "Thinking")]:
        suffix = "answer" if mode == "ans" else "thinking"
        emit(f"judge-sizes-{suffix}", [joined(source / f"genrm_sizes_{suffix}", f"{size}_{suffix}",
             f"{mode}_4b_{size}", f"{size.upper()} judge", color) for size, color in
             zip(["0.6b", "1.7b", "4b", "8b", "14b"], COLORS)], f"4B policy / {mode_title} rewarders")
        emit(f"policy-sizes-{suffix}", [joined(source / f"policy_sizes_{suffix}", f"4b_{suffix}",
             f"{mode}_{size}_4b", f"{size.upper()} policy", color) for size, color in
             zip(["0.6b", "1.7b", "4b", "8b"], COLORS)], f"4B {mode_title} rewarder / policy-size sweep")

    betas = ["0.1", "0.02", "0.01", "0.005", "0.002"]
    emit("beta-proxy-gold", [joined(source / "beta_coefficients", "4b_answer", f"ans_{beta}",
         f"beta = {beta}", color) for beta, color in zip(betas, COLORS)], "Online DPO regularization sweep")

    kl_rows = rows(source / "beta_coefficients/kl_betas.csv")
    c = canvas.Canvas(str(output / "beta-kl.pdf"), pagesize=(660, 285), invariant=1, pageCompression=1)
    c.setTitle("Online DPO beta sweep: exported KL coordinates and evaluator ratings")
    c.setAuthor("Angel Raychev, Nathan Lile, Rafael Rafailov")
    kl_audit = []
    for index, evaluator in enumerate(["gold", "proxy"]):
        table = ratings(source / "beta_coefficients" / ("gold.csv" if evaluator == "gold" else "4b_answer.csv"))
        curves = []
        for beta, color in zip(betas, COLORS):
            prefix = f"ans_{beta}"
            points = []
            for row in kl_rows:
                if row["prefix"] != prefix or row["source"] != evaluator:
                    continue
                candidates = [step for step, _, value in select(table, prefix)
                              if abs(value - float(row["rating"])) < 1e-7]
                if len(candidates) != 1:
                    raise ValueError(f"Nonunique KL join: {prefix}/{evaluator}: {candidates}")
                points.append((float(row["kl"]), float(row["rating"]), candidates[0]))
            points.sort(key=lambda p: p[2])
            if len({p[2] for p in points}) != len(points):
                raise ValueError(f"Duplicate KL join: {prefix}/{evaluator}")
            curves.append({"name": f"beta = {beta}", "color": color, "dashed": False, "points": points})
            kl_audit.append({"prefix": prefix, "source": evaluator, "points": len(points),
                             "omitted_rating_steps": sorted({s for s, _, _ in select(table, prefix)} - {p[2] for p in points}),
                             "negative_exported_coordinates": sum(p[0] < 0 for p in points)})
        plot(c, (index * 330, 12, 326, 273), curves,
             "Gold-evaluated performance" if evaluator == "gold" else "Own-proxy-evaluated performance",
             xlabel="Exported KL coordinate", ylabel="Gold Elo" if evaluator == "gold" else "Own-proxy Elo")
        record_points(f"beta-kl-{evaluator}", curves)
    text(c, 330, 2, "Historical KL-axis exports; square: first, circle: last", 10, color=MUTED, align="center")
    c.save()
    outputs.append(output / "beta-kl.pdf")

    for policy, judge in [("0.6b", "4b"), ("1.7b", "4b"), ("4b", "4b"), ("8b", "4b"),
                          ("4b", "0.6b"), ("4b", "1.7b"), ("4b", "8b"), ("4b", "14b")]:
        family = f"think_vs_ans_{policy}_{judge}"
        emit(f"matched-{policy}-{judge}", [joined(source / family, f"{judge}_{suffix}", f"{mode}_{policy}_{judge}",
             title, color) for mode, suffix, title, color in [("ans", "answer", "Answer-Only", COLORS[0]),
                                                           ("think", "thinking", "Thinking", COLORS[1])]],
             f"Policy {policy.upper()} / judge {judge.upper()}", width=300, height=250, columns=2)

    def relative(path):
        try:
            return str(path.relative_to(root))
        except ValueError:
            # Portable inventory: never emit private absolute paths.
            return str(path.relative_to(source)) if path.is_relative_to(source) else path.name

    points_path = manifest_path.parent / "plotted_points.csv"
    with points_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["figure", "series", "step", "x", "y"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(plotted_points)

    manifest = {
        "method": "retained exports; replotter adds no smoothing, fitting, interpolation, inferred coordinates or model calls",
        "proxy_gold_join": "exact player ID and exact trajectory-prefix equality; chronological order from final integer token",
        "arenas": "Each figure uses its own family's gold.csv. Own-proxy CSVs are separate evaluator fits, not a common calibrated scale.",
        "line_semantics": "Lines connect retained export coordinates in policy-training order, including x-axis reversals. Square is first; circle is last.",
        "plotted_point_export": {"path": relative(points_path), "rows": len(plotted_points), "sha256": digest(points_path)},
        "diagonal": "Omitted: equal values across separately fitted rating arenas do not establish verdict agreement.",
        "kl_join": "exact trajectory prefix plus unique evaluator rating, absolute tolerance 1e-7; no new coordinate inference",
        "beta_kl_preprocessing": {
            "status": "historically smoothed KL-axis export, not raw measurements",
            "source_revision": "b13b2057c6c87ba754b1271513504bc985e034d8",
            "source_file": "z_elo_calculation.py",
            "source_lines": "381-406; active beta source 901-904",
            "method": "Savitzky-Golay smoothing with window at most 51 and polynomial order 3 before rating/KL export",
            "active_source": "steps_vs_kl_est.csv",
            "negative_coordinates": "retained as exported; not evidence of negative true KL or finite-sample negativity of the trainer's nonnegative k3 expression"},
        "series": JOINS, "beta_kl_audit": kl_audit,
        "inputs": [{"path": relative(p), "sha256": digest(p), "rows": len(rows(p))} for p in sorted(INPUTS)],
        "outputs": [{"path": relative(p), "sha256": digest(p)} for p in outputs],
        "limitations": ["Checkpoints are not independent training repetitions.",
                        "Proxy and Gold axes describe separately fitted arenas, not response-level agreement.",
                        "No uncertainty intervals are reconstructed from aggregate rating CSVs.",
                        "Beta KL-axis coordinates were historically smoothed; they are not raw measurements or reconstructed unsmoothed KL.",
                        "These plots retain local arenas and should not be substituted for global-arena numerical summaries."]}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"figures": len(outputs), "joined_series": len(JOINS),
                      "input_files": len(INPUTS), "beta_kl_rows": sum(p["points"] for p in kl_audit),
                      "manifest": manifest_path.name}, indent=2))


if __name__ == "__main__":
    main()
