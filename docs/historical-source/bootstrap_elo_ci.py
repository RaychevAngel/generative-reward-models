#!/usr/bin/env python3
# Licensed under the Apache License, Version 2.0; see LICENSE.
"""Inspect or reuse the retained match-bootstrap method with an explicit local CSV.

Source-only transport adaptation; the original match inputs are not distributed.
This is not part of the self-contained CPU reproduction commands. NumPy and
SciPy are needed only for an explicitly requested fit, not for --help or input
validation. See results/receipts/README.md for provenance and limitations.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path


PLAYER_RE = re.compile(r"^(ans|think)_([^_]+)_([^_]+)_(\d+)$")
ELO_SCALE = 400.0 / math.log(10.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matches", type=Path, required=True,
                        help="Existing local CSV with exactly winner,loser columns; no download.")
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--maxiter", type=int, default=200)
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="New local output directory; existing paths are refused.")
    args = parser.parse_args()
    if args.bootstrap < 1 or args.maxiter < 1 or args.seed < 0:
        parser.error("bootstrap/maxiter must be positive and seed must be nonnegative")
    if not math.isfinite(args.alpha) or args.alpha < 0:
        parser.error("alpha must be finite and nonnegative")
    return args


def read_matches(path: Path) -> dict[str, list[str]]:
    """Read local comparison rows without filtering, reordering or deduplication."""
    if not path.is_file():
        raise ValueError("--matches must identify an existing local CSV file")
    columns = {"winner": [], "loser": []}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or len(reader.fieldnames) != 2 or set(reader.fieldnames) != set(columns):
            raise ValueError("match CSV must have exactly winner,loser columns")
        for row_number, row in enumerate(reader, start=2):
            if set(row) != set(columns) or any(not isinstance(row[key], str) or not row[key].strip() for key in columns):
                raise ValueError(f"invalid winner/loser values on CSV row {row_number}")
            for key in columns:
                columns[key].append(row[key])
    if not columns["winner"]:
        raise ValueError("match CSV must contain at least one comparison")
    if len(set(columns["winner"]) | set(columns["loser"])) < 2:
        raise ValueError("match CSV must contain at least two players")
    parsed = [parse_player(player) for player in set(columns["winner"]) | set(columns["loser"])]
    modes = defaultdict(set)
    for player in parsed:
        if player is not None:
            mode, policy, judge, _step = player
            modes[(policy, judge)].add(mode)
    if not any(mode_set == {"ans", "think"} for mode_set in modes.values()):
        raise ValueError("match CSV must include an ans/think matched configuration")
    return columns


def parse_player(player: str) -> tuple[str, str, str, int] | None:
    match = PLAYER_RE.match(player)
    if not match:
        return None
    mode, policy, judge, step = match.groups()
    return mode, policy, judge, int(step)


def fit_bt(
    winners: np.ndarray,
    losers: np.ndarray,
    n_players: int,
    anchor_idx: int,
    alpha: float,
    maxiter: int,
    init: np.ndarray | None = None,
) -> np.ndarray:
    free = np.array([i for i in range(n_players) if i != anchor_idx], dtype=np.int64)
    x0 = np.zeros(len(free), dtype=np.float64) if init is None else np.delete(init, anchor_idx)

    def unpack(x: np.ndarray) -> np.ndarray:
        theta = np.zeros(n_players, dtype=np.float64)
        theta[free] = x
        return theta

    def objective(x: np.ndarray) -> tuple[float, np.ndarray]:
        theta = unpack(x)
        diff = theta[losers] - theta[winners]
        probs = expit(diff)
        loss = float(np.logaddexp(0.0, diff).sum() + 0.5 * alpha * np.dot(x, x))
        grad = np.zeros(n_players, dtype=np.float64)
        np.add.at(grad, winners, -probs)
        np.add.at(grad, losers, probs)
        return loss, grad[free] + alpha * x

    result = minimize(
        fun=lambda x: objective(x)[0],
        x0=x0,
        jac=lambda x: objective(x)[1],
        method="L-BFGS-B",
        options={"maxiter": maxiter, "ftol": 1e-7},
    )
    if not result.success:
        print(f"warning: optimizer did not fully converge: {result.message}")
    return unpack(result.x)


def prefix_metrics(elo: np.ndarray, players: list[str]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for idx, player in enumerate(players):
        parsed = parse_player(player)
        if parsed is None:
            continue
        mode, policy, judge, _step = parsed
        grouped[f"{mode}_{policy}_{judge}"].append(float(elo[idx]))
    metrics: dict[str, dict[str, float]] = {}
    for prefix, values in grouped.items():
        ordered = sorted(values, reverse=True)
        metrics[prefix] = {
            "peak": ordered[0],
            "top5": float(np.mean(ordered[:5])),
        }
    return metrics


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.array(values, dtype=np.float64), q))


def main() -> None:
    args = parse_args()
    try:
        ds = read_matches(args.matches)
        if args.out_dir.exists() or args.out_dir.is_symlink():
            raise ValueError("--out-dir must be a new path; existing output is never overwritten")
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from error

    # Keep help, local-input validation and overwrite checks framework-free.
    global np, minimize, expit
    try:
        import numpy as np
        from scipy.optimize import minimize
        from scipy.special import expit
    except ImportError as error:
        raise SystemExit("An explicit fit requires separately installed NumPy and SciPy; no packages were installed.") from error
    args.out_dir.mkdir(parents=True, exist_ok=False)

    players = sorted(set(ds["winner"]) | set(ds["loser"]))
    player_to_idx = {player: idx for idx, player in enumerate(players)}
    winners = np.array([player_to_idx[player] for player in ds["winner"]], dtype=np.int64)
    losers = np.array([player_to_idx[player] for player in ds["loser"]], dtype=np.int64)
    anchor = "ans_4b_4b_0" if "ans_4b_4b_0" in player_to_idx else players[0]
    anchor_idx = player_to_idx[anchor]

    point_theta = fit_bt(winners, losers, len(players), anchor_idx, args.alpha, args.maxiter)
    point_elo = point_theta * ELO_SCALE
    point = prefix_metrics(point_elo, players)

    matched = sorted(
        {
            (policy, judge)
            for mode, policy, judge, _ in filter(None, (parse_player(player) for player in players))
        }
    )
    matched = [
        (policy, judge)
        for policy, judge in matched
        if f"ans_{policy}_{judge}" in point and f"think_{policy}_{judge}" in point
    ]

    rng = np.random.default_rng(args.seed)
    gap_samples: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for _ in range(args.bootstrap):
        sampled = rng.integers(0, len(winners), size=len(winners))
        theta = fit_bt(
            winners[sampled],
            losers[sampled],
            len(players),
            anchor_idx,
            args.alpha,
            args.maxiter,
            init=point_theta,
        )
        metrics = prefix_metrics(theta * ELO_SCALE, players)
        for policy, judge in matched:
            ans = metrics[f"ans_{policy}_{judge}"]
            think = metrics[f"think_{policy}_{judge}"]
            gap_samples[(policy, judge, "peak")].append(ans["peak"] - think["peak"])
            gap_samples[(policy, judge, "top5")].append(ans["top5"] - think["top5"])

    rows: list[dict[str, str | float]] = []
    for policy, judge in matched:
        ans = point[f"ans_{policy}_{judge}"]
        think = point[f"think_{policy}_{judge}"]
        for metric in ["peak", "top5"]:
            gap = ans[metric] - think[metric]
            samples = gap_samples[(policy, judge, metric)]
            rows.append(
                {
                    "policy": policy,
                    "judge": judge,
                    "metric": metric,
                    "answer": ans[metric],
                    "thinking": think[metric],
                    "gap_answer_minus_thinking": gap,
                    "ci_low": percentile(samples, 2.5),
                    "ci_high": percentile(samples, 97.5),
                    "answer_win_bootstrap_fraction": float(np.mean(np.array(samples) > 0.0)),
                }
            )

    csv_path = args.out_dir / "elo_bootstrap_gaps.csv"
    with csv_path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    stable_peak = sum(
        1
        for row in rows
        if row["metric"] == "peak" and float(row["ci_low"]) > 0.0
    )
    stable_top5 = sum(
        1
        for row in rows
        if row["metric"] == "top5" and float(row["ci_low"]) > 0.0
    )
    report = [
        "# Match-Level Elo Bootstrap",
        "",
        f"Input: local CSV `{args.matches.name}`. Preserve its row order and multiplicity.",
        f"Bootstrap replicates: {args.bootstrap}. Anchor player: `{anchor}`.",
        "",
        "This re-fits a Bradley-Terry model on bootstrap resamples of the",
        "stored `winner`/`loser` match table. It estimates uncertainty from the",
        "existing sampled matches, not from fresh policy generations.",
        "",
        f"- Matched Answer-Only/Thinking cells: {len(matched)}",
        f"- Peak gaps with 95% CI fully above zero: {stable_peak}/{len(matched)}",
        f"- Top-5 gaps with 95% CI fully above zero: {stable_top5}/{len(matched)}",
        "",
        "| Policy | Judge | Metric | Gap | 95% CI | Bootstrap P(gap>0) |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for row in rows:
        report.append(
            "| {policy} | {judge} | {metric} | {gap:.1f} | [{low:.1f}, {high:.1f}] | {prob:.2f} |".format(
                policy=row["policy"],
                judge=row["judge"],
                metric=row["metric"],
                gap=float(row["gap_answer_minus_thinking"]),
                low=float(row["ci_low"]),
                high=float(row["ci_high"]),
                prob=float(row["answer_win_bootstrap_fraction"]),
            )
        )
    report_path = args.out_dir / "elo_bootstrap_report.md"
    with report_path.open("x", encoding="utf-8") as handle:
        handle.write("\n".join(report) + "\n")
    print(report_path)
    print(csv_path)


if __name__ == "__main__":
    main()
