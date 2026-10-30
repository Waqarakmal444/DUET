"""Regenerate DUET Week 10 offline figures from explicit CSV inputs."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

METHODS = {
    "Streak": "DifficultyOnly streak",
    "History": "Frozen history",
    "MoPPS_1.0": "Bayesian diagnostic λ=1.0",
    "MoPPS_0.5": "Bayesian diagnostic λ=0.5",
}

def read_csv(path):
    with Path(path).open(
        encoding="utf-8-sig", newline=""
    ) as f:
        return list(csv.DictReader(f))

def validate(bootstrap_path, budget_path):
    rows = read_csv(bootstrap_path)
    metrics = {r["metric"]: r for r in rows}

    assert len(metrics) == len(rows), "Duplicate metric names"
    assert set(METHODS).issubset(metrics)

    for method in METHODS:
        r = metrics[method]
        value = float(r["point_estimate"])
        low = float(r["ci_lower"])
        high = float(r["ci_upper"])

        assert 0 <= low <= value <= high <= 1, method

    budgets = read_csv(budget_path)
    assert budgets, "Empty budget sweep"

    expected = [
        "budget_pct",
        "streak_dead_caught",
        "history_dead_caught",
        "streak_informative_lost",
        "history_informative_lost",
        "history_minus_streak_dead",
    ]

    assert set(expected).issubset(budgets[0])

    previous_budget = -1

    for row in budgets:
        budget = float(row["budget_pct"])
        streak = float(row["streak_dead_caught"])
        history = float(row["history_dead_caught"])
        streak_lost = float(row["streak_informative_lost"])
        history_lost = float(row["history_informative_lost"])
        difference = float(row["history_minus_streak_dead"])

        assert 0 <= budget <= 100
        assert budget > previous_budget
        previous_budget = budget

        assert min(streak, history, streak_lost, history_lost) >= 0

        assert abs((history - streak) - difference) < 1e-6

        # Each policy must spend the same expected skip budget.
        assert abs(
            (streak + streak_lost)
            - (history + history_lost)
        ) < 1e-6

    positive = [
        float(r["budget_pct"])
        for r in budgets
        if float(r["history_minus_streak_dead"]) > 0
    ]

    crossover = min(positive) if positive else None

    return metrics, budgets, crossover

def generate(bootstrap_path, budget_path, output_dir):
    metrics, budgets, crossover = validate(
        bootstrap_path, budget_path
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 300,
    })

    def save(fig, stem):
        for extension in ("pdf", "png"):
            fig.savefig(
                output_dir / f"{stem}.{extension}",
                bbox_inches="tight",
            )
        plt.close(fig)

    # Figure 1: ROC AUC estimates and confidence intervals.
    names = list(METHODS)
    values = [
        float(metrics[k]["point_estimate"]) for k in names
    ]
    lows = [
        float(metrics[k]["ci_lower"]) for k in names
    ]
    highs = [
        float(metrics[k]["ci_upper"]) for k in names
    ]

    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    ax.errorbar(
        values,
        range(len(names)),
        xerr=[
            [v - lo for v, lo in zip(values, lows)],
            [hi - v for v, hi in zip(values, highs)],
        ],
        fmt="o",
        capsize=4,
    )
    ax.set_yticks(
        range(len(names)),
        [METHODS[k] for k in names],
    )
    ax.invert_yaxis()
    ax.set_xlim(0.60, 0.82)
    ax.set_xlabel("ROC AUC (95% prompt-cluster bootstrap CI)")
    ax.set_title("Offline dead-prompt ranking performance")
    ax.grid(axis="x", alpha=0.2)

    for i, value in enumerate(values):
        ax.annotate(
            f"{value:.4f}",
            (value, i),
            xytext=(8, 6),
            textcoords="offset points",
        )

    save(fig, "w10_roc_auc_comparison")

    x = [float(r["budget_pct"]) for r in budgets]
    streak = [
        float(r["streak_dead_caught"]) for r in budgets
    ]
    history = [
        float(r["history_dead_caught"]) for r in budgets
    ]
    difference = [
        float(r["history_minus_streak_dead"])
        for r in budgets
    ]

    # Figure 2: Matched-budget comparison.
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.plot(x, streak, label="DifficultyOnly streak")
    ax.plot(x, history, label="Frozen history predictor")
    ax.set_xlabel("Expected skip budget (%)")
    ax.set_ylabel("Expected dead prompts identified")
    ax.set_title(
        "Dead-prompt identification at matched skip budgets"
    )
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    save(fig, "w10_matched_skip_budget")

    # Figure 3: History advantage.
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    ax.plot(x, difference)
    ax.axhline(0, linestyle="--", linewidth=1)
    if crossover is not None:
        ax.axvline(
            crossover,
            linestyle=":",
            label=f"First positive grid point: {crossover:g}%",
        )
        ax.legend(frameon=False)

    ax.set_xlabel("Expected skip budget (%)")
    ax.set_ylabel(
        "History minus streak: expected dead prompts"
    )
    ax.set_title("Budget-dependent ranking advantage")
    ax.grid(alpha=0.2)
    save(fig, "w10_predictor_advantage")

    # Machine-readable reproduction of the table.
    with (output_dir / "w10_predictor_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as f:
        writer = csv.writer(f)
        writer.writerow([
            "method", "roc_auc", "ci_lower",
            "ci_upper", "evaluation_type"
        ])
        for name, label in METHODS.items():
            r = metrics[name]
            writer.writerow([
                label,
                float(r["point_estimate"]),
                float(r["ci_lower"]),
                float(r["ci_upper"]),
                "offline_historical_evaluation",
            ])

    summary = {
        "methods": {
            name: {
                key: float(metrics[name][key])
                for key in (
                    "point_estimate",
                    "ci_lower",
                    "ci_upper",
                )
            }
            for name in METHODS
        },
        "crossover_grid_pct": crossover,
        "budget_points": len(budgets),
        "figure_count": 3,
        "evaluation_type": "offline_historical",
    }

    (output_dir / "reproduction_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    return summary

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", type=Path, required=True)
    parser.add_argument("--budget", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    print(json.dumps(generate(
        args.bootstrap, args.budget, args.output
    ), indent=2))

if __name__ == "__main__":
    main()
