"""Tests for Week 10 offline reporting invariants."""

import csv
import tempfile
import unittest
from pathlib import Path

from scripts.reproduce_week10 import validate, generate

class TestWeek10Reproducibility(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

        self.bootstrap = self.root / "bootstrap.csv"
        self.budget = self.root / "budget.csv"

        with self.bootstrap.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "metric", "point_estimate",
                "ci_lower", "ci_upper"
            ])
            writer.writerows([
                ["Streak", .6721, .62, .71],
                ["History", .7035, .67, .74],
                ["MoPPS_1.0", .7533, .71, .79],
                ["MoPPS_0.5", .7513, .71, .79],
            ])

        with self.budget.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "budget_pct",
                "streak_dead_caught",
                "history_dead_caught",
                "streak_informative_lost",
                "history_informative_lost",
                "history_minus_streak_dead",
            ])
            writer.writerows([
                [10, 8, 7, 2, 3, -1],
                [20, 14, 15, 6, 5, 1],
            ])

    def test_valid_data(self):
        metrics, budgets, crossover = validate(
            self.bootstrap, self.budget
        )
        self.assertEqual(len(metrics), 4)
        self.assertEqual(len(budgets), 2)
        self.assertEqual(crossover, 20)

    def test_reject_inconsistent_difference(self):
        text = self.budget.read_text()
        self.budget.write_text(text.replace("-1\n", "5\n"))
        with self.assertRaises(AssertionError):
            validate(self.bootstrap, self.budget)

    def test_reject_invalid_confidence_interval(self):
        text = self.bootstrap.read_text()
        self.bootstrap.write_text(
            text.replace("0.6721,0.62,0.71", "0.6721,0.8,0.9")
        )
        with self.assertRaises(AssertionError):
            validate(self.bootstrap, self.budget)

    def test_figures_regenerate(self):
        output = self.root / "figures"
        summary = generate(
            self.bootstrap, self.budget, output
        )
        self.assertEqual(summary["figure_count"], 3)

        for stem in [
            "w10_roc_auc_comparison",
            "w10_matched_skip_budget",
            "w10_predictor_advantage",
        ]:
            for ext in ("pdf", "png"):
                path = output / f"{stem}.{ext}"
                self.assertTrue(path.is_file())
                self.assertGreater(path.stat().st_size, 0)

if __name__ == "__main__":
    unittest.main()
