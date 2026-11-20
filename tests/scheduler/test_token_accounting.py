"""Offline regression tests for DUET cumulative-token accounting."""

import ast
from pathlib import Path


TRAINER_PATH = (
    Path(__file__).resolve().parents[2]
    / "duet" / "scheduler" / "trainer.py"
)


def load_compute_metrics():
    """Load the real method without importing GPU dependencies."""
    tree = ast.parse(TRAINER_PATH.read_text(encoding="utf-8"))

    trainer = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "DuetPPOTrainerSync"
    )

    method = next(
        node for node in trainer.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_compute_metrics"
    )

    # The upstream verl metrics method requires GPU dependencies.
    # Remove only that call; retain DUET's actual accounting logic.
    method.body = [
        node for node in method.body
        if not (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and node.value.func.attr == "_compute_metrics"
        )
    ]

    module = ast.Module(body=[method], type_ignores=[])
    ast.fix_missing_locations(module)

    namespace = {}
    exec(compile(module, str(TRAINER_PATH), "exec"), namespace)

    return namespace["_compute_metrics"]


class FakeTrainer:
    def __init__(self):
        self.cumulative_total_tokens = 0
        self.cumulative_response_tokens = 0
        self._last_token_accounted_step = -1

        self.response_tokens = 0
        self.fail_generation_metrics = False

    def _save_generation_times(self, batch, metrics, global_steps):
        if self.fail_generation_metrics:
            raise RuntimeError("Simulated token calculation failure")

        metrics["duet/generated_response_tokens"] = self.response_tokens


def run_step(trainer, step, total_tokens, response_tokens):
    trainer.response_tokens = response_tokens

    metrics = {
        "perf/total_num_tokens": total_tokens,
    }

    compute_metrics = load_compute_metrics()

    compute_metrics(
        trainer,
        batch=None,
        metrics=metrics,
        timing_raw={},
        global_steps=step,
        epoch=0,
    )

    return metrics


def test_cumulative_tokens_increase():
    trainer = FakeTrainer()

    first = run_step(trainer, 1, 1000, 800)
    second = run_step(trainer, 2, 1200, 900)
    third = run_step(trainer, 3, 1500, 1100)

    assert first["duet/cumulative_total_tokens"] == 1000
    assert first["duet/cumulative_response_tokens"] == 800

    assert second["duet/cumulative_total_tokens"] == 2200
    assert second["duet/cumulative_response_tokens"] == 1700

    assert third["duet/cumulative_total_tokens"] == 3700
    assert third["duet/cumulative_response_tokens"] == 2800


def test_same_step_is_not_double_counted():
    trainer = FakeTrainer()

    run_step(trainer, 1, 1000, 800)
    repeated = run_step(trainer, 1, 1000, 800)

    assert repeated["duet/cumulative_total_tokens"] == 1000
    assert repeated["duet/cumulative_response_tokens"] == 800

    assert trainer._last_token_accounted_step == 1


def test_missing_response_tokens_do_not_advance_counter():
    trainer = FakeTrainer()

    run_step(trainer, 1, 1000, 800)

    trainer.fail_generation_metrics = True

    run_step(trainer, 2, 1200, 900)

    # A failed measurement must not silently count zero response tokens
    # and mark the step as successfully accounted.
    assert trainer._last_token_accounted_step == 1
    assert trainer.cumulative_response_tokens == 800