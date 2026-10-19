
"""Regression coverage for JointScheduler history forwarding."""

import ast
from pathlib import Path


CORE_PATH = (
    Path(__file__).resolve().parents[2]
    / "duet" / "scheduler" / "core.py"
)


def load_joint_update():
    """Extract the real method without importing GPU dependencies."""
    source = CORE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)

    joint = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "JointScheduler"
    )

    update = next(
        node for node in joint.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "update"
    )

    module = ast.Module(body=[update], type_ignores=[])
    ast.fix_missing_locations(module)

    namespace = {"__builtins__": {}}
    exec(compile(module, str(CORE_PATH), "exec"), namespace)

    return namespace["update"]


class RecordingScheduler:
    def __init__(self):
        self.history = {}

    def update(
        self,
        prompt_id,
        step,
        observed_reward,
        mean_response_length=None,
    ):
        self.history.setdefault(prompt_id, []).append(
            (step, observed_reward, mean_response_length)
        )


def test_joint_scheduler_forwards_identical_observations():
    update = load_joint_update()

    class Harness:
        pass

    instance = Harness()
    instance.difficulty_scheduler = RecordingScheduler()
    instance.length_scheduler = RecordingScheduler()

    observations = [
        ("train_101", 1, 0.2, 400.0),
        ("train_202", 1, 0.8, 2000.0),
        ("train_101", 2, 0.6, 650.0),
        ("train_202", 2, 1.0, 1800.0),
        ("train_101", 3, 0.4, 900.0),
    ]

    for observation in observations:
        update(instance, *observation)

    expected = {
        "train_101": [
            (1, 0.2, 400.0),
            (2, 0.6, 650.0),
            (3, 0.4, 900.0),
        ],
        "train_202": [
            (1, 0.8, 2000.0),
            (2, 1.0, 1800.0),
        ],
    }

    assert instance.difficulty_scheduler.history == expected
    assert instance.length_scheduler.history == expected

    assert (
        instance.difficulty_scheduler.history
        is not instance.length_scheduler.history
    )
