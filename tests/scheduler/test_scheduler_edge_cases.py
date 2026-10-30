"""Offline boundary-condition tests for DifficultyOnlyScheduler."""

from tests.scheduler.helpers import (
    add_reward_history,
    make_prompt_pool,
    snapshot_reward_history,
)


def test_empty_pool(difficulty_scheduler):
    pool = make_prompt_pool(0)

    selected = difficulty_scheduler.select_prompts(
        pool, batch_size=50
    )

    assert len(selected) == 0


def test_zero_batch_size(difficulty_scheduler):
    pool = make_prompt_pool(10)

    selected = difficulty_scheduler.select_prompts(
        pool, batch_size=0
    )

    assert len(selected) == 0


def test_oversized_batch(difficulty_scheduler):
    pool = make_prompt_pool(10)

    selected = difficulty_scheduler.select_prompts(
        pool, batch_size=100
    )

    assert len(selected) <= len(pool)


def test_strict_empty_mode(
    difficulty_scheduler, monkeypatch
):
    pool = make_prompt_pool(10)

    add_reward_history(
        difficulty_scheduler,
        num_prompts=10,
        reward=1.0,
        history_length=20,
    )

    before = snapshot_reward_history(difficulty_scheduler)

    monkeypatch.setattr(
        difficulty_scheduler,
        "_skip_easy_prompts",
        lambda prompt, base_prob: True,
    )

    selected = difficulty_scheduler.select_prompts(
        pool,
        batch_size=10,
        allow_empty_batch=True,
    )

    assert len(selected) == 0
    assert snapshot_reward_history(
        difficulty_scheduler
    ) == before


def test_default_fallback(
    difficulty_scheduler, monkeypatch
):
    pool = make_prompt_pool(10)

    add_reward_history(
        difficulty_scheduler,
        num_prompts=10,
        reward=1.0,
        history_length=20,
    )

    monkeypatch.setattr(
        difficulty_scheduler,
        "_skip_easy_prompts",
        lambda prompt, base_prob: True,
    )

    selected = difficulty_scheduler.select_prompts(
        pool,
        batch_size=10,
    )

    assert len(selected) == 10


def test_history_preservation(difficulty_scheduler):
    pool = make_prompt_pool(10)

    add_reward_history(
        difficulty_scheduler,
        num_prompts=10,
        reward=1.0,
        history_length=5,
    )

    before = snapshot_reward_history(
        difficulty_scheduler
    )

    difficulty_scheduler.select_prompts(
        pool, batch_size=10
    )

    assert snapshot_reward_history(
        difficulty_scheduler
    ) == before


def test_selection_statistics(difficulty_scheduler):
    pool = make_prompt_pool(10)

    selected = difficulty_scheduler.select_prompts(
        pool, batch_size=5
    )

    stats = difficulty_scheduler.last_stats

    assert stats.selected >= 0
    assert stats.skipped_easy >= 0
    assert stats.skipped_hard >= 0
    assert stats.considered >= stats.selected
    assert stats.pool_size >= stats.considered
    assert len(selected) <= 5
