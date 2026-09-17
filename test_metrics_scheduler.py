"""CPU-only coverage for reusable metrics and arrival scheduling logic."""

from __future__ import annotations

import pytest

from arrival_scheduler import simulate_arrival_batches
from metrics import (
    account_request_timing,
    padding_ratio,
    percentile,
    summarize,
    summarize_arrival_scheduling,
    summarize_token_timing,
)


def test_padding_ratio_uses_actual_tokens_over_padded_capacity():
    # Three requests are padded to width five; twelve tokens are real.
    assert padding_ratio([3, 5, 4]) == pytest.approx(0.2)
    assert padding_ratio([3, 5], padded_length=8) == pytest.approx(0.5)
    assert padding_ratio([]) == 0.0


def test_percentile_and_summary_are_stable_for_empty_and_nonempty_values():
    assert percentile([], 95) == 0.0
    assert percentile([10.0, 20.0, 30.0, 40.0], 95) == pytest.approx(38.5)
    result = summarize([1.0, 2.0, 3.0])
    assert result["mean"] == pytest.approx(2.0)
    assert result["p50"] == pytest.approx(2.0)
    assert result["p95"] == pytest.approx(2.9)


def test_token_timing_empty_input_has_zero_durations_and_empty_itl():
    result = summarize_token_timing([], start_time_ms=12.5)

    assert result == {
        "ttft_ms": 0.0,
        "itl_values_ms": [],
        "itl_ms": {
            "mean": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "min": 0.0,
            "max": 0.0,
        },
        "decode_ms": 0.0,
        "total_generation_ms": 0.0,
        "output_tokens": 0,
    }


def test_token_timing_one_token_reports_ttft_but_no_decode_or_itl():
    result = summarize_token_timing([17.5], start_time_ms=10.0)

    assert result["ttft_ms"] == pytest.approx(7.5)
    assert result["itl_values_ms"] == []
    assert result["itl_ms"]["mean"] == 0.0
    assert result["decode_ms"] == 0.0
    assert result["total_generation_ms"] == pytest.approx(7.5)
    assert result["output_tokens"] == 1


def test_token_timing_multi_token_reports_deltas_and_duration_boundaries():
    result = summarize_token_timing([100.0, 112.0, 130.0], start_time_ms=95.0)

    assert result["ttft_ms"] == pytest.approx(5.0)
    assert result["itl_values_ms"] == pytest.approx([12.0, 18.0])
    assert result["itl_ms"]["mean"] == pytest.approx(15.0)
    assert result["itl_ms"]["p50"] == pytest.approx(15.0)
    assert result["decode_ms"] == pytest.approx(30.0)
    assert result["total_generation_ms"] == pytest.approx(35.0)
    assert result["output_tokens"] == 3


def test_token_timing_rejects_nonfinite_or_nonmonotonic_timestamps():
    with pytest.raises(ValueError):
        summarize_token_timing([10.0, 9.0])
    with pytest.raises(ValueError):
        summarize_token_timing([float("nan")])
    with pytest.raises(ValueError):
        summarize_token_timing([4.0], start_time_ms=5.0)


def test_request_timing_reports_queue_execution_and_end_to_end_components():
    timing = account_request_timing(
        arrival_time_ms=10.0,
        batch_start_time_ms=15.0,
        batch_end_time_ms=35.0,
    )
    assert timing["queue_wait_ms"] == pytest.approx(5.0)
    assert timing["batch_execution_ms"] == pytest.approx(20.0)
    assert timing["end_to_end_latency_ms"] == pytest.approx(25.0)


def test_arrival_scheduler_is_deterministic_and_respects_batch_limits():
    arrivals = [0.0, 1.0, 2.0, 20.0]
    first = simulate_arrival_batches(
        arrivals,
        max_batch_size=2,
        batch_wait_timeout_ms=5.0,
        execution_ms=10.0,
    )
    second = simulate_arrival_batches(
        arrivals,
        max_batch_size=2,
        batch_wait_timeout_ms=5.0,
        execution_time_ms=10.0,
    )

    assert first == second
    assert len(first) == len(arrivals)
    assert {record["batch_size"] for record in first} <= {1, 2}
    for record in first:
        assert record["queue_wait_ms"] >= 0.0
        assert record["batch_execution_ms"] == pytest.approx(10.0)
        assert record["end_to_end_latency_ms"] == pytest.approx(
            record["queue_wait_ms"] + record["batch_execution_ms"]
        )
        assert record["error"] is None
        assert record["ttft_ms"] is None
        assert record["itl_ms"] is None
        assert record["token_timing_boundary"] == "batch_generate_elapsed_ms_non_streaming"


def test_arrival_scheduler_flushes_a_partial_batch_at_timeout():
    records = simulate_arrival_batches(
        [0.0, 10.0],
        max_batch_size=4,
        batch_wait_timeout_ms=5.0,
        execution_ms=1.0,
    )
    assert [record["batch_size"] for record in records] == [1, 1]


def test_arrival_summary_ignores_failed_requests_for_latency_percentiles():
    records = [
        {
            "arrival_time_ms": 0.0,
            "batch_start_time_ms": 2.0,
            "batch_end_time_ms": 7.0,
            "queue_wait_ms": 2.0,
            "batch_execution_ms": 5.0,
            "end_to_end_latency_ms": 7.0,
            "batch_size": 1,
            "error": None,
        },
        {
            "arrival_time_ms": 1.0,
            "batch_start_time_ms": 2.0,
            "batch_end_time_ms": 7.0,
            "queue_wait_ms": 1.0,
            "batch_execution_ms": 5.0,
            "end_to_end_latency_ms": 6.0,
            "batch_size": 1,
            "error": "simulated failure",
        },
    ]
    summary = summarize_arrival_scheduling(records)
    assert summary["requested"] == 2
    assert summary["completed"] == 1
    assert summary["errors"] == 1
    assert summary["error_rate"] == pytest.approx(0.5)
    assert summary["queue_wait_ms"]["p50"] == pytest.approx(2.0)


def test_arrival_scheduler_marks_provider_failures_without_fabricating_tokens():
    records = simulate_arrival_batches(
        [0.0, 0.0],
        max_batch_size=2,
        batch_wait_timeout_ms=0.0,
        execution_ms={"batch_execution_ms": 4.0, "error": True, "error_message": "CPU test"},
    )
    assert len(records) == 2
    assert all(record["error"] == "CPU test" for record in records)
    assert all(record["error_message"] == "CPU test" for record in records)
    assert all(record["ttft_ms"] is None and record["itl_ms"] is None for record in records)


def test_scheduler_callback_receives_batch_ids_and_preserves_metadata():
    calls = []

    def provider(request_ids, batch_index):
        calls.append((request_ids, batch_index))
        return {"elapsed_ms": 2.5, "worker": "cpu-sim"}

    records = simulate_arrival_batches(
        [0.0, 0.0, 10.0],
        max_batch_size=2,
        batch_wait_timeout_ms=0.0,
        execution_ms=provider,
    )
    assert calls == [((0, 1), 0), ((2,), 1)]
    assert all(record["worker"] == "cpu-sim" for record in records)
    assert all(record["batch_execution_ms"] == pytest.approx(2.5) for record in records)
