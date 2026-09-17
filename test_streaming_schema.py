"""CPU-only coverage for the local streaming result schema."""

from __future__ import annotations

import pytest

import benchmark
from benchmark import STREAMING_TIMING_BOUNDARY, summarize_streaming_result


def test_summarize_streaming_result_exposes_batch_and_request_timing_fields():
    batch_results = [
        {
            "prefill_ms": 4.0,
            "ttft_ms": 5.0,
            "batch_itl_values_ms": [2.0, 3.0],
            "decode_ms": 5.0,
            "total_generation_latency_ms": 10.0,
        }
    ]
    request_template = {
        "input_tokens": 7,
        "output_tokens": 3,
        "batch_id": 0,
        "prefill_ms": 4.0,
        "ttft_ms": 5.0,
        "batch_itl_values_ms": [2.0, 3.0],
        "batch_itl_ms": {
            "mean": 2.5,
            "p50": 2.5,
            "p95": 2.95,
            "p99": 2.99,
            "min": 2.0,
            "max": 3.0,
        },
        "decode_ms": 5.0,
        "peak_gpu_memory_mb": 128.0,
        "timing_boundary": STREAMING_TIMING_BOUNDARY,
        "token_timing_boundary": STREAMING_TIMING_BOUNDARY,
        "error": None,
    }
    requests = [
        {**request_template, "request_id": 0, "total_generation_latency_ms": 10.0},
        {**request_template, "request_id": 1, "total_generation_latency_ms": 20.0},
    ]

    result = summarize_streaming_result(
        requested=2,
        completed=2,
        errors=0,
        elapsed_ms=10.0,
        batch_results=batch_results,
        requests=requests,
        batch_sizes=[2],
        peak_memory_mb=128.0,
    )

    assert result["mode"] == "streaming"
    assert result["requested"] == result["completed"] == 2
    assert result["errors"] == 0
    assert result["output_tokens_total"] == 6
    assert result["output_tokens"] == [3, 3]
    assert result["input_tokens_total"] == 14
    assert result["peak_gpu_memory_mb"] == pytest.approx(128.0)
    assert result["timing_boundary"] == STREAMING_TIMING_BOUNDARY
    assert result["token_timing_boundary"] == STREAMING_TIMING_BOUNDARY
    assert result["prefill_ms"]["mean"] == pytest.approx(4.0)
    assert result["ttft_ms"]["mean"] == pytest.approx(5.0)
    assert result["batch_itl_ms"]["mean"] == pytest.approx(2.5)
    assert result["batch_itl_values_ms"] == pytest.approx([2.0, 3.0])
    assert "itl_ms" not in result
    assert "itl_values_ms" not in result
    assert result["decode_ms"]["mean"] == pytest.approx(5.0)
    assert result["total_generation_latency_ms"]["mean"] == pytest.approx(10.0)
    assert result["request_latency_ms"]["p50"] == pytest.approx(15.0)
    assert [batch["batch_id"] for batch in result["batches"]] == [0]
    assert [batch["batch_size"] for batch in result["batches"]] == [2]

    request = result["requests"][0]
    required_request_fields = {
        "request_id",
        "input_tokens",
        "output_tokens",
        "prefill_ms",
        "ttft_ms",
        "batch_id",
        "batch_itl_values_ms",
        "batch_itl_ms",
        "decode_ms",
        "total_generation_latency_ms",
        "peak_gpu_memory_mb",
        "timing_boundary",
        "error",
    }
    assert required_request_fields <= request.keys()


def test_summarize_streaming_result_handles_empty_success_distribution():
    result = summarize_streaming_result(
        requested=0,
        completed=0,
        errors=0,
        elapsed_ms=0.0,
        batch_results=[],
        requests=[],
        batch_sizes=[],
        peak_memory_mb=0.0,
    )

    assert result["mode"] == "streaming"
    assert result["error_rate"] == 0.0
    assert result["output_tokens_total"] == 0
    assert result["prefill_ms"]["mean"] == 0.0
    assert result["batch_itl_ms"]["p50"] == 0.0


def _fake_streaming_batch(batch_size: int) -> dict:
    return {
        "prefill_ms": 4.0,
        "ttft_ms": 5.0,
        "batch_itl_values_ms": [2.0, 3.0],
        "batch_itl_ms": {
            "mean": 2.5,
            "p50": 2.5,
            "p95": 2.95,
            "p99": 2.99,
            "min": 2.0,
            "max": 3.0,
        },
        "decode_ms": 5.0,
        "total_generation_latency_ms": 10.0,
        "total_generation_ms": 10.0,
        "input_tokens": [7] * batch_size,
        "output_tokens": [3] * batch_size,
        "peak_memory_mb": 128.0,
    }


def test_summarize_streaming_result_fallback_builds_one_record_per_batch():
    batch_results = [_fake_streaming_batch(size) for size in (2, 2, 1)]
    requests = []
    for batch_id, batch_size in enumerate((2, 2, 1)):
        for request_offset in range(batch_size):
            requests.append(
                {
                    "request_id": len(requests),
                    "batch_id": batch_id,
                    "input_tokens": 7,
                    "output_tokens": 3,
                    "prefill_ms": 4.0,
                    "ttft_ms": 5.0,
                    "batch_itl_values_ms": [2.0, 3.0],
                    "batch_itl_ms": {
                        "mean": 2.5,
                        "p50": 2.5,
                        "p95": 2.95,
                        "p99": 2.99,
                        "min": 2.0,
                        "max": 3.0,
                    },
                    "decode_ms": 5.0,
                    "total_generation_latency_ms": 10.0,
                    "peak_gpu_memory_mb": 128.0,
                    "timing_boundary": STREAMING_TIMING_BOUNDARY,
                    "token_timing_boundary": STREAMING_TIMING_BOUNDARY,
                    "error": None,
                }
            )

    result = summarize_streaming_result(
        requested=5,
        completed=5,
        errors=0,
        elapsed_ms=10.0,
        batch_results=batch_results,
        requests=requests,
        batch_sizes=[2, 2, 1],
        peak_memory_mb=128.0,
    )

    assert result["batch_size_distribution"] == {2: 2, 1: 1}
    assert [batch["batch_id"] for batch in result["batches"]] == [0, 1, 2]
    assert [batch["batch_size"] for batch in result["batches"]] == [2, 2, 1]
    assert all("batch_itl_values_ms" in batch for batch in result["batches"])
    assert all("batch_itl_ms" in batch for batch in result["batches"])
    assert all("itl_values_ms" not in batch for batch in result["batches"])
    assert all("itl_ms" not in batch for batch in result["batches"])


def test_run_streaming_tracks_multiple_batch_ids_and_per_batch_distribution(monkeypatch):
    monkeypatch.setattr(
        benchmark,
        "generate_streaming_batch",
        lambda tokenizer, model, prompts, max_new_tokens: _fake_streaming_batch(
            len(prompts)
        ),
    )

    result = benchmark.run_streaming(
        tokenizer=None,
        model=None,
        prompts=["p0", "p1", "p2", "p3", "p4"],
        batch_size=2,
        max_new_tokens=3,
    )

    assert result["requested"] == result["completed"] == 5
    assert result["errors"] == 0
    assert result["batch_size_distribution"] == {2: 2, 1: 1}
    assert [batch["batch_size"] for batch in result["batches"]] == [2, 2, 1]
    assert [request["batch_id"] for request in result["requests"]] == [0, 0, 1, 1, 2]
    assert all("itl_ms" not in request for request in result["requests"])
    assert all("batch_itl_ms" in request for request in result["requests"])


def test_run_streaming_preserves_batch_id_on_failed_batch(monkeypatch):
    calls = 0

    def fake_streaming_batch(tokenizer, model, prompts, max_new_tokens):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated streaming failure")
        return _fake_streaming_batch(len(prompts))

    monkeypatch.setattr(benchmark, "generate_streaming_batch", fake_streaming_batch)

    result = benchmark.run_streaming(
        tokenizer=None,
        model=None,
        prompts=["p0", "p1", "p2"],
        batch_size=2,
        max_new_tokens=3,
    )

    assert result["requested"] == 3
    assert result["completed"] == 2
    assert result["errors"] == 1
    failed = result["requests"][2]
    assert failed["request_id"] == 2
    assert failed["batch_id"] == 1
    assert failed["error"] == "RuntimeError: simulated streaming failure"
    assert failed["batch_itl_values_ms"] == []
    assert failed["batch_itl_ms"]["mean"] == 0.0
    assert "itl_values_ms" not in failed
    assert "itl_ms" not in failed
