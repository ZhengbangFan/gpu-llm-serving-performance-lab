import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from benchmark import percentile, summarize, summarize_result  # noqa: E402


def test_percentile_interpolates_between_values():
    assert percentile([10.0, 20.0, 30.0, 40.0], 95) == pytest.approx(38.5)


def test_summarize_empty_and_nonempty_values():
    assert summarize([])["p50"] == 0.0
    result = summarize([1.0, 2.0, 3.0])
    assert result["mean"] == pytest.approx(2.0)
    assert result["p50"] == pytest.approx(2.0)


def test_summarize_result_reports_throughput_and_distribution():
    result = summarize_result(
        mode="batched",
        requested=4,
        completed=4,
        errors=0,
        elapsed_ms=1000.0,
        request_latencies=[500.0, 500.0, 500.0, 500.0],
        batch_latencies=[500.0, 500.0],
        input_tokens=[8, 8, 8, 8],
        output_tokens=[16, 16, 16, 16],
        batch_sizes=[2, 2, 2, 2],
        peak_memory_mb=1024.0,
    )
    assert result["throughput_requests_per_second"] == pytest.approx(4.0)
    assert result["generated_tokens_per_second"] == pytest.approx(64.0)
    assert result["error_rate"] == 0.0
    assert result["batch_size_distribution"] == {2: 4}


def test_saved_baseline_has_gpu_provenance():
    result_path = Path(__file__).parent / "results" / "qwen25_1.5b_baseline.json"
    if not result_path.exists():
        pytest.skip("baseline has not been generated")
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert payload["metadata"]["gpu"]["cuda_available"] is True
    assert payload["metadata"]["gpu"]["device"] == "NVIDIA GeForce RTX 4080"
    assert all(item["errors"] == 0 for item in payload["results"])
