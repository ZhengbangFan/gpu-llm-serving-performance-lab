import pytest


def test_aggregate_uses_repeat_medians_and_weighted_padding():
    from run_experiments import aggregate_rows, input_padding
    assert input_padding([2, 4, 8, 8], 2) == pytest.approx(1 - 22 / 24)
    records = []
    for repeat, rate in enumerate([10, 20, 90]):
        records.append(dict(workload="mixed", batch_size=2, new_tokens=16,
            repeat=repeat, padding_ratio=0.25, result=dict(
                generated_tokens_per_second=rate, throughput_requests_per_second=rate / 16,
                ttft_ms={"p50": 5}, batch_itl_ms={"p50": 2},
                prefill_ms={"p50": 4}, decode_ms={"p50": 30},
                total_generation_ms={"p50": 35}, peak_gpu_memory_mb=100,
                completed=8, errors=0)))
    row, = aggregate_rows(records)
    assert row["generated_tokens_per_second_median"] == 20
    assert row["generated_tokens_per_second_min"] == 10
    assert row["generated_tokens_per_second_max"] == 90
    assert row["repeats"] == 3
    assert row["padding_ratio"] == 0.25
    with pytest.raises(ValueError, match="duplicate"):
        aggregate_rows(records + [records[0]])
