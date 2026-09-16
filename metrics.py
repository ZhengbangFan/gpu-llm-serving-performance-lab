"""Pure-Python helpers for summarising LLM serving measurements.

The benchmark itself uses PyTorch, but these helpers intentionally depend only
on the standard library so that metric and scheduler tests can run on a CPU
without downloading a model.  Times are represented in milliseconds unless
otherwise stated.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
import math
import statistics
from typing import Any


def _materialize(values: Iterable[float]) -> list[float]:
    """Convert values to finite floats for deterministic metric arithmetic."""

    result = [float(value) for value in values]
    if any(not math.isfinite(value) for value in result):
        raise ValueError("metric values must be finite numbers")
    return result


def percentile(values: Iterable[float], percentile_value: float) -> float:
    """Return an interpolated percentile using the linear-rank method.

    ``percentile_value`` is expressed as a percentage in ``[0, 100]``.  Empty
    input returns ``0.0`` to keep result payloads useful for failed/empty runs.
    """

    try:
        percentile_value = float(percentile_value)
    except (TypeError, ValueError) as exc:
        raise ValueError("percentile must be a finite number in [0, 100]") from exc
    if not math.isfinite(percentile_value) or not 0.0 <= percentile_value <= 100.0:
        raise ValueError("percentile must be a finite number in [0, 100]")

    ordered = sorted(_materialize(values))
    if not ordered:
        return 0.0
    rank = (len(ordered) - 1) * percentile_value / 100.0
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def summarize(values: Iterable[float]) -> dict[str, float]:
    """Return the standard latency/throughput percentile summary.

    The shape intentionally matches the original benchmark helper so existing
    callers and published baseline JSON remain compatible.
    """

    materialized = _materialize(values)
    return {
        "mean": statistics.fmean(materialized) if materialized else 0.0,
        "p50": percentile(materialized, 50),
        "p95": percentile(materialized, 95),
        "p99": percentile(materialized, 99),
        "min": min(materialized) if materialized else 0.0,
        "max": max(materialized) if materialized else 0.0,
    }


def percentile_summary(values: Iterable[float]) -> dict[str, float]:
    """Descriptive alias for :func:`summarize`.

    ``summarize`` is retained as the short compatibility name used by the
    baseline benchmark; this name makes the percentile intent explicit for
    new serving experiments.
    """

    return summarize(values)


def padding_ratio(
    sequence_lengths: Iterable[int],
    padded_length: int | None = None,
) -> float:
    """Return the fraction of padded (rather than real) input tokens.

    ``sequence_lengths`` contains the unpadded token count for each request.
    If ``padded_length`` is omitted, the longest sequence determines the batch
    width, matching tensor batching with ordinary right/left padding.  A batch
    with no requests, or with zero width, has ratio ``0.0``.  Ratios are in
    ``[0.0, 1.0]``.
    """

    lengths: list[int] = []
    for raw_length in sequence_lengths:
        # bool is an int subclass but is almost certainly an input mistake.
        if isinstance(raw_length, bool):
            raise ValueError("sequence lengths must be non-negative integers")
        try:
            length = int(raw_length)
        except (TypeError, ValueError) as exc:
            raise ValueError("sequence lengths must be non-negative integers") from exc
        if length != raw_length or length < 0:
            raise ValueError("sequence lengths must be non-negative integers")
        lengths.append(length)

    if padded_length is None:
        width = max(lengths, default=0)
    else:
        if isinstance(padded_length, bool):
            raise ValueError("padded_length must be a non-negative integer")
        try:
            width = int(padded_length)
        except (TypeError, ValueError) as exc:
            raise ValueError("padded_length must be a non-negative integer") from exc
        if width != padded_length or width < 0:
            raise ValueError("padded_length must be a non-negative integer")

    if not lengths:
        return 0.0

    longest = max(lengths)
    if width < longest:
        raise ValueError("padded_length must be >= every sequence length")

    denominator = len(lengths) * width
    if denominator == 0:
        return 0.0
    ratio = (denominator - sum(lengths)) / denominator
    # Integer arithmetic above is exact; clamp protects callers that pass
    # numeric subclasses with surprising conversion behaviour.
    return max(0.0, min(1.0, float(ratio)))


def account_request_timing(
    arrival_time_ms: float,
    batch_start_time_ms: float,
    batch_end_time_ms: float,
    *,
    batch_size: int = 1,
    error: str | None = None,
) -> dict[str, Any]:
    """Build one request's arrival/scheduling accounting record.

    ``batch_start_time_ms`` is the dispatch boundary and
    ``batch_end_time_ms`` is the completion boundary.  The function validates
    monotonic timestamps rather than silently hiding scheduler bugs.  The
    returned keys are deliberately explicit so records can be serialized as
    JSON and consumed by later analyses.
    """

    arrival = _finite_time(arrival_time_ms, "arrival_time_ms")
    batch_start = _finite_time(batch_start_time_ms, "batch_start_time_ms")
    batch_end = _finite_time(batch_end_time_ms, "batch_end_time_ms")
    if batch_start < arrival:
        raise ValueError("batch_start_time_ms must be >= arrival_time_ms")
    if batch_end < batch_start:
        raise ValueError("batch_end_time_ms must be >= batch_start_time_ms")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")

    return {
        "arrival_time_ms": arrival,
        "queue_wait_ms": batch_start - arrival,
        "batch_execution_ms": batch_end - batch_start,
        "end_to_end_latency_ms": batch_end - arrival,
        "batch_size": batch_size,
        "error": error,
    }


# Short name for callers that treat this helper as the request-level timing
# boundary.  Keep the explicit ``account_request_timing`` spelling as the
# canonical API while offering a discoverable compatibility alias.
request_timing = account_request_timing


def _finite_time(value: float, name: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be a finite number")
    return numeric


def summarize_arrival_scheduling(
    records: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize per-request arrival and scheduler records.

    Records may contain the fields emitted by :func:`account_request_timing`.
    Missing timing values (for example, a request that failed before dispatch)
    are omitted from that timing distribution while still contributing to
    ``requested``/``errors``.  Error records are recognized from a truthy
    ``error`` value or a positive ``errors`` field.
    """

    materialized = list(records)
    requested = len(materialized)
    errors = 0
    queue_waits: list[float] = []
    execution_times: list[float] = []
    latencies: list[float] = []
    batch_sizes: list[int] = []
    arrivals: list[float] = []

    for record in materialized:
        if not isinstance(record, Mapping):
            raise TypeError("records must contain mappings")
        error_value = record.get("error")
        if error_value or _positive_number(record.get("errors", 0)):
            errors += 1

        # Latency distributions describe completed requests.  Failed records
        # still count toward requested/errors but should not skew p50/p95.
        if not error_value and not _positive_number(record.get("errors", 0)):
            _append_finite(record.get("queue_wait_ms"), queue_waits)
            _append_finite(record.get("batch_execution_ms"), execution_times)
            _append_finite(record.get("end_to_end_latency_ms"), latencies)
        _append_finite(record.get("arrival_time_ms"), arrivals)

        raw_batch_size = record.get("batch_size")
        if raw_batch_size is not None:
            if isinstance(raw_batch_size, bool):
                raise ValueError("batch_size values must be positive integers")
            try:
                batch_size = int(raw_batch_size)
            except (TypeError, ValueError) as exc:
                raise ValueError("batch_size values must be positive integers") from exc
            if batch_size != raw_batch_size or batch_size < 1:
                raise ValueError("batch_size values must be positive integers")
            batch_sizes.append(batch_size)

    completed = requested - errors
    inter_arrival: list[float] = []
    if len(arrivals) > 1:
        # Preserve record order: callers generally provide a deterministic
        # arrival stream, and sorting would hide accidental out-of-order data.
        inter_arrival = [later - earlier for earlier, later in zip(arrivals, arrivals[1:])]
        if any(value < 0 for value in inter_arrival):
            raise ValueError("arrival_time_ms values must be non-decreasing")

    return {
        "requested": requested,
        "completed": completed,
        "errors": errors,
        "error_rate": errors / requested if requested else 0.0,
        "queue_wait_ms": summarize(queue_waits),
        "batch_execution_ms": summarize(execution_times),
        "end_to_end_latency_ms": summarize(latencies),
        "inter_arrival_ms": summarize(inter_arrival),
        "batch_size_distribution": dict(Counter(batch_sizes)),
    }


def _append_finite(value: Any, target: list[float]) -> None:
    if value is None:
        return
    target.append(_finite_time(value, "record timing"))


def _positive_number(value: Any) -> bool:
    if value is None or isinstance(value, bool):
        return False
    try:
        return float(value) > 0
    except (TypeError, ValueError):
        return False


# Explicit aliases make the accounting boundary easy to discover without
# forcing callers to depend on one particular naming convention.
arrival_scheduling_metrics = account_request_timing
summarize_arrival_metrics = summarize_arrival_scheduling


__all__ = [
    "account_request_timing",
    "arrival_scheduling_metrics",
    "padding_ratio",
    "percentile",
    "percentile_summary",
    "request_timing",
    "summarize",
    "summarize_arrival_metrics",
    "summarize_arrival_scheduling",
]
