"""Deterministic virtual-time arrival and batching scheduler.

The scheduler intentionally has no CUDA, Transformers, or wall-clock
dependencies.  It is used by the arrival benchmark and can be exercised in
CPU-only CI with a fixed execution-time provider.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any


TOKEN_TIMING_BOUNDARY = "batch_generate_elapsed_ms_non_streaming"


ExecutionProvider = Callable[[tuple[int, ...], int], Any]


def _validate_arrivals(arrival_times_ms: Sequence[float]) -> list[float]:
    arrivals = [float(value) for value in arrival_times_ms]
    if any(not math.isfinite(value) or value < 0.0 for value in arrivals):
        raise ValueError("arrival times must be finite and non-negative")
    if any(later < earlier for earlier, later in zip(arrivals, arrivals[1:])):
        raise ValueError("arrival times must be sorted in non-decreasing order")
    return arrivals


def _execution_result(
    provider: float | Sequence[float] | ExecutionProvider,
    batch: tuple[int, ...],
    batch_index: int,
) -> tuple[float, bool, str | None, Mapping[str, Any]]:
    """Normalize a fixed duration, duration sequence, or callback result."""

    if callable(provider):
        value = provider(batch, batch_index)
    elif isinstance(provider, Sequence) and not isinstance(provider, (str, bytes)):
        if batch_index >= len(provider):
            raise ValueError("execution duration sequence is shorter than batch count")
        value = provider[batch_index]
    else:
        value = provider

    extras: Mapping[str, Any] = {}
    if isinstance(value, Mapping):
        # Keep timing/error control fields out of per-request metadata; only
        # caller-defined annotations should be copied onto each record.
        reserved = {
            "batch_execution_ms",
            "elapsed_ms",
            "error",
            "error_message",
        }
        extras = {key: item for key, item in value.items() if key not in reserved}
        duration = value.get("batch_execution_ms", value.get("elapsed_ms", 0.0))
        raw_error = value.get("error", False)
        failed = bool(raw_error)
        error_message = value.get("error_message")
        if failed and not error_message and isinstance(raw_error, str):
            error_message = raw_error
    else:
        duration = value
        failed = False
        error_message = None

    try:
        duration_ms = float(duration)
    except (TypeError, ValueError) as exc:
        raise ValueError("execution duration must be numeric") from exc
    if not math.isfinite(duration_ms) or duration_ms < 0.0:
        raise ValueError("execution duration must be finite and non-negative")
    return duration_ms, failed, str(error_message) if error_message else None, extras


def _record(
    request_id: int,
    arrival_time_ms: float,
    batch_index: int,
    batch_start_time_ms: float,
    batch_end_time_ms: float,
    batch_size: int,
    error: bool,
    error_message: str | None,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    execution_ms = max(0.0, batch_end_time_ms - batch_start_time_ms)
    # ``None`` is the success value so JSON records distinguish a clean
    # request from an actual error.  Preserve a truthy marker even when the
    # caller did not provide a human-readable message (for example when an
    # error request id is supplied to the convenience simulator).
    error_value = (error_message or True) if error else None
    record = {
        "request_id": request_id,
        "arrival_time_ms": arrival_time_ms,
        "batch_id": batch_index,
        "batch_start_time_ms": batch_start_time_ms,
        "batch_end_time_ms": batch_end_time_ms,
        "queue_wait_ms": max(0.0, batch_start_time_ms - arrival_time_ms),
        "batch_execution_ms": execution_ms,
        "end_to_end_latency_ms": max(0.0, batch_end_time_ms - arrival_time_ms),
        "batch_size": batch_size,
        "error": error_value,
        "error_message": error_message,
        # model.generate returns only after completion; these are deliberately
        # null rather than fabricated token-stream timings.
        "ttft_ms": None,
        "itl_ms": None,
        "token_timing_boundary": TOKEN_TIMING_BOUNDARY,
    }
    # Preserve measured fields and timing boundaries even if a provider uses a
    # colliding metadata key. Custom annotations remain available in JSON.
    reserved = set(record)
    record.update({key: value for key, value in metadata.items() if key not in reserved})
    return record


def run_virtual_arrival_schedule(
    arrival_times_ms: Sequence[float],
    max_batch_size: int,
    batch_wait_timeout_ms: float,
    execution_provider: float | Sequence[float] | ExecutionProvider = 0.0,
    *,
    error_request_ids: Iterable[int] | None = None,
) -> list[dict[str, Any]]:
    """Run a deterministic single-worker arrival/batching simulation.

    ``arrival_times_ms`` are planned request arrivals on a virtual clock.  A
    batch dispatches as soon as it reaches ``max_batch_size``; otherwise it
    dispatches when the oldest queued request reaches
    ``batch_wait_timeout_ms``.  The worker executes one batch at a time.  The
    execution provider receives ``(request_ids, batch_index)`` and returns a
    duration in milliseconds, or a mapping containing ``batch_execution_ms``
    (or ``elapsed_ms``), ``error``, ``error_message``, and optional metadata.

    This function never sleeps and never reads the system clock, which keeps
    scheduler tests fast and reproducible.
    """

    if isinstance(max_batch_size, bool) or not isinstance(max_batch_size, int) or max_batch_size < 1:
        raise ValueError("max_batch_size must be positive")
    if not math.isfinite(float(batch_wait_timeout_ms)) or batch_wait_timeout_ms < 0.0:
        raise ValueError("batch_wait_timeout_ms must be finite and non-negative")

    arrivals = _validate_arrivals(arrival_times_ms)
    error_ids = {int(value) for value in (error_request_ids or ())}
    pending_index = 0
    queue: list[int] = []
    records: list[dict[str, Any]] = []
    now_ms = 0.0
    batch_index = 0
    epsilon = 1e-9

    def enqueue_arrivals_through(current_ms: float) -> None:
        nonlocal pending_index
        while pending_index < len(arrivals) and arrivals[pending_index] <= current_ms + epsilon:
            queue.append(pending_index)
            pending_index += 1

    while pending_index < len(arrivals) or queue:
        if not queue:
            # An idle worker advances to the next planned arrival.
            now_ms = max(now_ms, arrivals[pending_index])
            enqueue_arrivals_through(now_ms)
        else:
            # Requests can arrive while the previous batch is executing.
            enqueue_arrivals_through(now_ms)

        if not queue:
            continue

        oldest_arrival = arrivals[queue[0]]
        deadline_ms = oldest_arrival + float(batch_wait_timeout_ms)
        if len(queue) < max_batch_size:
            next_arrival = arrivals[pending_index] if pending_index < len(arrivals) else None
            if now_ms + epsilon < deadline_ms and next_arrival is not None and next_arrival <= deadline_ms + epsilon:
                # Wait for another request, up to the oldest request's deadline.
                now_ms = max(now_ms, next_arrival)
                enqueue_arrivals_through(now_ms)
                continue
            if now_ms + epsilon < deadline_ms:
                now_ms = deadline_ms

        batch = tuple(queue[:max_batch_size])
        del queue[: len(batch)]
        duration_ms, batch_failed, error_message, metadata = _execution_result(
            execution_provider, batch, batch_index
        )
        batch_start_ms = now_ms
        batch_end_ms = batch_start_ms + duration_ms
        for request_id in batch:
            request_failed = batch_failed or request_id in error_ids
            records.append(
                _record(
                    request_id=request_id,
                    arrival_time_ms=arrivals[request_id],
                    batch_index=batch_index,
                    batch_start_time_ms=batch_start_ms,
                    batch_end_time_ms=batch_end_ms,
                    batch_size=len(batch),
                    error=request_failed,
                    error_message=error_message if request_failed else None,
                    metadata=metadata,
                )
            )
        now_ms = batch_end_ms
        batch_index += 1

    # Dispatch order is normally request order, but sorting makes the API
    # stable if callers provide equal-time arrivals or inspect records by id.
    records.sort(key=lambda item: int(item["request_id"]))
    return records


def simulate_arrival_batches(
    arrival_times_ms: Sequence[float],
    max_batch_size: int,
    batch_wait_timeout_ms: float,
    execution_ms: float | Sequence[float] = 0.0,
    *,
    execution_time_ms: float | Sequence[float] | None = None,
    error_request_ids: Iterable[int] | None = None,
) -> list[dict[str, Any]]:
    """Convenience wrapper for CPU-only tests and deterministic experiments."""

    if execution_time_ms is not None:
        execution_ms = execution_time_ms
    return run_virtual_arrival_schedule(
        arrival_times_ms,
        max_batch_size,
        batch_wait_timeout_ms,
        execution_provider=execution_ms,
        error_request_ids=error_request_ids,
    )


__all__ = [
    "TOKEN_TIMING_BOUNDARY",
    "run_arrival_schedule",
    "run_virtual_arrival_schedule",
    "simulate_arrival_schedule",
    "simulate_arrival_batches",
]

# Compatibility aliases for callers that prefer the shorter schedule name.
run_arrival_schedule = run_virtual_arrival_schedule
simulate_arrival_schedule = simulate_arrival_batches
