# Token timing validation

Model: Qwen/Qwen2.5-1.5B-Instruct. Hardware: NVIDIA GeForce RTX 4080. PyTorch: 2.14.0+cu126. Fixed-length greedy manual forward/cache decode, synchronized after each token. Raw JSON files in this directory retain full run provenance.

| Run | Requests | Batch limit | Tokens/request | Warmup requests | Mean batch TTFT ms | Mean batch ITL ms | Aggregate tokens/s | Errors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| streaming_multibatch_smoke | 3 | 2 | 8 | 0 | 154.18 | 22.84 | 32.43 | 0 |
| streaming_warm_validation | 8 | 4 | 32 | 4 | 21.04 | 20.94 | 190.51 | 0 |

These are single validation runs, not a repeated performance sweep. Warmup=4 means one unmeasured four-request batch, not four repetitions. The zero-warmup smoke includes first-run initialization and must not be directly compared with the warmed run as a speedup experiment. TTFT excludes model loading, tokenization, input transfer, network and queue time. Synchronized wall-clock decode includes host dispatch and synchronization overhead; it is not pure GPU kernel time. Batch ITL is aggregated once per batch timeline, not once per request. No HTTP streaming or continuous batching is claimed.

## Reproduce

```text
.venv/Scripts/python.exe benchmark.py --model Qwen/Qwen2.5-1.5B-Instruct --mode streaming --requests 3 --batch-size 2 --max-new-tokens 8 --warmup 0 --output results/streaming_multibatch_smoke.json
.venv/Scripts/python.exe benchmark.py --model Qwen/Qwen2.5-1.5B-Instruct --mode streaming --requests 8 --batch-size 4 --max-new-tokens 32 --warmup 4 --output results/streaming_warm_validation.json
.venv/Scripts/python.exe -m pytest -q
```

Validation: 21 tests passed (one upstream pynvml deprecation warning). Both GPU runs completed with zero errors. JSON assertions checked request counts, token-interval counts, batch-only aggregation, and TTFT + decode = total generation time. The original baseline remains unchanged in content.
