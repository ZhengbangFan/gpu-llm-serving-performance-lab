"""Bounded offline GPU correctness and repeated manual-decode experiments."""
from __future__ import annotations
import argparse
import csv
import json
import os
import platform
import random
import statistics
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


def input_padding(lengths, batch_size):
    capacity = sum(len(part) * max(part) for i in range(0, len(lengths), batch_size)
                   if (part := lengths[i:i + batch_size]))
    return 1 - sum(lengths) / capacity if capacity else 0.0


def aggregate_rows(records):
    groups = defaultdict(list)
    seen = set()
    for record in records:
        key = (record['workload'], record['batch_size'], record['new_tokens'])
        identity = (*key, record['repeat'])
        if identity in seen:
            raise ValueError('duplicate repeat')
        seen.add(identity)
        groups[key].append(record)
    rows = []
    for (workload, batch_size, tokens), items in sorted(groups.items()):
        row = dict(workload=workload, batch_size=batch_size, new_tokens=tokens,
                   repeats=len(items), padding_ratio=items[0]['padding_ratio'])
        for name in ['generated_tokens_per_second', 'throughput_requests_per_second',
                     'ttft_ms', 'batch_itl_ms', 'prefill_ms', 'decode_ms',
                     'total_generation_ms', 'peak_gpu_memory_mb']:
            values = [r['result'][name] for r in items]
            values = [v['p50'] if isinstance(v, dict) else v for v in values]
            for stat, value in [('median', statistics.median(values)),
                                ('min', min(values)), ('max', max(values)),
                                ('stdev', statistics.stdev(values) if len(values) > 1 else 0)]:
                row[f'{name}_{stat}'] = value
        row['completed'] = sum(r['result']['completed'] for r in items)
        row['errors'] = sum(r['result']['errors'] for r in items)
        rows.append(row)
    return rows


def save_json(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def prompts_for(workload):
    short = 'Explain why a GPU uses high bandwidth memory.'
    lengths = [1] * 8 if workload == 'short' else [1, 4, 12, 24, 1, 4, 12, 24]
    return [short + (' Context: A server batches requests and caches attention keys and values.' * (n - 1))
            for n in lengths]


def encode(tokenizer, prompts):
    texts = [tokenizer.apply_chat_template([{'role': 'user', 'content': p}],
                 tokenize=False, add_generation_prompt=True) for p in prompts]
    return tokenizer(texts, return_tensors='pt', padding=True, truncation=True)


def run(output):
    os.environ['HF_HUB_OFFLINE'] = '1'
    import torch
    import transformers
    from transformers import GenerationConfig
    import benchmark
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    output.mkdir(parents=True, exist_ok=False)
    model_name = 'Qwen/Qwen2.5-1.5B-Instruct'
    tokenizer, model = benchmark.load_model(model_name)
    metadata = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
        model=model_name, model_revision=getattr(model.config, '_commit_hash', None),
        gpu=torch.cuda.get_device_name(), torch=torch.__version__,
        transformers=transformers.__version__, cuda=torch.version.cuda,
        python=platform.python_version(), platform=platform.platform(),
        dtype=str(model.dtype), attention_implementation=model.config._attn_implementation,
        hf_hub_offline=os.environ['HF_HUB_OFFLINE'], padding_side=tokenizer.padding_side,
        git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        requests_per_run=8, repeats=3, warmup='one complete 8-request run per shape',
        capture_in_performance=False, order_seed=17,
        timing_boundary=benchmark.STREAMING_TIMING_BOUNDARY)
    save_json(output / 'metadata.json', metadata)
    correctness = []
    for workload in ['short', 'mixed']:
        for batch_size in [1, 2, 4]:
            prompts = prompts_for(workload)[:batch_size]
            for tokens in [1, 16]:
                encoded = {k: v.to(model.device) for k, v in encode(tokenizer, prompts).items()}
                config = GenerationConfig(max_new_tokens=tokens, do_sample=False,
                    num_beams=1, use_cache=True, eos_token_id=None,
                    pad_token_id=tokenizer.pad_token_id, repetition_penalty=1.0,
                    min_length=0, min_new_tokens=None, forced_bos_token_id=None,
                    forced_eos_token_id=None, suppress_tokens=None, begin_suppress_tokens=None)
                manual = benchmark.generate_streaming_batch(tokenizer, model, prompts, tokens,
                                                            capture_token_ids=True)
                with torch.inference_mode():
                    reference = model.generate(**encoded, generation_config=config)
                reference = reference[:, encoded['input_ids'].shape[1]:].cpu().tolist()
                actual = manual['generated_token_ids']
                record = dict(workload=workload, batch_size=batch_size, new_tokens=tokens,
                    input_tokens=manual['input_tokens'], prompts=prompts,
                    attention_mask=encoded['attention_mask'].cpu().tolist(),
                    manual_token_ids=actual, reference_token_ids=reference,
                    exact_match=actual == reference, generation_config=config.to_dict())
                correctness.append(record)
                save_json(output / 'correctness.json', correctness)
                if actual != reference:
                    raise AssertionError(f'Exact token mismatch: {workload}, batch={batch_size}, tokens={tokens}')
                print(f'correctness PASS {workload} b={batch_size} n={tokens}', flush=True)
    records = []
    shapes = [(w, b, n) for w in ['short', 'mixed'] for b in [1, 2, 4] for n in [16, 64]]
    random.Random(17).shuffle(shapes)
    for workload, batch_size, tokens in shapes:
        prompts = prompts_for(workload)
        lengths = encode(tokenizer, prompts)['attention_mask'].sum(1).tolist()
        warmup = benchmark.run_streaming(tokenizer, model, prompts, batch_size, tokens)
        save_json(output / f'warmup_{workload}_b{batch_size}_n{tokens}.json', warmup)
        if warmup['errors'] or warmup['completed'] != 8:
            raise RuntimeError('Warmup failed')
        for repeat in range(3):
            result = benchmark.run_streaming(tokenizer, model, prompts, batch_size, tokens)
            record = dict(workload=workload, batch_size=batch_size, new_tokens=tokens,
                          repeat=repeat, input_tokens=lengths, prompts=prompts,
                          padding_ratio=input_padding(lengths, batch_size), result=result)
            save_json(output / f'raw_{workload}_b{batch_size}_n{tokens}_r{repeat}.json', record)
            records.append(record)
            save_json(output / 'raw_index.json', records)
            if result['errors'] or result['completed'] != 8:
                raise RuntimeError('Measured run failed')
            print(f'performance {workload} b={batch_size} n={tokens} r={repeat}: '
                  f'{result["generated_tokens_per_second"]:.3f} tok/s', flush=True)
    rows = aggregate_rows(records)
    assert len(records) == 36 and len(rows) == 12 and len(correctness) == 12
    write_report(output, rows, correctness, metadata)
    print(f'COMPLETE: {len(records)} repeats, {sum(r["result"]["completed"] for r in records)} requests', flush=True)


def write_report(output, rows, correctness, metadata):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with (output / 'aggregate.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for metric, ylabel, filename in [
        ('generated_tokens_per_second', 'Generated tokens / second', 'throughput.png'),
        ('ttft_ms', 'Batch TTFT p50 (ms)', 'ttft.png'),
        ('batch_itl_ms', 'Batch ITL p50 (ms)', 'itl.png')]:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
        for ax, workload in zip(axes, ['short', 'mixed']):
            for tokens in [16, 64]:
                selected = [r for r in rows if r['workload'] == workload and r['new_tokens'] == tokens]
                values = [r[f'{metric}_median'] for r in selected]
                lower = [v - r[f'{metric}_min'] for v, r in zip(values, selected)]
                upper = [r[f'{metric}_max'] - v for v, r in zip(values, selected)]
                ax.errorbar([r['batch_size'] for r in selected], values,
                            yerr=[lower, upper], marker='o', capsize=4, label=f'{tokens} output tokens')
            ax.set(title=workload, xlabel='Batch size', xticks=[1, 2, 4])
            ax.grid(alpha=.25)
            ax.legend()
        axes[0].set_ylabel(ylabel)
        fig.suptitle('Median of 3 runs; whiskers show observed min/max')
        fig.tight_layout()
        fig.savefig(output / filename, dpi=160)
        plt.close(fig)
    table = ['| Workload | Batch | Output | Padding | tok/s median [min, max] | TTFT p50 ms | ITL p50 ms | Peak MiB |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        table.append(f'| {r["workload"]} | {r["batch_size"]} | {r["new_tokens"]} | '
            f'{r["padding_ratio"]:.1%} | {r["generated_tokens_per_second_median"]:.2f} '
            f'[{r["generated_tokens_per_second_min"]:.2f}, {r["generated_tokens_per_second_max"]:.2f}] | '
            f'{r["ttft_ms_median"]:.2f} | {r["batch_itl_ms_median"]:.2f} | {r["peak_gpu_memory_mb_median"]:.1f} |')
    comparisons = []
    for workload in ['short', 'mixed']:
        for tokens in [16, 64]:
            points = {r['batch_size']: r for r in rows if r['workload'] == workload and r['new_tokens'] == tokens}
            gain = points[4]['generated_tokens_per_second_median'] / points[1]['generated_tokens_per_second_median']
            comparisons.append(f'- {workload}, {tokens} output tokens: batch 4 / batch 1 throughput = {gain:.2f}x.')
    cv = max(r['generated_tokens_per_second_stdev'] / statistics.mean(
        [r['generated_tokens_per_second_min'], r['generated_tokens_per_second_median'], r['generated_tokens_per_second_max']])
        for r in rows)
    report = f'''# Repeated manual-decode GPU experiment

## Scope and correctness

Model: {metadata['model']} ({metadata['dtype']}); GPU: {metadata['gpu']}.
PyTorch {metadata['torch']}; Transformers {metadata['transformers']}; CUDA {metadata['cuda']}.
All {len(correctness)} exact-token comparisons passed: short and mixed prompts, batch sizes 1/2/4,
output lengths 1/16. Mixed batches 2/4 exercise left padding; a singleton cannot contain padding.
Both full token sequences and attention masks are in `correctness.json`.
The reference uses a fresh GenerationConfig: greedy argmax, no repetition or other logits
penalties, no EOS stopping or minimum-length EOS suppression, cache enabled, identical pad ID.
Thus EOS is selectable and generation continues for exactly the requested number of tokens.
The model's shipped generation defaults are deliberately not inherited. These comparisons
cover this model and these inputs; they do not prove equivalence for all models or prompts.

## Method

One model load, offline Hugging Face access, 12 shapes, 3 measured repeats per shape,
8 requests per repeat (288 measured requests). Each shape gets a complete 8-request
warmup using its own batch size, prompts, and output length. Shape order is shuffled with
seed 17; repeats within a shape are consecutive. All warmups and each measured run are saved.
Short workload repeats one prompt. Mixed workload adds context at repetition factors
1/4/12/24 twice; exact prompts and post-template token lengths are in the raw files.
Token capture is disabled during performance measurement.

Throughput includes tokenization, device transfers, generation, and Python result assembly
in `run_streaming` wall time, excluding model load and warmup. TTFT starts after input transfer
and position-ID preparation, at synchronized prefill start. ITL is the shared batch timeline,
not independent request timing. Every token selection synchronizes CUDA. Padding is
1 - total real input tokens / sum(batch size * batch maximum input length).
Memory is peak allocated CUDA memory, not reserved memory or full device usage.
CSV includes median, min, max, and sample standard deviation across repeats for all metrics;
latency columns aggregate per-run p50 values. Three repeats are not confidence intervals.

## Results

''' + '\n'.join(table) + '\n\n' + '\n'.join(comparisons) + f'''

Maximum throughput sample coefficient of variation across the 12 shapes: {cv:.2%}.
All measured requests completed with zero errors. Larger batches improve aggregate throughput
in the comparisons above, while mixed prompt padding increases input work. This is a bounded
local measurement; run order, GPU clocks, desktop load, and only three repeats limit inference.
It is not a controlled causal estimate of padding overhead or production capacity. No network,
continuous batching, queueing service, or concurrent-client behavior is measured.

## Plots

![Throughput](throughput.png)
![TTFT](ttft.png)
![ITL](itl.png)

## Reproduction

From the repository root (choose a new output directory; existing outputs are refused):

```bash
HF_HUB_OFFLINE=1 .venv/Scripts/python.exe run_experiments.py --output experiments/repeated_gpu_run
```

Install plotting dependency only in the project venv if needed:
`uv pip install --python .venv/Scripts/python.exe matplotlib==3.11.2`.
See `metadata.json`, `aggregate.csv`, `raw_index.json`, individual `raw_*.json`,
and `warmup_*.json`. Existing baseline and validation artifacts were preserved.
'''
    (output / 'report.md').write_text(report, encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.output)
