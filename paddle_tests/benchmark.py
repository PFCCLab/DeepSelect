from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import statistics

import numpy as np
import paddle

import deep_select
from common import Case, aligned_input, make_end, run_deep_select, verify_case, estimated_bytes


DEFAULT_CASES = [
    Case("bf16-cluster", 6, 524288, 1024, "bf16"),
    Case("bf16-256k", 256, 262144, 1024, "bf16"),
    Case("bf16-large-batch", 4096, 16384, 1024, "bf16"),
    Case("fp32-sampling-small", 6, 129280, 512, "fp32"),
    Case("fp32-sampling", 256, 129280, 512, "fp32"),
    Case("fp32-target-k2048", 256, 262144, 2048, "fp32"),
]


def percentile(samples, value):
    return float(np.percentile(np.asarray(samples), value))


def summarize(samples):
    median = statistics.median(samples)
    absolute = [abs(sample - median) for sample in samples]
    return {
        "median_us": median,
        "mean_us": statistics.mean(samples),
        "stdev_us": statistics.pstdev(samples),
        "cv": statistics.pstdev(samples) / statistics.mean(samples),
        "mad_us": statistics.median(absolute),
        "min_us": min(samples),
        "p10_us": percentile(samples, 10),
        "p90_us": percentile(samples, 90),
        "p95_us": percentile(samples, 95),
        "p99_us": percentile(samples, 99),
        "max_us": max(samples),
        "samples_us": samples,
    }


def timed_call(fn, start, end):
    start.record()
    fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0


def benchmark_case(case, warmups, samples, rounds):
    input_tensor = aligned_input(case)
    end_tensor = make_end(case)
    correctness_case = Case(**{**case.__dict__, "return_value": True, "index_dtype": "int64"})
    _, _, _, values, indices = run_deep_select(correctness_case, input_tensor, end_tensor)
    paddle.device.synchronize()
    verify_case(correctness_case, input_tensor, end_tensor, None, values, indices)

    positions = paddle.arange(case.width, dtype="int32").reshape([1, -1])
    full_end = paddle.full([case.rows], case.width, dtype="int32")
    ragged_end = paddle.to_tensor(
        [case.width - row % min(case.width, 4096) for row in range(case.rows)], dtype="int32"
    )
    backends = {
        "equivalent/deep_select_values_int64": lambda: deep_select.topk(
            input_tensor, case.topk, indices_type=paddle.int64, return_value=True,
            abort_when_nan_found=False,
        ),
        "equivalent/paddle_topk": lambda: paddle.topk(input_tensor, case.topk, sorted=False),
        "production/deep_select_indices_int32": lambda: deep_select.topk(
            input_tensor, case.topk, indices_type=paddle.int32, return_value=False,
            abort_when_nan_found=False,
        ),
        "end/deep_select_native": lambda: deep_select.topk(
            input_tensor, case.topk, end=ragged_end, indices_type=paddle.int32,
            return_value=False, abort_when_nan_found=False,
        ),
        "end/paddle_where_topk": lambda: paddle.topk(
            paddle.where(
                positions < ragged_end.reshape([-1, 1]),
                input_tensor,
                paddle.full_like(input_tensor, float("-inf")),
            ),
            case.topk,
            sorted=False,
        ),
    }
    order = list(backends)
    for _ in range(warmups):
        for name in order:
            backends[name]()
    paddle.device.synchronize()

    all_samples = {name: [] for name in order}
    starts = {name: paddle.device.Event(enable_timing=True) for name in order}
    ends = {name: paddle.device.Event(enable_timing=True) for name in order}
    rng = random.Random(case.seed)
    for round_index in range(rounds):
        for sample_index in range(samples):
            pair_order = order if (round_index + sample_index) % 2 == 0 else list(reversed(order))
            if sample_index % len(order) == 0:
                rng.shuffle(order)
            for name in pair_order:
                all_samples[name].append(timed_call(backends[name], starts[name], ends[name]))

    stats = {name: summarize(values) for name, values in all_samples.items()}
    equivalent_base = stats["equivalent/paddle_topk"]["median_us"]
    production_base = stats["equivalent/paddle_topk"]["median_us"]
    end_base = stats["end/paddle_where_topk"]["median_us"]
    stats["equivalent/deep_select_values_int64"]["speedup"] = equivalent_base / stats["equivalent/deep_select_values_int64"]["median_us"]
    stats["production/deep_select_indices_int32"]["application_visible_speedup"] = production_base / stats["production/deep_select_indices_int32"]["median_us"]
    stats["end/deep_select_native"]["end_emulation_speedup"] = end_base / stats["end/deep_select_native"]["median_us"]
    return {"case": case.__dict__, "stats": stats}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", action="append", help="rows,width,k,dtype")
    parser.add_argument("--warmups", type=int, default=50)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--max-gib", type=float, default=64.0)
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", default="deep_select_paddle_benchmark.json")
    args = parser.parse_args()
    cases = DEFAULT_CASES
    if args.case:
        cases = []
        for index, spec in enumerate(args.case):
            rows, width, topk, dtype = spec.split(",")
            cases.append(Case(f"cli-{index}", int(rows), int(width), int(topk), dtype))
    if args.list_cases or args.dry_run:
        for case in cases:
            print(json.dumps({"case": case.__dict__, "estimated_gib": estimated_bytes(case) / 1024**3}))
        return

    paddle.set_device("gpu:0")
    limit = int(args.max_gib * 1024**3)
    results = []
    for case in cases:
        estimate = estimated_bytes(case)
        if estimate > limit:
            print(json.dumps({"status": "SKIP_MEMORY", "case": case.name, "estimated_gib": estimate / 1024**3}))
            continue
        result = benchmark_case(case, args.warmups, args.samples, args.rounds)
        print(json.dumps(result))
        results.append(result)
    Path(args.output).write_text(json.dumps({"results": results}, indent=2))


if __name__ == "__main__":
    main()
