from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import paddle

from common import Case, aligned_input, output_digest, run_deep_select, verify_case

HERE = Path(__file__).resolve().parent


def repeat_case(case: Case, runs: int):
    input_tensor = aligned_input(case)
    baseline = None
    for run in range(runs):
        _, end, offset, values, indices = run_deep_select(case, input_tensor)
        paddle.device.synchronize()
        if run == 0:
            verify_case(case, input_tensor, end, offset, values, indices)
            baseline = output_digest(values, indices)
        else:
            assert output_digest(values, indices) == baseline
    return baseline


def stream_case(case: Case, stream_count: int, rounds: int):
    input_tensor = aligned_input(case)
    _, end, offset, values, indices = run_deep_select(case, input_tensor)
    paddle.device.synchronize()
    baseline = output_digest(values, indices)
    streams = [paddle.device.Stream() for _ in range(stream_count)]
    for _ in range(rounds):
        outputs = []
        ready = paddle.device.Event()
        ready.record()
        for stream in streams:
            stream.wait_event(ready)
            with paddle.device.stream_guard(stream):
                _, _, _, values, indices = run_deep_select(case, input_tensor, end, offset)
                done = paddle.device.Event()
                done.record(stream)
            outputs.append((done, values, indices))
        for done, values, indices in outputs:
            done.synchronize()
            assert output_digest(values, indices) == baseline
    return baseline


def interleave_cases(cases, rounds):
    inputs = {case.name: aligned_input(case) for case in cases}
    baselines = {}
    for _ in range(rounds):
        for case in cases:
            _, end, offset, values, indices = run_deep_select(case, inputs[case.name])
            paddle.device.synchronize()
            digest = output_digest(values, indices)
            if case.name in baselines:
                assert digest == baselines[case.name]
            else:
                verify_case(case, inputs[case.name], end, offset, values, indices)
                baselines[case.name] = digest
    return baselines


def fresh_process(case: Case, count: int, timeout: float):
    digests = []
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    for _ in range(count):
        result = subprocess.run(
            [sys.executable, str(HERE / "determinism_worker.py"), "--case-json", case.to_json()],
            env=env,
            text=True,
            capture_output=True,
            check=True,
            timeout=timeout,
        )
        payload = json.loads(result.stdout.strip().splitlines()[-1])
        digests.append((payload["input_digest"], payload["output_digest"]))
    assert len(set(digests)) == 1
    return digests[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--small-runs", type=int, default=1000)
    parser.add_argument("--large-runs", type=int, default=100)
    parser.add_argument("--fresh-processes", type=int, default=5)
    parser.add_argument("--streams", type=int, default=4)
    parser.add_argument("--stream-rounds", type=int, default=20)
    parser.add_argument("--process-timeout", type=float, default=300.0)
    args = parser.parse_args()
    paddle.set_device("gpu:0")

    strong_ties = [
        Case("all-equal", 32, 32768, 1024, "fp32", distribution="all_equal"),
        Case("low-cardinality", 32, 32768, 1024, "bf16", distribution="low_cardinality_16"),
        Case("boundary-tie", 32, 32768, 1024, "fp32", distribution="boundary_tie"),
    ]
    results = {case.name: repeat_case(case, args.small_runs) for case in strong_ties}

    large = Case("target-256k-tie", 256, 262144, 1024, "bf16", distribution="boundary_tie")
    results[large.name] = repeat_case(large, args.large_runs)
    results["streams"] = stream_case(strong_ties[-1], args.streams, args.stream_rounds)

    interleaved = [
        Case("bf16-k512", 64, 32768, 512, "bf16", distribution="all_equal", return_value=False),
        Case("bf16-cluster", 6, 524288, 1024, "bf16", distribution="boundary_tie"),
        Case("bf16-k4096", 16, 32768, 4096, "bf16", distribution="low_cardinality_16"),
        Case("fp32-sorted", 64, 129280, 512, "fp32", "int64", "boundary_tie", sorted_value=True),
        Case("fp32-k1025", 64, 32768, 1025, "fp32", distribution="boundary_tie"),
    ]
    results["interleave"] = interleave_cases(interleaved, 20)
    results["fresh_process"] = fresh_process(strong_ties[-1], args.fresh_processes, args.process_timeout)
    print(json.dumps({"status": "PASS", "results": results}, sort_keys=True))


if __name__ == "__main__":
    main()
