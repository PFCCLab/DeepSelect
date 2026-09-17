from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import paddle

from common import Case, aligned_input, make_end, make_offset, run_deep_select, verify_case, case_summary


DISTRIBUTIONS = ["unique", "all_equal", "low_cardinality_2", "low_cardinality_16", "boundary_tie", "signed_zero", "infinity"]


def smoke_cases():
    cases = []
    for dtype in ("fp32", "bf16"):
        for index_dtype in ("int32", "int64"):
            for topk in (1, 31, 512, 513, 1024, 1025, 2048, 4096):
                width = max(topk + 1, 8192)
                for distribution in DISTRIBUTIONS:
                    cases.append(Case(f"smoke-{dtype}-{index_dtype}-{topk}-{distribution}", 7, width, topk, dtype, index_dtype, distribution))
                cases.append(Case(f"end-{dtype}-{index_dtype}-{topk}", 7, width, topk, dtype, index_dtype, "boundary_tie", True))
                cases.append(Case(f"indices-{dtype}-{index_dtype}-{topk}", 3, width, topk, dtype, index_dtype, "low_cardinality_16", False, False))
                cases.append(Case(f"sorted-index-{dtype}-{index_dtype}-{topk}", 3, width, topk, dtype, index_dtype, "boundary_tie", False, True, False, True))
                if dtype == "fp32":
                    cases.append(Case(f"sorted-value-{index_dtype}-{topk}", 3, width, topk, dtype, index_dtype, "boundary_tie", False, True, True))
    return cases


def full_cases():
    return smoke_cases() + [
        Case("target-fp32-unique", 256, 262144, 1024, "fp32", distribution="unique"),
        Case("target-fp32-tie", 256, 262144, 1024, "fp32", distribution="boundary_tie"),
        Case("target-bf16-unique", 256, 262144, 1024, "bf16", distribution="unique"),
        Case("target-bf16-tie", 256, 262144, 1024, "bf16", distribution="boundary_tie"),
        Case("cluster-bf16", 6, 524288, 1024, "bf16", distribution="boundary_tie"),
        Case("multi-wave-bf16", 149, 262144, 1024, "bf16", distribution="low_cardinality_16"),
        Case("large-batch", 4096, 8192, 1024, "bf16", distribution="boundary_tie"),
        Case("fp32-sampling", 256, 129280, 512, "fp32", "int64", "unique", False, True, True),
    ]


def run_cases(cases, max_bytes):
    passed = skipped = 0
    for index, case in enumerate(cases, 1):
        summary = case_summary(case)
        if summary["estimated_bytes"] > max_bytes:
            print(json.dumps({"status": "SKIP_MEMORY", **summary}))
            skipped += 1
            continue
        input_tensor, end, offset, values, indices = run_deep_select(case)
        paddle.device.synchronize()
        verify_case(case, input_tensor, end, offset, values, indices)
        print(json.dumps({"status": "PASS", "case": case.name, "progress": [index, len(cases)]}))
        passed += 1
    return passed, skipped


def check_output_buffer_and_offset():
    case = Case("output-buffer-offset", 7, 8192, 1024, "fp32", "int32", "boundary_tie", True, True, False, False, True)
    input_tensor = aligned_input(case)
    end = make_end(case)
    offset = make_offset(case)
    physical = (case.topk + 7) // 8 * 8
    storage = paddle.full([case.rows, physical], -777, dtype="int32")
    output = storage[:, : case.topk]
    _, _, _, values, indices = run_deep_select(case, input_tensor, end, offset, output)
    paddle.device.synchronize()
    assert indices.data_ptr() == output.data_ptr()
    verify_case(case, input_tensor, end, offset, values, indices)
    assert paddle.all(storage[:, case.topk :] == -777).item()


def check_nan_semantics():
    case = Case("nan-sentinel", 2, 8192, 1024, "fp32")
    input_tensor = aligned_input(case)
    input_tensor[0, 17] = float("nan")
    _, _, _, _, indices = run_deep_select(case, input_tensor)
    paddle.device.synchronize()
    assert int(indices[0, 0]) == 0x3F3F3F3F
    assert int(indices[1, 0]) != 0x3F3F3F3F

    end_case = Case("nan-outside-end", 2, 8192, 1024, "fp32", use_end=True)
    input_tensor = aligned_input(end_case)
    end = paddle.to_tensor([2048, 4096], dtype="int32")
    input_tensor[0, 4096] = float("nan")
    _, _, _, values, indices = run_deep_select(end_case, input_tensor, end)
    paddle.device.synchronize()
    verify_case(end_case, input_tensor, end, None, values, indices)


def expect_error(fn, text):
    try:
        fn()
    except Exception as error:
        assert text in str(error), (text, str(error))
    else:
        raise AssertionError(f"expected error containing {text!r}")


def check_safe_errors():
    base = paddle.randn([2, 8192], dtype="float32")
    expect_error(lambda: __import__("deep_select").topk(base, 0), "topk must > 0")
    expect_error(lambda: __import__("deep_select").topk(base, 4097), "topk must be <= 4096")
    expect_error(lambda: __import__("deep_select").topk(base.cast("float16"), 32), "FLOAT16")
    expect_error(lambda: __import__("deep_select").topk(base, 32, sorted=True, return_value=False), "return_value")
    expect_error(lambda: __import__("deep_select").topk(base, 32, sorted=True, sorted_index=True), "cannot be used")


def check_stream():
    case = Case("custom-stream", 16, 32768, 1024, "fp32", distribution="boundary_tie")
    input_tensor = aligned_input(case)
    ready = paddle.device.Event()
    ready.record()
    stream = paddle.device.Stream()
    stream.wait_event(ready)
    with paddle.device.stream_guard(stream):
        _, _, _, values, indices = run_deep_select(case, input_tensor)
        done = paddle.device.Event()
        done.record(stream)
    done.synchronize()
    verify_case(case, input_tensor, None, None, values, indices)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--max-gib", type=float, default=32.0)
    parser.add_argument("--list-cases", action="store_true")
    args = parser.parse_args()
    cases = smoke_cases() if args.tier == "smoke" else full_cases()
    if args.list_cases:
        for case in cases:
            print(case.to_json())
        return
    paddle.set_device("gpu:0")
    passed, skipped = run_cases(cases, int(args.max_gib * 1024**3))
    check_output_buffer_and_offset()
    check_nan_semantics()
    check_safe_errors()
    check_stream()
    print(json.dumps({"status": "PASS", "tier": args.tier, "passed": passed, "skipped": skipped}))


if __name__ == "__main__":
    main()
