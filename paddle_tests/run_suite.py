from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import paddle

import deep_select
from common import aligned_input, estimated_bytes, run_deep_select, verify_case, digest_tensor, output_digest
from correctness import run_contract_checks
from generators import generate_rows, self_test as generator_self_test
from manifests import coverage, expected_dispatch, manifest
from reporting import Reporter

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def environment():
    properties = paddle.device.cuda.get_device_properties(0)
    so_path = Path(deep_select.interface._backend.__file__)
    return {
        "python": sys.executable,
        "paddle": paddle.__version__,
        "cuda": paddle.version.cuda(),
        "gpu": properties.name,
        "compute_capability": [properties.major, properties.minor],
        "sm_count": properties.multi_processor_count,
        "total_memory": properties.total_memory,
        "extension": str(so_path),
        "extension_sha256": file_sha256(so_path),
        "git_head": subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
        "git_status": subprocess.run(["git", "-C", str(ROOT), "status", "--short"], capture_output=True, text=True).stdout.splitlines(),
    }


def run_case(case):
    generated, metadata = generate_rows(case)
    start = time.perf_counter()
    input_tensor = aligned_input(case, generated)
    input_tensor, end, offset, values, indices = run_deep_select(case, input_tensor)
    paddle.device.synchronize()
    verify_case(case, input_tensor, end, offset, values, indices)
    return {
        "duration_s": time.perf_counter() - start,
        "generator": metadata,
        "input_digest": digest_tensor(input_tensor),
        "output_digest": output_digest(values, indices),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", choices=("smoke", "standard", "stress"), default="standard")
    parser.add_argument("--artifact-dir", default=str(HERE / "artifacts"))
    parser.add_argument("--max-gib", type=float, default=64.0)
    parser.add_argument("--continue-on-failure", action="store_true")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    paddle.set_device("gpu:0")
    env = environment()
    cases = manifest(args.tier, env["sm_count"])
    cover = coverage(cases, env["sm_count"])
    if args.list:
        print(json.dumps({"coverage": cover, "cases": [{**asdict(case), "expected_dispatch": expected_dispatch(case, env["sm_count"])} for case in cases]}, indent=2))
        return 0

    run_id = time.strftime("%Y%m%d-%H%M%S") + f"-{os.getpid()}-{args.tier}"
    directory = Path(args.artifact_dir) / run_id
    manifest_records = [{"case": asdict(case), "expected_dispatch": expected_dispatch(case, env["sm_count"]), "required": True} for case in cases]
    reporter = Reporter(directory, manifest_records)
    (directory / "environment.json").write_text(json.dumps({**env, "coverage": cover}, indent=2, sort_keys=True))
    try:
        generator_self_test()
        run_contract_checks()
    except Exception as error:
        reporter.emit(status="ERROR", stage="harness-self-test", error=repr(error), count_as_case=False)
        return reporter.finish(len(cases))

    limit = int(args.max_gib * 1024**3)
    for case in cases:
        estimate = estimated_bytes(case)
        base = {"case": case.name, "expected_dispatch": expected_dispatch(case, env["sm_count"]), "estimated_bytes": estimate}
        if estimate > limit:
            reporter.emit(status="SKIP_RESOURCE", **base)
            continue
        try:
            result = run_case(case)
            reporter.emit(status="PASS", **base, **result)
        except AssertionError as error:
            reporter.emit(status="FAIL", **base, error=repr(error))
            if not args.continue_on_failure:
                break
        except Exception as error:
            reporter.emit(status="ERROR", **base, error=repr(error))
            if not args.continue_on_failure:
                break
    return reporter.finish(len(cases))


if __name__ == "__main__":
    raise SystemExit(main())
