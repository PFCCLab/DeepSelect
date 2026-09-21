from __future__ import annotations

import argparse
import json

import paddle

from common import Case, aligned_input, digest_tensor, output_digest, run_deep_select, verify_case


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-json", required=True)
    args = parser.parse_args()
    case = Case.from_json(args.case_json)
    paddle.seed(case.seed)
    paddle.set_device("gpu:0")
    input_tensor = aligned_input(case)
    _, end, offset, values, indices = run_deep_select(case, input_tensor)
    paddle.device.synchronize()
    verify_case(case, input_tensor, end, offset, values, indices)
    print(json.dumps({
        "case": case.name,
        "input_digest": digest_tensor(input_tensor),
        "output_digest": output_digest(values, indices),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
