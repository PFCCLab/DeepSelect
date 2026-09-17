from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json

import numpy as np
import paddle

import deep_select
from generators import generate_rows
from oracle import paddle_differential, verify_selection


DTYPES = {"fp32": paddle.float32, "bf16": paddle.bfloat16}
INDEX_DTYPES = {"int32": paddle.int32, "int64": paddle.int64}


@dataclass(frozen=True)
class Case:
    name: str
    rows: int
    width: int
    topk: int
    dtype: str = "fp32"
    index_dtype: str = "int32"
    distribution: str = "unique"
    use_end: bool = False
    return_value: bool = True
    sorted_value: bool = False
    sorted_index: bool = False
    use_offset: bool = False
    seed: int = 20260918

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @classmethod
    def from_json(cls, value: str) -> "Case":
        return cls(**json.loads(value))


def aligned_input(case: Case, host_rows: np.ndarray | None = None) -> paddle.Tensor:
    dtype = DTYPES[case.dtype]
    itemsize = 4 if dtype == paddle.float32 else 2
    input_alignment = deep_select.get_stride_requirement()[0] // itemsize
    physical = (case.width + input_alignment - 1) // input_alignment * input_alignment
    storage = paddle.full([case.rows, physical], float("nan"), dtype=dtype)
    if host_rows is None:
        host_rows, _ = generate_rows(case)
    storage[:, : case.width] = paddle.to_tensor(host_rows, dtype=dtype)
    return storage[:, : case.width]


def make_end(case: Case) -> paddle.Tensor | None:
    if not case.use_end:
        return None
    candidates = [0, 1, max(case.topk - 1, 0), case.topk, case.topk + 1, max(case.width - 1, 0), case.width]
    values = [min(candidates[row % len(candidates)], case.width) for row in range(case.rows)]
    return paddle.to_tensor(values, dtype="int32")


def make_offset(case: Case) -> paddle.Tensor | None:
    if not case.use_offset:
        return None
    return paddle.to_tensor([(row - case.rows // 2) * 17 for row in range(case.rows)], dtype="int32")


def raw_numpy(tensor: paddle.Tensor) -> np.ndarray:
    array = tensor.cpu().numpy()
    if tensor.dtype == paddle.bfloat16:
        return array.view(np.uint16)
    if tensor.dtype == paddle.float32:
        return array.view(np.uint32)
    return array


def digest_tensor(tensor: paddle.Tensor | None) -> str:
    digest = hashlib.sha256()
    if tensor is None:
        digest.update(b"NONE")
    else:
        digest.update(str(list(tensor.shape)).encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(list(tensor.strides)).encode())
        digest.update(raw_numpy(tensor).tobytes())
    return digest.hexdigest()


def run_deep_select(case: Case, input_tensor=None, end=None, offset=None, output_idx=None):
    input_tensor = aligned_input(case) if input_tensor is None else input_tensor
    end = make_end(case) if end is None else end
    offset = make_offset(case) if offset is None else offset
    values, indices = deep_select.topk(
        input_tensor,
        case.topk,
        end=end,
        indices_type=INDEX_DTYPES[case.index_dtype],
        sorted=case.sorted_value,
        sorted_index=case.sorted_index,
        output_idx=output_idx,
        output_idx_offset=offset,
        idx_oob_fill_value=-1,
        value_oob_fill_value=float("-inf"),
        return_value=case.return_value,
        abort_when_nan_found=False,
    )
    return input_tensor, end, offset, values, indices


def verify_case(case: Case, input_tensor, end, offset, values, indices):
    if case.return_value:
        assert values is not None
    else:
        assert values is None
    assert indices.dtype == INDEX_DTYPES[case.index_dtype]
    assert list(indices.shape) == [case.rows, case.topk]
    lengths = [case.width] * case.rows if end is None else end.cpu().numpy().tolist()
    offsets = [0] * case.rows if offset is None else offset.cpu().numpy().tolist()
    input_np = raw_numpy(input_tensor)
    index_np = indices.cpu().numpy().astype(np.int64)
    value_np = None if values is None else raw_numpy(values)

    for row, length in enumerate(lengths):
        count = min(int(length), case.topk)
        raw_index = index_np[row, :count] - offsets[row]
        assert len(set(raw_index.tolist())) == count
        assert np.all((raw_index >= 0) & (raw_index < length))
        gathered_bits = input_np[row, raw_index]
        if value_np is not None:
            assert np.array_equal(gathered_bits, value_np[row, :count])
        if count < case.topk:
            assert np.all(index_np[row, count:] == -1)
            if value_np is not None:
                if case.dtype == "fp32":
                    assert np.all(value_np[row, count:] == np.array(-np.inf, np.float32).view(np.uint32))
                else:
                    assert np.all(np.isneginf(values[row, count:].cast("float32").cpu().numpy()))
        if not count:
            continue

        theta, mandatory_count, boundary_count, quota = verify_selection(
            input_tensor[row].cast("float32").cpu().numpy(), length, case.topk, raw_index
        )
        comparison_mode = paddle_differential(paddle, input_tensor[row], length, case.topk, raw_index)

        if case.sorted_index and count > 1:
            assert np.all(index_np[row, 1:count] >= index_np[row, : count - 1])
        if case.sorted_value and count > 1:
            numeric = values[row, :count].cast("float32").cpu().numpy()
            assert np.all(numeric[:-1] >= numeric[1:])


def output_digest(values, indices) -> str:
    digest = hashlib.sha256()
    digest.update(digest_tensor(values).encode())
    digest.update(digest_tensor(indices).encode())
    return digest.hexdigest()


def estimated_bytes(case: Case, multiplier: int = 6) -> int:
    itemsize = 4 if case.dtype == "fp32" else 2
    alignment = deep_select.get_stride_requirement()[0] // itemsize
    physical = (case.width + alignment - 1) // alignment * alignment
    return case.rows * physical * itemsize * multiplier + case.rows * case.topk * 12


def case_summary(case: Case) -> dict:
    return {**asdict(case), "estimated_bytes": estimated_bytes(case)}
