from __future__ import annotations

import numpy as np


def cutoff_classification(row: np.ndarray, length: int, k: int):
    count = min(length, k)
    if not count:
        return None, set(), set(), 0
    valid = np.asarray(row[:length], dtype=np.float32)
    theta = np.partition(valid, length - count)[length - count]
    mandatory = set(np.flatnonzero(valid > theta).tolist())
    boundary = set(np.flatnonzero(valid == theta).tolist())
    return theta, mandatory, boundary, count - len(mandatory)


def verify_selection(row, length, k, indices):
    count = min(length, k)
    selected = np.asarray(indices[:count], dtype=np.int64)
    assert selected.size == len(set(selected.tolist()))
    assert np.all((selected >= 0) & (selected < length))
    theta, mandatory, boundary, quota = cutoff_classification(row, length, k)
    chosen = set(selected.tolist())
    assert mandatory <= chosen
    assert chosen - mandatory <= boundary
    assert len(chosen - mandatory) == quota
    return theta, len(mandatory), len(boundary), quota


def paddle_differential(paddle, tensor, length, k, selected_indices):
    count = min(length, k)
    if not count:
        return "valid_prefix_only"
    reference_values, reference_indices = paddle.topk(tensor[:length], count, sorted=True)
    reference_indices = set(reference_indices.cpu().numpy().tolist())
    selected = set(np.asarray(selected_indices[:count], dtype=np.int64).tolist())
    reference_np = reference_values.cast("float32").cpu().numpy()
    theta = reference_np[-1]
    valid_np = tensor[:length].cast("float32").cpu().numpy()
    tie_free = np.count_nonzero(valid_np == theta) == 1
    if tie_free:
        assert selected == reference_indices
        return "exact_index_set"
    selected_values = tensor[np.asarray(list(selected), dtype=np.int64)]
    assert np.array_equal(
        np.sort(selected_values.cast("float32").cpu().numpy()),
        np.sort(reference_np),
    )
    return "selected_value_multiset"
