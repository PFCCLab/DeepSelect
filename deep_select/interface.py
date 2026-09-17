import functools
from typing import Optional, Tuple

from . import deep_select_cuda as _backend

_FRAMEWORK = getattr(_backend, "framework", lambda: "torch")()
if _FRAMEWORK == "paddle":
    import paddle as _framework
else:
    import torch as _framework

Tensor = _framework.Tensor
_ITEMSIZE = {
    _framework.bfloat16: 2,
    _framework.float32: 4,
    _framework.int32: 4,
    _framework.int64: 8,
}


@functools.lru_cache(maxsize=1)
def get_stride_requirement() -> Tuple[int, int]:
    """
    Returns the stride requirement for input / output tensors, in bytes
    """
    return _backend.get_alignment_requirement()


def topk(
    input: Tensor,
    topk: int,
    sorted: bool = False,
    begin: Optional[Tensor] = None,
    end: Optional[Tensor] = None,
    indices_type=None,
    sorted_index: bool = False,
    hint: Optional[Tensor] = None,
    output_idx: Optional[Tensor] = None,
    output_idx_offset: Optional[Tensor] = None,
    idx_oob_fill_value: int = 2147483647,
    value_oob_fill_value: float = float("-inf"),
    return_value: bool = True,
    abort_when_nan_found: bool = True,
) -> Tuple[Optional[Tensor], Tensor]:
    """
    Arguments:
        input: (b, vocab_size), dtype bfloat16/float32. stride(0) must be a multiple of `deep_select.get_stride_requirement()[0]` bytes, and stride(1) must be 1.
        topk: int. Select topk elements for each row.
        sorted: bool. Whether to return sorted **output_val**. Only supports fp32.
        begin(optional): (b,), dtype=int32. CURRENTLY NOT SUPPORTED. The left(inclusive) range for input row, default is 0.
        end(optional): (b,), dtype=int32. The right(exclusive) range for input row, default is vocab_size. The stride of this tensor must be 1.
                       Note when end[i] <= topk, valid elements will be gathered at the beginning of values and indices returned. The rest of `values` will be filled with `value_oob_fill_value`, while the rest of `indices` will be filled with `idx_oob_fill_value` (won't be plused by `output_idx_offset`).
                       `end` <= `vocab_size` must be held
        indices_type: output index dtype, only int32 and int64 are supported.
        sorted_index: bool. Whether to return sorted **output_idx**.
        hint(optional): CURRENTLY NOT SUPPORTED
        output_idx(optional): (b, topk), dtype=indices_type. A contiguous tensor to store output.
        output_idx_offset(optional): (b,), dtype=int32. If provided, all output_idx (`idx_oob_fill_value` not included) will += output_idx_offset.
        idx_oob_fill_value: int. See comments above when end[i]-begin[i]<topk.
        return_value: bool. If False, only return indices without values to accelerate the kernel. The return value is still a Tuple, but the first element will be None.
        abort_when_nan_found: bool. When a NaN is found, if True, aborts the whole kernel; if False, writes 0x3F3F3F3F to the corresponding output_idx[batch_idx][0] and exits.
                The NaN check itself is always enabled. Exception: when the row's length <= topk, it is skipped.

    Return:
        output_val: (b, topk), dtype=input.dtype.
        output_idx: (b, topk), dtype=indices_type.
                    The output tensors may not be contiguous, when topk * sizeof(input.dtype or indices_dtype) is not a multiple of 32 Bytes
    """

    if indices_type is None:
        indices_type = _framework.int64
    if begin is not None:
        raise ValueError("`begin` is not supported now")
    if hint is not None:
        raise ValueError("`hint` is not supported now")
    N = input.shape[0]

    def get_empty_and_aligned_tensor(dim0: int, dim1: int, dtype):
        """Return a tensor backed by storage whose row stride is 32-byte aligned."""
        itemsize = _ITEMSIZE[dtype]
        output_stride_requirement = get_stride_requirement()[1] // itemsize
        dim1_rounded = (dim1 + output_stride_requirement - 1) // output_stride_requirement * output_stride_requirement
        if _FRAMEWORK == "paddle":
            storage = _framework.empty((dim0, dim1_rounded), dtype=dtype)
        else:
            storage = _framework.empty((dim0, dim1_rounded), device=input.device, dtype=dtype)
        return storage[:, :dim1]

    if _FRAMEWORK == "paddle":
        original_device = _framework.device.get_device()
        input_device = f"gpu:{input.place.gpu_device_id()}"
        if original_device != input_device:
            _framework.device.set_device(input_device)
        output_val = get_empty_and_aligned_tensor(N, topk, input.dtype) if return_value else None
        if output_idx is None:
            output_idx = get_empty_and_aligned_tensor(N, topk, indices_type)
        _backend.topk(
            input, topk, begin, end, sorted, sorted_index,
            output_val, output_idx, output_idx_offset, idx_oob_fill_value,
            value_oob_fill_value, return_value, abort_when_nan_found,
        )
    else:
        output_val = get_empty_and_aligned_tensor(N, topk, input.dtype) if return_value else None
        if output_idx is None:
            output_idx = get_empty_and_aligned_tensor(N, topk, indices_type)
        _backend.topk(
            input, topk, begin, end, sorted, sorted_index,
            output_val, output_idx, output_idx_offset, idx_oob_fill_value,
            value_oob_fill_value, return_value, abort_when_nan_found,
        )
    if output_idx.dtype != indices_type:
        raise TypeError(f"output_idx must have dtype {indices_type}")

    return output_val, output_idx
