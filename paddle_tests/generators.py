from __future__ import annotations

import hashlib

import numpy as np


GENERATOR_VERSION = 2


def _rng(seed: int, name: str, row: int = 0):
    payload = f"{GENERATOR_VERSION}:{seed}:{name}:{row}".encode()
    entropy = np.frombuffer(hashlib.sha256(payload).digest(), dtype=np.uint32)
    return np.random.default_rng(np.random.SeedSequence(entropy))


def fp32_to_bf16_bits(values: np.ndarray) -> np.ndarray:
    bits = np.asarray(values, dtype=np.float32).view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)
    return (rounded >> 16).astype(np.uint16)


def bf16_bits_to_fp32(bits: np.ndarray) -> np.ndarray:
    return (np.asarray(bits, dtype=np.uint32) << 16).view(np.float32)


def quantize(values: np.ndarray, dtype: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return values if dtype == "fp32" else bf16_bits_to_fp32(fp32_to_bf16_bits(values))


def generate_rows(case) -> tuple[np.ndarray, dict]:
    rows = np.empty((case.rows, case.width), dtype=np.float32)
    requested = case.distribution
    for row in range(case.rows):
        rng = _rng(case.seed, requested, row)
        width = case.width
        if requested in {"unique", "unique_permuted"}:
            values = np.arange(width, dtype=np.float32)
            values = (values - width / 2) / max(width, 1)
            rng.shuffle(values)
        elif requested in {"normal", "llm_logits"}:
            scale = 1.0 if requested == "normal" else (0.5 + row % 7 * 0.5)
            values = rng.normal(0.0, scale, width).astype(np.float32)
            if requested == "llm_logits" and width:
                positions = rng.choice(width, min(16, width), replace=False)
                values[positions] += rng.choice([-16.0, 16.0], len(positions)).astype(np.float32)
        elif requested.startswith("low_cardinality_"):
            cardinality = int(requested.rsplit("_", 1)[1])
            values = np.arange(width, dtype=np.int64) % cardinality
            rng.shuffle(values)
            values = values.astype(np.float32)
        elif requested == "all_equal":
            values = np.full(width, 7.0, dtype=np.float32)
        elif requested.startswith("boundary_tie"):
            greater = min(case.topk // 3, width)
            equal = min(max(case.topk * 4, 2), width - greater)
            values = np.zeros(width, dtype=np.float32)
            values[:greater] = 2.0
            values[greater : greater + equal] = 1.0
            rng.shuffle(values)
        elif requested == "ascending":
            values = np.arange(width, dtype=np.float32)
        elif requested == "descending":
            values = np.arange(width, 0, -1, dtype=np.float32)
        elif requested == "sawtooth":
            values = (np.arange(width) % 513).astype(np.float32)
        elif requested == "signed_zero":
            bits = np.where(np.arange(width) % 2, np.uint32(0x80000000), np.uint32(0))
            values = bits.view(np.float32)
            rng.shuffle(values)
        elif requested == "subnormal":
            bits = (np.arange(width, dtype=np.uint32) % np.uint32(0x007FFFFF)) + 1
            bits[1::2] |= np.uint32(0x80000000)
            values = bits.view(np.float32)
            rng.shuffle(values)
        elif requested == "finite_extremes":
            choices = np.array([np.finfo(np.float32).max, -np.finfo(np.float32).max,
                                np.finfo(np.float32).tiny, -np.finfo(np.float32).tiny, 0.0], np.float32)
            values = np.resize(choices, width)
            rng.shuffle(values)
        elif requested == "infinity":
            values = rng.normal(size=width).astype(np.float32)
            values[: min(max(case.topk // 2, 1), width)] = np.inf
            if width:
                values[-1] = -np.inf
            rng.shuffle(values)
        elif requested in {"scan_front", "scan_back", "scan_striped"}:
            values = np.zeros(width, np.float32)
            hot = min(max(case.topk * 2, 2), width)
            highs = np.arange(hot, dtype=np.float32) + 10.0
            if requested == "scan_front":
                values[:hot] = highs
            elif requested == "scan_back":
                values[-hot:] = highs
            else:
                positions = np.linspace(0, max(width - 1, 0), hot, dtype=np.int64)
                values[positions] = highs
        else:
            raise ValueError(f"unknown distribution {requested}")
        rows[row] = quantize(values, case.dtype)

    metadata = validate_generated(case, rows)
    metadata["generator_version"] = GENERATOR_VERSION
    return rows, metadata


def validate_generated(case, rows: np.ndarray) -> dict:
    if rows.shape != (case.rows, case.width):
        raise RuntimeError(f"generator produced {rows.shape}, expected {(case.rows, case.width)}")
    unique_counts = [int(np.unique(row).size) for row in rows]
    if case.distribution.startswith("low_cardinality_"):
        expected = min(int(case.distribution.rsplit("_", 1)[1]), case.width)
        if any(count != expected for count in unique_counts):
            raise RuntimeError(f"cardinality mismatch: expected {expected}, got {unique_counts}")
    if case.distribution == "all_equal" and any(count != min(1, case.width) for count in unique_counts):
        raise RuntimeError("all_equal generator is not equal")
    if case.distribution.startswith("boundary_tie") and case.width > case.topk:
        for row in rows:
            theta = np.partition(row, case.width - case.topk)[case.width - case.topk]
            if np.count_nonzero(row == theta) < 2:
                raise RuntimeError("boundary_tie did not create a cutoff tie")
    if case.distribution == "signed_zero" and case.width >= 2:
        bits = rows.view(np.uint32)
        if not (np.any(bits == 0) and np.any(bits == 0x80000000)):
            raise RuntimeError("signed_zero generator lost a zero sign")
    return {
        "unique_min": min(unique_counts, default=0),
        "unique_max": max(unique_counts, default=0),
        "nan_count": int(np.isnan(rows).sum()),
        "positive_inf_count": int(np.isposinf(rows).sum()),
        "negative_inf_count": int(np.isneginf(rows).sum()),
        "host_sha256": hashlib.sha256(rows.view(np.uint32).tobytes()).hexdigest(),
    }


def self_test():
    from common import Case
    cases = [
        Case("g-unique", 3, 4096, 512, distribution="unique"),
        Case("g-low", 3, 4096, 512, distribution="low_cardinality_16"),
        Case("g-tie", 3, 4096, 512, distribution="boundary_tie"),
        Case("g-zero", 3, 4096, 512, distribution="signed_zero"),
        Case("g-bf16", 3, 4096, 512, dtype="bf16", distribution="llm_logits"),
    ]
    for case in cases:
        first, first_meta = generate_rows(case)
        second, second_meta = generate_rows(case)
        assert np.array_equal(first.view(np.uint32), second.view(np.uint32))
        assert first_meta == second_meta
