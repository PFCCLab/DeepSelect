from __future__ import annotations

from dataclasses import replace

from common import Case


DISTRIBUTIONS = [
    "unique", "normal", "llm_logits", "all_equal", "low_cardinality_2",
    "low_cardinality_16", "low_cardinality_256", "boundary_tie",
    "ascending", "descending", "sawtooth", "signed_zero", "subnormal",
    "finite_extremes", "infinity", "scan_front", "scan_back", "scan_striped",
]


def expected_dispatch(case: Case, sm_count: int) -> str:
    if case.dtype == "fp32":
        return "fp32-k512" if case.topk <= 512 else "fp32-k1024" if case.topk <= 1024 else "fp32-k4096"
    if case.rows <= 6 and case.width >= 524288 and case.topk <= 1024:
        return "bf16-cluster"
    wave = "one-wave" if case.rows <= sm_count else "multi-wave"
    tier = "k512" if case.topk <= 512 else "k1024" if case.topk <= 1024 else "k4096"
    return f"bf16-{wave}-{tier}"


def smoke_manifest(sm_count: int):
    cases = []
    ks = [1, 2, 31, 32, 511, 512, 513, 1023, 1024, 1025, 2048, 4095, 4096]
    for dtype in ("fp32", "bf16"):
        for index, topk in enumerate(ks):
            width = max(8192, topk + 1)
            dist = DISTRIBUTIONS[index % len(DISTRIBUTIONS)]
            cases.append(Case(f"smoke-{dtype}-k{topk}-{dist}", 7, width, topk, dtype, "int32", dist))
            cases.append(Case(f"smoke-{dtype}-k{topk}-tie", 3, width, topk, dtype, "int64", "boundary_tie", use_end=index % 2 == 0))
    cases += [
        Case("cluster-inside", 6, 524288, 1024, "bf16", distribution="boundary_tie"),
        Case("cluster-batch-outside", 7, 524288, 1024, "bf16", distribution="low_cardinality_16"),
        Case("cluster-width-outside", 6, 524287, 1024, "bf16", distribution="all_equal"),
        Case("cluster-k-outside", 6, 524288, 1025, "bf16", distribution="boundary_tie"),
        Case("one-wave-boundary", sm_count, 32768, 1024, "bf16", distribution="llm_logits"),
        Case("multi-wave-boundary", sm_count + 1, 32768, 1024, "bf16", distribution="low_cardinality_256"),
    ]
    return cases


def standard_manifest(sm_count: int):
    cases = smoke_manifest(sm_count)
    cases += [
        Case("dsa-packed-fp32-random", 256, 262144, 2048, "fp32", distribution="llm_logits", return_value=False),
        Case("dsa-packed-fp32-tie", 256, 262144, 2048, "fp32", distribution="boundary_tie", return_value=False),
        Case("thd-small-fp32", 8192, 8192, 2048, "fp32", distribution="llm_logits", use_end=True, return_value=False),
        Case("thd-medium-fp32", 8192, 32768, 2048, "fp32", distribution="low_cardinality_256", use_end=True, return_value=False),
        Case("bf16-indexer-random", 256, 262144, 1024, "bf16", distribution="llm_logits", return_value=False),
        Case("bf16-indexer-tie", 256, 262144, 1024, "bf16", distribution="boundary_tie", return_value=False),
        Case("bf16-prefill", 4096, 16384, 1024, "bf16", distribution="llm_logits", return_value=False),
        Case("sampling-fp32", 256, 129280, 512, "fp32", "int64", "llm_logits", sorted_value=True),
    ]
    return cases


def stress_manifest(sm_count: int):
    cases = standard_manifest(sm_count)
    for seed in (11, 29, 47):
        for distribution in DISTRIBUTIONS:
            cases.append(Case(f"stress-{distribution}-s{seed}", 32, 65536, 1024, "bf16", "int32", distribution, use_end=seed == 29, return_value=seed != 47, seed=seed))
    cases += [
        Case("thd-long-fp32", 8192, 131072, 2048, "fp32", distribution="llm_logits", use_end=True, return_value=False),
        Case("thd-full-fp32", 8192, 262144, 2048, "fp32", distribution="boundary_tie", use_end=True, return_value=False),
        Case("fp32-width-tail-262143", 64, 262143, 2048, "fp32", distribution="scan_back"),
        Case("fp32-width-tail-262145", 64, 262145, 2048, "fp32", distribution="scan_striped"),
        Case("bf16-cluster-width-plus", 6, 524289, 512, "bf16", distribution="scan_front"),
    ]
    return cases


def manifest(tier: str, sm_count: int):
    return {"smoke": smoke_manifest, "standard": standard_manifest, "stress": stress_manifest}[tier](sm_count)


def coverage(cases, sm_count):
    dispatches = {expected_dispatch(case, sm_count) for case in cases}
    required = {"fp32-k512", "fp32-k1024", "fp32-k4096", "bf16-cluster",
                "bf16-one-wave-k1024", "bf16-multi-wave-k1024"}
    missing = required - dispatches
    if missing:
        raise RuntimeError(f"manifest misses dispatches {sorted(missing)}")
    return {"cases": len(cases), "dispatches": sorted(dispatches), "distributions": sorted({case.distribution for case in cases})}
