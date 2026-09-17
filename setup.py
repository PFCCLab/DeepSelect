import os
import subprocess
from datetime import datetime
from pathlib import Path

from setuptools import setup, find_packages

BUILD_FRAMEWORK = os.getenv("DEEP_SELECT_BUILD_FRAMEWORK", "torch").lower()
if BUILD_FRAMEWORK not in {"torch", "paddle"}:
    raise ValueError("DEEP_SELECT_BUILD_FRAMEWORK must be 'torch' or 'paddle'")

exec(open("deep_select/__version__.py").read())

CUDA_SOURCES = [
    "csrc/api.cpp",

    # Generated via `scripts/generate_instantiations.py`

"csrc/cuda_kernels/v3/instantiations/value_bf16_outidx_i32_sv0_si0_rv0_maxtopk_512_numthreads_256_occupancy_2_b_4096_b2_4096_tma_4.cu",
"csrc/cuda_kernels/v3/instantiations/value_bf16_outidx_i32_sv0_si0_rv0_maxtopk_512_numthreads_512_occupancy_1_b_8192_b2_4096_tma_5.cu",
"csrc/cuda_kernels/v3/instantiations/value_bf16_outidx_i32_sv0_si0_rv0_maxtopk_1024_numthreads_256_occupancy_2_b_4096_b2_4096_tma_3.cu",
"csrc/cuda_kernels/v3/instantiations/value_bf16_outidx_i32_sv0_si0_rv0_maxtopk_1024_numthreads_512_occupancy_1_b_8192_b2_4096_tma_5.cu",
"csrc/cuda_kernels/v3/instantiations/value_bf16_outidx_i32_sv0_si0_rv0_maxtopk_4096_numthreads_512_occupancy_1_b_8192_b2_4096_tma_3.cu",
"csrc/cuda_kernels/v3/instantiations/value_bf16_outidx_i32_sv0_si0_rv1_maxtopk_512_numthreads_256_occupancy_2_b_4096_b2_4096_tma_4.cu",
"csrc/cuda_kernels/v3/instantiations/value_bf16_outidx_i32_sv0_si0_rv1_maxtopk_512_numthreads_512_occupancy_1_b_8192_b2_4096_tma_5.cu",
"csrc/cuda_kernels/v3/instantiations/value_bf16_outidx_i32_sv0_si0_rv1_maxtopk_1024_numthreads_256_occupancy_2_b_4096_b2_4096_tma_3.cu",
"csrc/cuda_kernels/v3/instantiations/value_bf16_outidx_i32_sv0_si0_rv1_maxtopk_1024_numthreads_512_occupancy_1_b_8192_b2_4096_tma_5.cu",
"csrc/cuda_kernels/v3/instantiations/value_bf16_outidx_i32_sv0_si0_rv1_maxtopk_4096_numthreads_512_occupancy_1_b_8192_b2_4096_tma_3.cu",
"csrc/cuda_kernels/v3/instantiations/value_bf16_outidx_i32_sv0_si1_rv0_maxtopk_512_numthreads_256_occupancy_2_b_4096_b2_4096_tma_4.cu",
"csrc/cuda_kernels/v3/instantiations/value_bf16_outidx_i32_sv0_si1_rv0_maxtopk_512_numthreads_512_occupancy_1_b_8192_b2_4096_tma_5.cu",
"csrc/cuda_kernels/v3/instantiations/value_bf16_outidx_i32_sv0_si1_rv0_maxtopk_1024_numthreads_256_occupancy_2_b_4096_b2_4096_tma_3.cu",
"csrc/cuda_kernels/v3/instantiations/value_bf16_outidx_i32_sv0_si1_rv0_maxtopk_1024_numthreads_512_occupancy_1_b_8192_b2_4096_tma_5.cu",
"csrc/cuda_kernels/v3/instantiations/value_bf16_outidx_i32_sv0_si1_rv0_maxtopk_4096_numthreads_512_occupancy_1_b_8192_b2_4096_tma_3.cu",
"csrc/cuda_kernels/v3/instantiations/value_bf16_outidx_i32_sv0_si1_rv1_maxtopk_512_numthreads_256_occupancy_2_b_4096_b2_4096_tma_4.cu",
"csrc/cuda_kernels/v3/instantiations/value_bf16_outidx_i32_sv0_si1_rv1_maxtopk_512_numthreads_512_occupancy_1_b_8192_b2_4096_tma_5.cu",
"csrc/cuda_kernels/v3/instantiations/value_bf16_outidx_i32_sv0_si1_rv1_maxtopk_1024_numthreads_256_occupancy_2_b_4096_b2_4096_tma_3.cu",
"csrc/cuda_kernels/v3/instantiations/value_bf16_outidx_i32_sv0_si1_rv1_maxtopk_1024_numthreads_512_occupancy_1_b_8192_b2_4096_tma_5.cu",
"csrc/cuda_kernels/v3/instantiations/value_bf16_outidx_i32_sv0_si1_rv1_maxtopk_4096_numthreads_512_occupancy_1_b_8192_b2_4096_tma_3.cu",
"csrc/cuda_kernels/v3/instantiations/value_bf16_outidx_i64_sv0_si0_rv0_maxtopk_512_numthreads_256_occupancy_2_b_4096_b2_4096_tma_4.cu",
"csrc/cuda_kernels/v3/instantiations/value_bf16_outidx_i64_sv0_si0_rv0_maxtopk_512_numthreads_512_occupancy_1_b_8192_b2_4096_tma_5.cu",
"csrc/cuda_kernels/v3/instantiations/value_bf16_outidx_i64_sv0_si0_rv0_maxtopk_1024_numthreads_256_occupancy_2_b_4096_b2_4096_tma_3.cu",
"csrc/cuda_kernels/v3/instantiations/value_bf16_outidx_i64_sv0_si0_rv0_maxtopk_1024_numthreads_512_occupancy_1_b_8192_b2_4096_tma_5.cu",
"csrc/cuda_kernels/v3/instantiations/value_bf16_outidx_i64_sv0_si0_rv0_maxtopk_4096_numthreads_512_occupancy_1_b_8192_b2_4096_tma_3.cu",
"csrc/cuda_kernels/v3/instantiations/value_bf16_outidx_i64_sv0_si0_rv1_maxtopk_512_numthreads_256_occupancy_2_b_4096_b2_4096_tma_4.cu",
"csrc/cuda_kernels/v3/instantiations/value_bf16_outidx_i64_sv0_si0_rv1_maxtopk_512_numthreads_512_occupancy_1_b_8192_b2_4096_tma_5.cu",
"csrc/cuda_kernels/v3/instantiations/value_bf16_outidx_i64_sv0_si0_rv1_maxtopk_1024_numthreads_256_occupancy_2_b_4096_b2_4096_tma_3.cu",
"csrc/cuda_kernels/v3/instantiations/value_bf16_outidx_i64_sv0_si0_rv1_maxtopk_1024_numthreads_512_occupancy_1_b_8192_b2_4096_tma_5.cu",
"csrc/cuda_kernels/v3/instantiations/value_bf16_outidx_i64_sv0_si0_rv1_maxtopk_4096_numthreads_512_occupancy_1_b_8192_b2_4096_tma_3.cu",
"csrc/cuda_kernels/v3/instantiations/value_bf16_outidx_i64_sv0_si1_rv0_maxtopk_512_numthreads_256_occupancy_2_b_4096_b2_4096_tma_4.cu",
"csrc/cuda_kernels/v3/instantiations/value_bf16_outidx_i64_sv0_si1_rv0_maxtopk_512_numthreads_512_occupancy_1_b_8192_b2_4096_tma_5.cu",
"csrc/cuda_kernels/v3/instantiations/value_bf16_outidx_i64_sv0_si1_rv0_maxtopk_1024_numthreads_256_occupancy_2_b_4096_b2_4096_tma_3.cu",
"csrc/cuda_kernels/v3/instantiations/value_bf16_outidx_i64_sv0_si1_rv0_maxtopk_1024_numthreads_512_occupancy_1_b_8192_b2_4096_tma_5.cu",
"csrc/cuda_kernels/v3/instantiations/value_bf16_outidx_i64_sv0_si1_rv0_maxtopk_4096_numthreads_512_occupancy_1_b_8192_b2_4096_tma_3.cu",
"csrc/cuda_kernels/v3/instantiations/value_bf16_outidx_i64_sv0_si1_rv1_maxtopk_512_numthreads_256_occupancy_2_b_4096_b2_4096_tma_4.cu",
"csrc/cuda_kernels/v3/instantiations/value_bf16_outidx_i64_sv0_si1_rv1_maxtopk_512_numthreads_512_occupancy_1_b_8192_b2_4096_tma_5.cu",
"csrc/cuda_kernels/v3/instantiations/value_bf16_outidx_i64_sv0_si1_rv1_maxtopk_1024_numthreads_256_occupancy_2_b_4096_b2_4096_tma_3.cu",
"csrc/cuda_kernels/v3/instantiations/value_bf16_outidx_i64_sv0_si1_rv1_maxtopk_1024_numthreads_512_occupancy_1_b_8192_b2_4096_tma_5.cu",
"csrc/cuda_kernels/v3/instantiations/value_bf16_outidx_i64_sv0_si1_rv1_maxtopk_4096_numthreads_512_occupancy_1_b_8192_b2_4096_tma_3.cu",

"csrc/cuda_kernels/v3_fp32/instantiations/value_fp32_outidx_i32_sv0_si0_rv0_maxtopk_512_numthreads_512_occupancy_1_b_8192_b2_4096_tma_3.cu",
"csrc/cuda_kernels/v3_fp32/instantiations/value_fp32_outidx_i32_sv0_si0_rv0_maxtopk_1024_numthreads_512_occupancy_1_b_8192_b2_4096_tma_3.cu",
"csrc/cuda_kernels/v3_fp32/instantiations/value_fp32_outidx_i32_sv0_si0_rv0_maxtopk_4096_numthreads_256_occupancy_1_b_4096_b2_4096_tma_3.cu",
"csrc/cuda_kernels/v3_fp32/instantiations/value_fp32_outidx_i32_sv0_si0_rv1_maxtopk_512_numthreads_512_occupancy_1_b_8192_b2_4096_tma_3.cu",
"csrc/cuda_kernels/v3_fp32/instantiations/value_fp32_outidx_i32_sv0_si0_rv1_maxtopk_1024_numthreads_512_occupancy_1_b_8192_b2_4096_tma_3.cu",
"csrc/cuda_kernels/v3_fp32/instantiations/value_fp32_outidx_i32_sv0_si0_rv1_maxtopk_4096_numthreads_256_occupancy_1_b_4096_b2_4096_tma_3.cu",
"csrc/cuda_kernels/v3_fp32/instantiations/value_fp32_outidx_i32_sv0_si1_rv0_maxtopk_512_numthreads_512_occupancy_1_b_8192_b2_4096_tma_3.cu",
"csrc/cuda_kernels/v3_fp32/instantiations/value_fp32_outidx_i32_sv0_si1_rv0_maxtopk_1024_numthreads_512_occupancy_1_b_8192_b2_4096_tma_3.cu",
"csrc/cuda_kernels/v3_fp32/instantiations/value_fp32_outidx_i32_sv0_si1_rv0_maxtopk_4096_numthreads_256_occupancy_1_b_4096_b2_4096_tma_3.cu",
"csrc/cuda_kernels/v3_fp32/instantiations/value_fp32_outidx_i32_sv0_si1_rv1_maxtopk_512_numthreads_512_occupancy_1_b_8192_b2_4096_tma_3.cu",
"csrc/cuda_kernels/v3_fp32/instantiations/value_fp32_outidx_i32_sv0_si1_rv1_maxtopk_1024_numthreads_512_occupancy_1_b_8192_b2_4096_tma_3.cu",
"csrc/cuda_kernels/v3_fp32/instantiations/value_fp32_outidx_i32_sv0_si1_rv1_maxtopk_4096_numthreads_256_occupancy_1_b_4096_b2_4096_tma_3.cu",
"csrc/cuda_kernels/v3_fp32/instantiations/value_fp32_outidx_i32_sv1_si0_rv1_maxtopk_512_numthreads_512_occupancy_1_b_8192_b2_4096_tma_3.cu",
"csrc/cuda_kernels/v3_fp32/instantiations/value_fp32_outidx_i32_sv1_si0_rv1_maxtopk_1024_numthreads_512_occupancy_1_b_8192_b2_4096_tma_3.cu",
"csrc/cuda_kernels/v3_fp32/instantiations/value_fp32_outidx_i32_sv1_si0_rv1_maxtopk_4096_numthreads_256_occupancy_1_b_4096_b2_4096_tma_3.cu",
"csrc/cuda_kernels/v3_fp32/instantiations/value_fp32_outidx_i64_sv0_si0_rv0_maxtopk_512_numthreads_512_occupancy_1_b_8192_b2_4096_tma_3.cu",
"csrc/cuda_kernels/v3_fp32/instantiations/value_fp32_outidx_i64_sv0_si0_rv0_maxtopk_1024_numthreads_512_occupancy_1_b_8192_b2_4096_tma_3.cu",
"csrc/cuda_kernels/v3_fp32/instantiations/value_fp32_outidx_i64_sv0_si0_rv0_maxtopk_4096_numthreads_256_occupancy_1_b_4096_b2_4096_tma_3.cu",
"csrc/cuda_kernels/v3_fp32/instantiations/value_fp32_outidx_i64_sv0_si0_rv1_maxtopk_512_numthreads_512_occupancy_1_b_8192_b2_4096_tma_3.cu",
"csrc/cuda_kernels/v3_fp32/instantiations/value_fp32_outidx_i64_sv0_si0_rv1_maxtopk_1024_numthreads_512_occupancy_1_b_8192_b2_4096_tma_3.cu",
"csrc/cuda_kernels/v3_fp32/instantiations/value_fp32_outidx_i64_sv0_si0_rv1_maxtopk_4096_numthreads_256_occupancy_1_b_4096_b2_4096_tma_3.cu",
"csrc/cuda_kernels/v3_fp32/instantiations/value_fp32_outidx_i64_sv0_si1_rv0_maxtopk_512_numthreads_512_occupancy_1_b_8192_b2_4096_tma_3.cu",
"csrc/cuda_kernels/v3_fp32/instantiations/value_fp32_outidx_i64_sv0_si1_rv0_maxtopk_1024_numthreads_512_occupancy_1_b_8192_b2_4096_tma_3.cu",
"csrc/cuda_kernels/v3_fp32/instantiations/value_fp32_outidx_i64_sv0_si1_rv0_maxtopk_4096_numthreads_256_occupancy_1_b_4096_b2_4096_tma_3.cu",
"csrc/cuda_kernels/v3_fp32/instantiations/value_fp32_outidx_i64_sv0_si1_rv1_maxtopk_512_numthreads_512_occupancy_1_b_8192_b2_4096_tma_3.cu",
"csrc/cuda_kernels/v3_fp32/instantiations/value_fp32_outidx_i64_sv0_si1_rv1_maxtopk_1024_numthreads_512_occupancy_1_b_8192_b2_4096_tma_3.cu",
"csrc/cuda_kernels/v3_fp32/instantiations/value_fp32_outidx_i64_sv0_si1_rv1_maxtopk_4096_numthreads_256_occupancy_1_b_4096_b2_4096_tma_3.cu",
"csrc/cuda_kernels/v3_fp32/instantiations/value_fp32_outidx_i64_sv1_si0_rv1_maxtopk_512_numthreads_512_occupancy_1_b_8192_b2_4096_tma_3.cu",
"csrc/cuda_kernels/v3_fp32/instantiations/value_fp32_outidx_i64_sv1_si0_rv1_maxtopk_1024_numthreads_512_occupancy_1_b_8192_b2_4096_tma_3.cu",
"csrc/cuda_kernels/v3_fp32/instantiations/value_fp32_outidx_i64_sv1_si0_rv1_maxtopk_4096_numthreads_256_occupancy_1_b_4096_b2_4096_tma_3.cu",

"csrc/cuda_kernels/v3_cluster/instantiations/value_bf16_outidx_i32_sv0_si0_rv0_maxtopk_1024_numthreads_256_occupancy_1_b_4096_b2_4096_tma_16_cluster_16.cu",
"csrc/cuda_kernels/v3_cluster/instantiations/value_bf16_outidx_i32_sv0_si0_rv1_maxtopk_1024_numthreads_256_occupancy_1_b_4096_b2_4096_tma_16_cluster_16.cu",
"csrc/cuda_kernels/v3_cluster/instantiations/value_bf16_outidx_i32_sv0_si1_rv0_maxtopk_1024_numthreads_256_occupancy_1_b_4096_b2_4096_tma_16_cluster_16.cu",
"csrc/cuda_kernels/v3_cluster/instantiations/value_bf16_outidx_i32_sv0_si1_rv1_maxtopk_1024_numthreads_256_occupancy_1_b_4096_b2_4096_tma_16_cluster_16.cu",
"csrc/cuda_kernels/v3_cluster/instantiations/value_bf16_outidx_i64_sv0_si0_rv0_maxtopk_1024_numthreads_256_occupancy_1_b_4096_b2_4096_tma_16_cluster_16.cu",
"csrc/cuda_kernels/v3_cluster/instantiations/value_bf16_outidx_i64_sv0_si0_rv1_maxtopk_1024_numthreads_256_occupancy_1_b_4096_b2_4096_tma_16_cluster_16.cu",
"csrc/cuda_kernels/v3_cluster/instantiations/value_bf16_outidx_i64_sv0_si1_rv0_maxtopk_1024_numthreads_256_occupancy_1_b_4096_b2_4096_tma_16_cluster_16.cu",
"csrc/cuda_kernels/v3_cluster/instantiations/value_bf16_outidx_i64_sv0_si1_rv1_maxtopk_1024_numthreads_256_occupancy_1_b_4096_b2_4096_tma_16_cluster_16.cu",

]

def build_on_cuda_platform():
    if BUILD_FRAMEWORK == "paddle":
        from paddle.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME
    else:
        from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME

    assert CUDA_HOME is not None, f"{BUILD_FRAMEWORK} must be compiled with CUDA support"

    def append_nvcc_threads(nvcc_extra_args):
        nvcc_threads = os.getenv("NVCC_THREADS") or "16"
        return nvcc_extra_args + ["--threads", nvcc_threads]

    nvcc_version = subprocess.check_output(
        [os.path.join(CUDA_HOME, "bin", "nvcc"), "--version"],
        stderr=subprocess.STDOUT,
    ).decode("utf-8")
    nvcc_version_number = nvcc_version.split("release ")[1].split(",")[0].strip()
    major, minor = map(int, nvcc_version_number.split("."))
    print(f"Compiling using NVCC {major}.{minor}")
    if major < 12 or (major == 12 and minor <= 8):
        raise RuntimeError("sm100 compilation requires NVCC 12.9 or higher.")

    cc_flag = [
        # Currently skip build for sm80 and sm90 to speed up compilation
        # "-gencode", "arch=compute_80,code=sm_80",
        # "-gencode", "arch=compute_90a,code=sm_90a",

        # Compile sm100 and sm103 separately to give the compiler more room for optimization
        "-gencode", "arch=compute_100a,code=sm_100a",
        "-gencode", "arch=compute_103a,code=sm_103a",
    ]

    this_dir = os.path.dirname(os.path.abspath(__file__))

    cxx_args = ["-O3", "-std=c++20", "-DNDEBUG", "-Wno-deprecated-declarations", "-DKERUTILS_IS_BUILD_ON_CUDA"]
    if BUILD_FRAMEWORK == "paddle":
        cxx_args.extend(["-DPADDLE_WITH_CUDA", "-DDEEP_SELECT_USE_PADDLE"])

    ext_modules = [CUDAExtension(
        name="deep_select.deep_select_cuda",
        sources=CUDA_SOURCES,
        extra_compile_args={
            "cxx": cxx_args,
            "nvcc": append_nvcc_threads([
                "-O3",
                "-std=c++20",
                "-Wno-deprecated-declarations",
                "-U__CUDA_NO_HALF_OPERATORS__",
                "-U__CUDA_NO_HALF_CONVERSIONS__",
                "-U__CUDA_NO_HALF2_OPERATORS__",
                "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                "--expt-relaxed-constexpr",
                "--expt-extended-lambda",
                "--use_fast_math",
                "--ftz=false",  # Don't flush subnormals to zero, since we're going to use float addition to simulate integer addition (in order to be faster)
                "--ptxas-options=-v,--register-usage-level=10,--warn-on-spills,--warn-on-double-precision-use",
                "-lineinfo",
                "--source-in-ptx",
            ] + cc_flag)
        },
        include_dirs=[
            Path(this_dir) / "csrc",
            Path(this_dir) / "csrc" / "3rdparty" / "cutlass" / "include",
            Path(this_dir) / "csrc" / "3rdparty" / "kerutils" / "include",
            Path(CUDA_HOME) / "targets" / "x86_64-linux" / "include" / "cccl",
            Path(CUDA_HOME) / "targets" / "sbsa-linux" / "include" / "cccl",
        ],
        extra_link_args=[
            f'-L{Path(CUDA_HOME) / "targets" / "x86_64-linux" / "lib" / "stubs"}',
            f'-L{Path(CUDA_HOME) / "targets" / "sbsa-linux" / "lib" / "stubs"}',
            "-lcuda",
        ],
    )]

    class SpillCheckBuildExtension(BuildExtension):
        STACK_BASELINE = 8  # Because we're using `printf`

        def run(self):
            super().run()
            if BUILD_FRAMEWORK == "paddle" and self.inplace:
                return  # Paddle renames the artifact after build_ext; ptxas already enforces --warn-on-spills.
            for ext in self.extensions:
                so_path = self.get_ext_fullpath(ext.name)
                if BUILD_FRAMEWORK == "paddle":
                    cuda_home = CUDA_HOME if CUDA_HOME is not None else "/usr/local/cuda"
                    result = subprocess.run(
                        [os.path.join(cuda_home, "bin/cuobjdump"), "-res-usage", so_path],
                        capture_output=True,
                        text=True,
                        check=True,
                    )
                    spills = []
                    for line in result.stdout.splitlines():
                        if "LOCAL:" in line or "STACK:" in line:
                            fields = dict(part.split(":", 1) for part in line.split() if ":" in part)
                            if int(fields.get("LOCAL", 0)) > 0 or int(fields.get("STACK", 0)) > self.STACK_BASELINE:
                                spills.append(line)
                else:
                    from tests.kernelkit.build import check_kernel_reg_spill_in_artifact

                    spills = check_kernel_reg_spill_in_artifact(
                        so_path,
                        stack_baseline=self.STACK_BASELINE,
                        suppress_checking_env_var="DEEP_SELECT_DISABLE_REG_SPILL_CHECK",
                    )
                if spills:
                    raise RuntimeError("Register spilling detected. Build failed!")

    return (ext_modules, SpillCheckBuildExtension)


build_target_platform = "CUDA"
overrided_platform = os.environ.get('DEEP_SELECT_BUILD_TARGET_PLATFORM', None)
if overrided_platform is not None and overrided_platform != "CUDA":
    raise ValueError("Invalid `DEEP_SELECT_BUILD_TARGET_PLATFORM`: only CUDA is supported")
print(f"Build target: {build_target_platform}")

try:
    cmd = ['git', 'rev-parse', '--short', 'HEAD']
    git_rev = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode('ascii').rstrip()
except Exception:
    # e.g. building from a source tarball that has no `.git`. Keep the version
    # a valid PEP 440 local version in that case.
    git_rev = "unknown"

datetime_rev = datetime.now().strftime("%Y%m%d.%H%M%S")

ext_modules, build_ext = build_on_cuda_platform()


setup(
    name="deep_select",
    version=f"{__version__}+{git_rev}.{datetime_rev}",
    packages=find_packages(include=['deep_select']),
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
