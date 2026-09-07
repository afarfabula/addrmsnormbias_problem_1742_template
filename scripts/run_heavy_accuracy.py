#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
本地“重负荷/长时间”用例精度+性能回归（复现评测中 ms 级大张量场景，
如 海量行 x 小 D 的 3D fp16/fp32/bf16）。

对每个 HEAVY_CASES 生成输入并运行 kernel（每 case 独立进程，与官方类似），
同时用 numpy fp32 golden 校验并报告 dev_avg_us。

用法:
  cd build && python3 ../scripts/run_heavy_accuracy.py [--times 50] [--limit n]
"""
import argparse, os, struct, subprocess, sys, time, zlib
import numpy as np
try:
    from ml_dtypes import bfloat16
except Exception:
    bfloat16 = None

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from AddRmsNormBias import impl as numpy_golden

DTYPE_CODE = {np.float32: 0, np.float16: 1}
if bfloat16 is not None:
    DTYPE_CODE[bfloat16] = 2
TOL = {np.float32: (1e-4, 1e-4), np.float16: (1e-3, 1e-3)}
if bfloat16 is not None:
    TOL[bfloat16] = (1e-3, 1e-3)
FRAC_ALLOWED = 1e-3

# (name, shape, np_dtype, eps, range)
HEAVY_CASES = [
    ("h_fp16_rows8192_d64",      [8192, 64],          np.float16, 1e-5, (-1, 1)),
    ("h_fp16_rows32768_d64",     [32768, 64],         np.float16, 1e-5, (-1, 1)),
    ("h_fp16_rows8192_d128",     [8192, 128],         np.float16, 1e-5, (-1, 1)),
    ("h_fp16_3d_1M_rows_d64",    [1024, 1024, 64],    np.float16, 1e-5, (-1, 1)),
    ("h_fp16_3d_2M_rows_d64",    [1024, 2048, 64],    np.float16, 1e-5, (-1, 1)),
    ("h_fp16_3d_4M_rows_d64",    [2048, 2048, 64],    np.float16, 1e-5, (-1, 1)),
    ("h_fp16_3d_rows524K_d128",  [8192, 64, 128],     np.float16, 1e-5, (-1, 1)),
    ("h_fp16_3d_rows131K_d512",  [4096, 32, 512],     np.float16, 1e-5, (-1, 1)),
    ("h_fp32_rows8192_d64",      [8192, 64],          np.float32, 1e-5, (-1, 1)),
    ("h_fp32_rows4096_d128",     [4096, 128],         np.float32, 1e-5, (-1, 1)),
    ("h_fp32_3d_rows1M_d64",     [1024, 1024, 64],    np.float32, 1e-5, (-1, 1)),
]
if bfloat16 is not None:
    HEAVY_CASES += [
        ("h_bf16_rows8192_d64",  [8192, 64],          bfloat16, 1e-5, (-1, 1)),
        ("h_bf16_3d_rows1M_d64", [1024, 1024, 64],    bfloat16, 1e-5, (-1, 1)),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", default=".")
    ap.add_argument("--times", type=int, default=30)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    build = os.path.abspath(args.build)
    inp = os.path.join(build, "input")
    os.makedirs(inp, exist_ok=True)
    exe = os.path.join(build, "add_rms_norm_bias_custom")
    cases = HEAVY_CASES[: args.limit] if args.limit else HEAVY_CASES

    n_pass = 0
    print(f"{'case':32s} {'dtype':7s} {'elems(M)':>9s} | {'avg_us':>10s} | verdict")
    for name, shape, dt, eps, (lo, hi) in cases:
        D = shape[-1]
        rng = np.random.default_rng(zlib.crc32(name.encode()))
        x = rng.uniform(lo, hi, shape).astype(dt)
        r = rng.uniform(lo, hi, shape).astype(dt)
        g = rng.uniform(0.8, 1.2, [D]).astype(dt)
        b = rng.uniform(-0.3, 0.3, [D]).astype(dt)
        golden = numpy_golden(x, r, g, b, epsilon=eps).astype(dt)

        for fn, arr in [("x.bin", x), ("residual.bin", r), ("gamma.bin", g), ("bias.bin", b)]:
            arr.tofile(os.path.join(inp, fn))
        cfg = struct.pack("<q" + "q" * 8 + "if", len(shape),
                          *(shape + [0] * (8 - len(shape))), DTYPE_CODE[dt], eps)
        with open(os.path.join(inp, "case.bin"), "wb") as f:
            f.write(cfg)
        env = dict(os.environ, ARNBS_TIMES=str(args.times), ARNBS_NOSYNC="1")
        t0 = time.time()
        proc = subprocess.run([exe], cwd=build, capture_output=True, text=True,
                              timeout=1800, env=env)
        gen = time.time() - t0
        avg_us = None
        for line in proc.stderr.splitlines():
            if "[KERNTIME]" in line and "dev_avg_us=" in line:
                avg_us = float(line.split("dev_avg_us=")[1].split()[0])
        if proc.returncode != 0 or avg_us is None:
            print(f"{name:32s} FAILED rc={proc.returncode} {proc.stderr[-200:]}")
            continue
        out = np.fromfile(os.path.join(build, "output/output.bin"), dtype=dt).reshape(shape)
        rt, at = TOL[dt]
        if dt == bfloat16:
            ok = (out.view(np.uint16) == golden.view(np.uint16))
            frac = 1.0 - float(ok.mean())
        else:
            frac = 1.0 - float(np.isclose(out.astype(np.float64), golden.astype(np.float64),
                                          rtol=rt, atol=at, equal_nan=True).mean())
        verdict = "PASS" if frac <= FRAC_ALLOWED else "FAIL"
        if verdict == "PASS":
            n_pass += 1
        print(f"{name:32s} {str(dt):7s} {x.size/1e6:>9.1f} | {avg_us:>10.1f} | {verdict}  (gen+run {gen:.1f}s)")
    print(f"heavy: pass={n_pass}/{len(cases)}")
    sys.exit(0 if n_pass == len(cases) else 1)


if __name__ == "__main__":
    main()
