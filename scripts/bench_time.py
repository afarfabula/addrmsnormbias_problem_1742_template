#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""本地性能测量：对每个 case 写输入并多次运行内核，输出平均耗时(us)。

不依赖 torch / torch_npu / 精度参考，仅统计 main.asc 打印的 [KERNTIME]。
用法:
  cd build && ARNBS_TIMES=200 python3 ../scripts/bench_time.py
"""
import os
import struct
import subprocess
import sys
import zlib
import numpy as np

try:
    from ml_dtypes import bfloat16
except Exception:
    bfloat16 = None

DTYPE_CODE = {np.float32: 0, np.float16: 1}
if bfloat16 is not None:
    DTYPE_CODE[bfloat16] = 2

CASES = [
    ("fp16_2d_basic",        [1, 64],         np.float16, 1e-5, (-2, 2)),
    ("fp16_2d_many_rows",    [2048, 64],      np.float16, 1e-5, (-2, 2)),
    ("fp16_3d",              [32, 64, 64],    np.float16, 1e-5, (-2, 2)),
    ("fp16_4d",              [4, 8, 8, 64],   np.float16, 1e-5, (-2, 2)),
    ("fp16_large_D",         [4, 4096],       np.float16, 1e-5, (-1, 1)),
    ("fp16_3d_large",        [1, 128, 512],   np.float16, 1e-5, (-1, 1)),
    ("fp16_D576",            [8, 576],        np.float16, 1e-5, (-2, 2)),
    ("fp16_D100_misaligned", [4, 100],        np.float16, 1e-5, (-2, 2)),
    ("fp16_2x32768",         [2, 32768],      np.float16, 1e-5, (-1, 1)),
    ("fp16_eps6",            [8, 256],        np.float16, 1e-6, (-2, 2)),
    ("fp32_2d",              [16, 128],       np.float32, 1e-5, (-1, 1)),
    ("fp32_3d_576",          [4, 32, 576],    np.float32, 1e-5, (-1, 1)),
    ("fp32_large_D",         [4, 4096],       np.float32, 1e-5, (-1, 1)),
    ("fp32_D100_misaligned", [4, 100],        np.float32, 1e-5, (-1, 1)),
]
if bfloat16 is not None:
    CASES += [
        ("bf16_2d",          [8, 256],        bfloat16, 1e-5, (-2, 2)),
        ("bf16_3d",          [4, 64, 128],    bfloat16, 1e-5, (-1, 1)),
    ]


def main():
    build = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else ".")
    times = int(os.environ.get("ARNBS_TIMES", "200"))
    exe = os.path.join(build, "add_rms_norm_bias_custom")
    inp = os.path.join(build, "input")
    os.makedirs(inp, exist_ok=True)
    env = dict(os.environ, ARNBS_TIMES=str(times))

    print(f"reps/case={times}  exe={exe}")
    print(f"{'case':24s} {'dtype':8s} {'shape':28s} {'#elem':>10s} | {'avg_us':>10s}")
    for name, shape, dt, eps, (lo, hi) in CASES:
        D = shape[-1]
        rng = np.random.default_rng(zlib.crc32(name.encode()))
        x = rng.uniform(lo, hi, shape).astype(dt)
        r = rng.uniform(lo, hi, shape).astype(dt)
        g = rng.uniform(0.8, 1.2, [D]).astype(dt)
        b = rng.uniform(-0.3, 0.3, [D]).astype(dt)

        def wb(path, arr):
            arr.tofile(path)

        wb(os.path.join(inp, "x.bin"), x)
        wb(os.path.join(inp, "residual.bin"), r)
        wb(os.path.join(inp, "gamma.bin"), g)
        wb(os.path.join(inp, "bias.bin"), b)
        assert len(shape) <= 8
        cfg = struct.pack("<q" + "q" * 8 + "if", len(shape),
                          *(shape + [0] * (8 - len(shape))), DTYPE_CODE[dt], eps)
        with open(os.path.join(inp, "case.bin"), "wb") as f:
            f.write(cfg)

        proc = subprocess.run([exe], cwd=build, capture_output=True, text=True,
                              timeout=600, env=env)
        avg_us = None
        for line in proc.stderr.splitlines():
            if "[KERNTIME]" in line and "dev_avg_us=" in line:
                avg_us = float(line.split("dev_avg_us=")[1].split()[0])
        if avg_us is None:
            print(f"{name:24s} FAILED rc={proc.returncode}: {proc.stderr[-300:]}")
            continue
        print(f"{name:24s} {str(dt):8s} {str(shape):28s} {x.size:>10d} | "
              f"{avg_us:>10.1f}")


if __name__ == "__main__":
    main()
