#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""任意 shape/dtype 的本地性能测量（官方口径近似：热循环 dev_avg_us）。

用法:
  cd build && ASCEND_VISIBLE_DEVICES=3 ARNBS_TIMES=300 \
      python3 ../scripts/bench_shape.py . 2048,2048,64 fp16
  # dtype: fp16 | bf16 | fp32；eps 可选第 4 参（默认 1e-5）
"""
import os, struct, subprocess, sys, numpy as np
try:
    from ml_dtypes import bfloat16
except Exception:
    bfloat16 = None

CODE = {"fp32": 0, "fp16": 1, "bf16": 2}
def npdt(dtn):
    if dtn == "bf16":
        if bfloat16 is None:
            raise SystemExit("bf16 needs ml_dtypes")
        return bfloat16
    return {"fp32": np.float32, "fp16": np.float16}[dtn]

def main():
    build = os.path.abspath(sys.argv[1])
    shape = [int(v) for v in sys.argv[2].split(",")]
    dtn = sys.argv[3]
    eps = float(sys.argv[4]) if len(sys.argv) > 4 else 1e-5
    reps = int(os.environ.get("ARNBS_TIMES", "300"))
    D = shape[-1]
    dt = npdt(dtn)
    rng = np.random.default_rng(1234)
    x = rng.uniform(-1, 1, shape).astype(dt)
    r = rng.uniform(-1, 1, shape).astype(dt)
    g = rng.uniform(0.8, 1.2, [D]).astype(dt)
    b = rng.uniform(-0.3, 0.3, [D]).astype(dt)
    inp = os.path.join(build, "input")
    os.makedirs(inp, exist_ok=True)
    for f, a in [("x.bin", x), ("residual.bin", r), ("gamma.bin", g), ("bias.bin", b)]:
        a.tofile(os.path.join(inp, f))
    cfg = struct.pack("<q" + "q" * 8 + "if", len(shape), *(shape + [0] * (8 - len(shape))), CODE[dtn], eps)
    open(os.path.join(inp, "case.bin"), "wb").write(cfg)
    env = dict(os.environ, ARNBS_NOSYNC="1", ARNBS_TIMES=str(reps))
    exe = os.path.join(build, "add_rms_norm_bias_custom")
    for _ in range(2):  # 首个 kernel 可能有频率/冷启动抖动，多跑一轮取最后
        p = subprocess.run([exe], cwd=build, capture_output=True, text=True, env=env)
    for line in p.stderr.splitlines():
        if "dev_avg_us" in line:
            print(shape, dtn, line.strip().split("dev_avg_us=")[-1])
            return
    sys.stderr.write(p.stderr)
    raise SystemExit(1)

if __name__ == "__main__":
    main()
