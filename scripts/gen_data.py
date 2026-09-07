#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为本地直调测试生成 case0 数据（与 scripts/AddRmsNormBias.py 的 numpy golden 对齐）。
在 build/ 目录下执行: python3 ../scripts/gen_data.py

生成:
  input/case.bin       CaseConfig(shape, dtype, epsilon)，与 main.asc 的 struct 一致
  input/x.bin ...      x/residual/gamma/bias
  output/golden_output.bin
"""
import os
import struct
import numpy as np

sys_path = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.insert(0, sys_path)
from AddRmsNormBias import impl

os.makedirs("input", exist_ok=True)
os.makedirs("output", exist_ok=True)

# ---- Case 0: fp16 [1, 64]，与 main/run 默认冒烟用例一致 ----
np.random.seed(42)
shape = [1, 64]
D = shape[-1]
dtype = np.float16

x = np.random.uniform(low=-2, high=2, size=shape).astype(dtype)
residual = np.random.uniform(low=-2, high=2, size=shape).astype(dtype)
gamma = np.random.uniform(low=0.8, high=1.2, size=(D,)).astype(dtype)
bias = np.random.uniform(low=-0.3, high=0.3, size=(D,)).astype(dtype)
epsilon = 1e-05

x.tofile("input/x.bin")
residual.tofile("input/residual.bin")
gamma.tofile("input/gamma.bin")
bias.tofile("input/bias.bin")

golden = impl(x, residual, gamma, bias, epsilon=epsilon)
golden.tofile("output/golden_output.bin")

# CaseConfig: int64 numDims + int64 shape[8] + int32 dtype + float epsilon
assert len(shape) <= 8
packed = struct.pack("<q" + "q" * 8 + "if", len(shape),
                     *(shape + [0] * (8 - len(shape))), 1, epsilon)
with open("input/case.bin", "wb") as f:
    f.write(packed)

print(f"Generated case0: shape={shape} D={D} dtype=fp16 eps={epsilon}")
print("Files: input/{case,x,residual,gamma,bias}.bin, output/golden_output.bin")
