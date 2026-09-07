#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
本地用例精度校验（case0: fp16 [1,64]）：
  ./output/output.bin  vs  ./output/golden_output.bin

官方 fp16 判据: 相对误差 < 1e-3 且绝对误差 < 1e-3。
"""
import os
import sys
import numpy as np


def isclose_frac(output, golden, rtol, atol):
    a = output.astype(np.float32)
    b = golden.astype(np.float32)
    return np.isclose(a, b, rtol=rtol, atol=atol, equal_nan=True)


def verify_result(output_path, golden_path, dtype, rtol, atol, tol=0.0):
    output = np.fromfile(output_path, dtype=dtype)
    golden = np.fromfile(golden_path, dtype=dtype)
    total = golden.size
    if output.size != total:
        print(f"FAILED: size mismatch output={output.size} golden={total}")
        return False
    close = isclose_frac(output, golden, rtol, atol)
    errors = int(np.sum(~close))
    frac = errors / total if total else 0.0
    diff = np.abs(output.astype(np.float32) - golden.astype(np.float32))
    if frac <= tol:
        print(f"PASSED: {os.path.basename(output_path)} vs {os.path.basename(golden_path)}")
        if errors > 0:
            print(f"  Mismatched: {errors}/{total} ({frac*100:.2f}%), tol={tol}")
        print(f"  Max diff: {np.max(diff) if diff.size else 0.0}")
        return True
    print(f"FAILED: {os.path.basename(output_path)} vs {os.path.basename(golden_path)}")
    print(f"  Mismatched: {errors}/{total} ({frac*100:.2f}%), tol={tol}")
    print(f"  Max diff: {np.max(diff) if diff.size else 0.0}")
    return False


def main():
    # case0: fp16
    output_path = os.path.join("output", "output.bin")
    golden_path = os.path.join("output", "golden_output.bin")
    if not os.path.exists(output_path):
        print(f"FAILED: {output_path} not found")
        return 1
    if not os.path.exists(golden_path):
        print(f"FAILED: {golden_path} not found")
        return 1
    ok = verify_result(output_path, golden_path, np.float16, 0.001, 0.001, 0.001)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
