#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AddRmsNormBias 算子调试脚本
用 PyTorch 标准算子生成 golden，同时与 numpy golden 对比。
支持生成 .bin 文件供 C kernel 验证。

用法:
    python3 scripts/test_debug.py
"""

import os
import sys
import struct
import numpy as np
import torch

# ─── PyTorch 参考实现 ───
def torch_add_rms_norm_bias(x, residual, gamma, bias, eps=1e-5):
    """用纯 PyTorch 实现: output = RMSNorm(x + residual, gamma, eps) + bias"""
    D = x.shape[-1]
    # Step 1: 残差加法
    y = x + residual
    # Step 2: RMS 归一化 (沿最后一维)
    # torch.nn.functional.rms_norm 在 2.4+ 可用
    if hasattr(torch.nn.functional, 'rms_norm'):
        normed = torch.nn.functional.rms_norm(y, [D], weight=gamma, eps=eps)
    else:
        # 手动实现
        rms = torch.sqrt(torch.mean(y.float() ** 2, dim=-1, keepdim=True) + eps).to(y.dtype)
        normed = y / rms * gamma
    # Step 3: 偏置加法
    out = normed + bias
    return out


# ─── Numpy golden (与 AddRmsNormBias.py 一致) ───
def numpy_golden(x, residual, gamma, bias, epsilon=1e-5):
    """与 scripts/AddRmsNormBias.py 完全一致的 numpy golden"""
    orig_dtype = x.dtype
    x_f = x.astype(np.float32)
    r_f = residual.astype(np.float32)
    g_f = gamma.astype(np.float32)
    b_f = bias.astype(np.float32)
    y = x_f + r_f
    rms = np.sqrt(np.mean(y * y, axis=-1, keepdims=True) + epsilon)
    out = y / rms * g_f
    out = out + b_f
    if orig_dtype == np.float16:
        return out.astype(np.float16)
    if orig_dtype == np.float32:
        return out.astype(np.float32)
    return out


def compare(label, golden_np, torch_out_np, rtol=1e-3, atol=1e-3):
    """对比两组结果，打印详细 diff 信息"""
    g = golden_np.astype(np.float32).flatten()
    t = torch_out_np.astype(np.float32).flatten()
    if g.shape != t.shape:
        print(f"  [{label}] SHAPE MISMATCH: golden={g.shape} torch={t.shape}")
        return False
    abs_diff = np.abs(g - t)
    max_abs = float(np.max(abs_diff))
    mean_abs = float(np.mean(abs_diff))
    # 相对误差 (避开 0 附近)
    denom = np.maximum(np.abs(g), 1e-7)
    rel_diff = abs_diff / denom
    max_rel = float(np.max(rel_diff))
    close = np.isclose(g, t, rtol=rtol, atol=atol)
    n_fail = int(np.sum(~close))
    total = g.size
    pass_flag = n_fail == 0
    status = "PASS" if pass_flag else "FAIL"
    print(f"  [{label}] {status}  total={total}  fail={n_fail}  "
          f"max_abs={max_abs:.2e}  mean_abs={mean_abs:.2e}  max_rel={max_rel:.2e}")
    if not pass_flag and n_fail <= 10:
        idxs = np.where(~close)[0][:10]
        for i in idxs:
            print(f"    idx={i}  golden={g[i]:.6f}  torch={t[i]:.6f}  "
                  f"abs_diff={abs_diff[i]:.2e}  rel_diff={rel_diff[i]:.2e}")
    elif not pass_flag:
        idxs = np.where(~close)[0][:5]
        for i in idxs:
            print(f"    idx={i}  golden={g[i]:.6f}  torch={t[i]:.6f}  "
                  f"abs_diff={abs_diff[i]:.2e}  rel_diff={rel_diff[i]:.2e}")
        print(f"    ... ({n_fail} failures total)")
    return pass_flag


def save_bin(path, arr):
    arr.tofile(path)


# ─── 测试用例定义 ───
TEST_CASES = [
    # (name, shape, D, dtype_np, dtype_torch, eps, x_range, r_range, g_range, b_range)
    ("fp16_2d_basic",     [1, 64],         64,   np.float16, torch.float16, 1e-5, (-2,2), (-2,2), (0.8,1.2), (-0.3,0.3)),
    ("fp16_2d_large_D",   [4, 4096],      4096, np.float16, torch.float16, 1e-5, (-1,1), (-1,1), (0.9,1.1), (-0.1,0.1)),
    ("fp16_2d_unaligned", [2, 192],        192, np.float16, torch.float16, 1e-5, (-2,2), (-2,2), (0.8,1.2), (-0.3,0.3)),
    ("fp16_2d_576",       [3, 576],        576, np.float16, torch.float16, 1e-5, (-2,2), (-2,2), (0.8,1.2), (-0.3,0.3)),
    ("fp32_2d_basic",     [2, 64],          64, np.float32, torch.float32, 1e-5, (-1,1), (-1,1), (0.9,1.1), (-0.1,0.1)),
    ("fp32_2d_large_D",   [4, 4096],      4096, np.float32, torch.float32, 1e-5, (-1,1), (-1,1), (0.9,1.1), (-0.1,0.1)),
    ("fp16_3d_seq",       [2, 8, 128],     128, np.float16, torch.float16, 1e-5, (-2,2), (-2,2), (0.8,1.2), (-0.3,0.3)),
    ("fp16_3d_large",     [1, 128, 4096], 4096, np.float16, torch.float16, 1e-5, (-1,1), (-1,1), (0.9,1.1), (-0.1,0.1)),
    ("fp16_4d_heads",     [2, 4, 2, 64],   64, np.float16, torch.float16, 1e-5, (-2,2), (-2,2), (0.8,1.2), (-0.3,0.3)),
    ("fp16_2d_eps6",      [2, 256],        256, np.float16, torch.float16, 1e-6, (-2,2), (-2,2), (0.8,1.2), (-0.3,0.3)),
]


def run_test_case(tc):
    name, shape, D, dtype_np, dtype_torch, eps, xr, rr, gr, br = tc
    print(f"\n{'='*60}")
    print(f"TEST: {name}  shape={shape}  D={D}  dtype={dtype_np}  eps={eps}")
    print(f"{'='*60}")

    np.random.seed(42)
    x_np = np.random.uniform(xr[0], xr[1], size=shape).astype(dtype_np)
    r_np = np.random.uniform(rr[0], rr[1], size=shape).astype(dtype_np)
    g_np = np.random.uniform(gr[0], gr[1], size=(D,)).astype(dtype_np)
    b_np = np.random.uniform(br[0], br[1], size=(D,)).astype(dtype_np)

    # ── Numpy golden ──
    golden_np = numpy_golden(x_np, r_np, g_np, b_np, epsilon=eps)

    # ── PyTorch golden ──
    x_t = torch.from_numpy(x_np.astype(np.float32)).to(dtype_torch)
    r_t = torch.from_numpy(r_np.astype(np.float32)).to(dtype_torch)
    g_t = torch.from_numpy(g_np.astype(np.float32)).to(dtype_torch)
    b_t = torch.from_numpy(b_np.astype(np.float32)).to(dtype_torch)
    torch_out = torch_add_rms_norm_bias(x_t, r_t, g_t, b_t, eps=eps)
    torch_out_np = torch_out.cpu().numpy()

    # ── 对比 numpy golden vs PyTorch golden ──
    ok = compare("numpy vs torch", golden_np, torch_out_np,
                 rtol=1e-3 if dtype_np == np.float16 else 1e-4,
                 atol=1e-3 if dtype_np == np.float16 else 1e-4)

    # ── 打印前几个元素的详细值 ──
    g_flat = golden_np.flatten().astype(np.float32)[:8]
    t_flat = torch_out_np.flatten().astype(np.float32)[:8]
    print(f"  golden(first 8): {g_flat}")
    print(f"  torch (first 8): {t_flat}")

    # ── 保存 .bin 供 C kernel 使用 ──
    out_dir = os.path.join(os.path.dirname(__file__), "..", "build", "debug", name)
    os.makedirs(out_dir, exist_ok=True)
    save_bin(os.path.join(out_dir, "x.bin"), x_np)
    save_bin(os.path.join(out_dir, "residual.bin"), r_np)
    save_bin(os.path.join(out_dir, "gamma.bin"), g_np)
    save_bin(os.path.join(out_dir, "bias.bin"), b_np)
    save_bin(os.path.join(out_dir, "golden_output.bin"), golden_np)
    print(f"  Saved .bin files to: {out_dir}")

    return ok


def main():
    print("AddRmsNormBias 调试测试")
    print(f"PyTorch version: {torch.__version__}")
    print(f"NumPy version: {np.__version__}")
    has_rms_norm = hasattr(torch.nn.functional, 'rms_norm')
    print(f"torch.nn.functional.rms_norm available: {has_rms_norm}")

    all_pass = True
    results = []
    for tc in TEST_CASES:
        ok = run_test_case(tc)
        results.append((tc[0], ok))
        if not ok:
            all_pass = False

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}: {name}")
    print(f"\nOverall: {'ALL PASSED' if all_pass else 'SOME FAILED'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
