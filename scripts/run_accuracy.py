#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AddRmsNormBias kernel 精度验收驱动（解耦调试用）

流程：
  1) 对每个 case 生成 x/residual/gamma/bias（.bin）；
  2) 分别用 PyTorch 原生组合 (torch.add + torch.nn.functional.rms_norm +
     torch.add) 与 numpy fp32 golden 计算参考；
  3) 调用 build/add_rms_norm_bias_custom 跑 NPU kernel；
  4) 输出 kernel vs torch-native 与 kernel vs numpy-golden 的准确度报告，
     并单独报告每个解耦子步骤（add / rms / bias）参考自身的误差预算。

用法:
  cd build && python3 ../scripts/run_accuracy.py [--cases fp16_2d,...]
"""
import argparse
import os
import subprocess
import sys
import struct
import numpy as np
import torch

try:
    import torch_npu  # noqa: F401
    _HAS_NPU = torch_npu.npu.device_count() > 0
except Exception:
    _HAS_NPU = False

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from AddRmsNormBias import impl as numpy_golden  # noqa: E402

# ----------------------------------------------------------------------------
# dtype 映射
# ----------------------------------------------------------------------------
DTYPE_CODE = {np.float32: 0, np.float16: 1}
try:
    from ml_dtypes import bfloat16
    DTYPE_CODE[bfloat16] = 2
except Exception:
    bfloat16 = None

TOL = {np.float32: (1e-4, 1e-4), np.float16: (1e-3, 1e-3)}
if bfloat16 is not None:
    TOL[bfloat16] = (1e-3, 1e-3)

# 官方 tolerance：允许误差元素占比 <= 1e-3
FRAC_ALLOWED = 1e-3

# vs torch-native 的判定容差：fp32/fp16 与官方一致；bf16 下 torch-native
# 自带的 rms_norm 仅约 ~2 ULP（见 torch-vs-numpy 列），故对 bf16 放 2 ULP。
TOL_TORCH = dict(TOL)
if bfloat16 is not None:
    TOL_TORCH[bfloat16] = (1.6e-2, 1.6e-2)

CASES = [
    # (name, shape, np_dtype, eps, range)
    ("fp16_2d_basic",     [1, 64],              np.float16, 1e-5, (-2, 2)),
    ("fp16_2d_many_rows", [2048, 64],           np.float16, 1e-5, (-2, 2)),
    ("fp16_3d",           [32, 64, 64],         np.float16, 1e-5, (-2, 2)),
    ("fp16_4d",           [4, 8, 8, 64],        np.float16, 1e-5, (-2, 2)),
    ("fp16_large_D",      [4, 4096],            np.float16, 1e-5, (-1, 1)),
    ("fp16_3d_large",     [1, 128, 512],        np.float16, 1e-5, (-1, 1)),
    ("fp16_D576",         [8, 576],             np.float16, 1e-5, (-2, 2)),
    ("fp16_D100_misaligned", [4, 100],          np.float16, 1e-5, (-2, 2)),
    ("fp16_2x32768",      [2, 32768],           np.float16, 1e-5, (-1, 1)),
    ("fp16_eps6",         [8, 256],             np.float16, 1e-6, (-2, 2)),
    ("fp32_2d",           [16, 128],            np.float32, 1e-5, (-1, 1)),
    ("fp32_3d_576",       [4, 32, 576],         np.float32, 1e-5, (-1, 1)),
    ("fp32_large_D",      [4, 4096],            np.float32, 1e-5, (-1, 1)),
    ("fp32_D100_misaligned", [4, 100],          np.float32, 1e-5, (-1, 1)),
]
if bfloat16 is not None:
    CASES += [
        ("bf16_2d",        [8, 256],            bfloat16, 1e-5, (-2, 2)),
        ("bf16_3d",        [4, 64, 128],        bfloat16, 1e-5, (-1, 1)),
    ]


def to_torch(dtype_np):
    if dtype_np == np.float32:
        return torch.float32
    if dtype_np == np.float16:
        return torch.float16
    return torch.bfloat16


def np2torch(arr, t):
    """ml_dtypes.bfloat16 不能直接 torch.from_numpy，统一经 float32 中转。"""
    return torch.from_numpy(np.ascontiguousarray(arr.astype(np.float32))).to(t)


def torch2np(ts):
    if ts.dtype == torch.bfloat16:
        return ts.float().cpu().numpy()
    return ts.cpu().numpy()


def stats(label, out, ref, rtol, atol):
    a = out.astype(np.float64).reshape(-1)
    b = ref.astype(np.float64).reshape(-1)
    if a.shape != b.shape:
        return {"label": label, "fail": True, "reason": "shape mismatch"}
    abs_diff = np.abs(a - b)
    denom = np.maximum(np.abs(b), 1e-7)
    rel_diff = abs_diff / denom
    ok = np.isclose(a, b, rtol=rtol, atol=atol, equal_nan=True)
    n_mis = int(np.sum(~ok))
    return {
        "label": label,
        "fail": False,
        "n": int(a.size),
        "n_mis": n_mis,
        "frac_mis": n_mis / a.size if a.size else 0.0,
        "max_abs": float(np.nanmax(abs_diff)) if a.size else 0.0,
        "mean_abs": float(np.nanmean(abs_diff)) if a.size else 0.0,
        "max_rel": float(np.nanmax(rel_diff)) if a.size else 0.0,
    }


def torch_chain(x, residual, gamma, bias, eps, device):
    """PyTorch 原生组合：add -> rms_norm -> add，与题目 3.1 一致。"""
    D = x.shape[-1]
    y = x + residual
    normed = torch.nn.functional.rms_norm(y, [D], weight=gamma, eps=eps)
    out = normed + bias
    return y, normed, out


def run_case(cfg, build_dir, device):
    name, shape, dtype_np, eps, (lo, hi) = cfg
    D = shape[-1]
    rng = np.random.default_rng(__import__("zlib").crc32(name.encode()))
    x = rng.uniform(lo, hi, shape).astype(dtype_np)
    r = rng.uniform(lo, hi, shape).astype(dtype_np)
    g = rng.uniform(0.8, 1.2, [D]).astype(dtype_np)
    b = rng.uniform(-0.3, 0.3, [D]).astype(dtype_np)

    inp_dir = os.path.join(build_dir, "input")
    out_dir = os.path.join(build_dir, "output")
    os.makedirs(inp_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    # npu 参考（原生 torch 组合）
    torch_out_np = None
    stage = {}
    if _HAS_NPU and device is not None:
        t = to_torch(dtype_np)
        x_t = np2torch(x, t).to(device)
        r_t = np2torch(r, t).to(device)
        g_t = np2torch(g, t).to(device)
        b_t = np2torch(b, t).to(device)
        y_t, normed_t, out_t = torch_chain(x_t, r_t, g_t, b_t, eps, device)
        torch_out_np = torch2np(out_t)
        # 解耦子步骤：torch native 各段（先各自回读）
        stage["add"] = torch2np(y_t)
        stage["rms"] = torch2np(normed_t)
        stage["full"] = torch_out_np
        # 子步骤与 numpy fp32 参考的偏差（显示解耦后各段自身的数值一致性）
        xf = x.astype(np.float32)
        rf = r.astype(np.float32)
        gf = g.astype(np.float32)
        bf = b.astype(np.float32)
        y_np = (xf + rf).astype(dtype_np)
        rms = np.sqrt(np.mean((xf + rf) ** 2, axis=-1, keepdims=True) + eps)
        normed_np = ((xf + rf) / rms * gf).astype(dtype_np)
        stage["add_diff"] = np.abs(torch2np(y_t).astype(np.float32) - y_np.astype(np.float32))
        stage["rms_diff"] = np.abs(torch2np(normed_t).astype(np.float32) - normed_np.astype(np.float32))

    # numpy golden（fp32 内部计算）
    golden = numpy_golden(x, r, g, b, epsilon=eps)

    # 写 .bin
    def wb(path, arr):
        arr.astype(dtype_np).tofile(path)

    wb(os.path.join(inp_dir, "x.bin"), x)
    wb(os.path.join(inp_dir, "residual.bin"), r)
    wb(os.path.join(inp_dir, "gamma.bin"), g)
    wb(os.path.join(inp_dir, "bias.bin"), b)
    wb(os.path.join(out_dir, "golden_output.bin"), golden)
    # CaseConfig: numDims int64 + shape[8] int64 + dtype int32 + eps float
    assert len(shape) <= 8
    fmt = "<q" + "q" * 8 + "if"
    cfg_bytes = struct.pack(fmt, len(shape), *(shape + [0] * (8 - len(shape))), DTYPE_CODE[dtype_np], eps)
    with open(os.path.join(inp_dir, "case.bin"), "wb") as f:
        f.write(cfg_bytes)

    # 跑 kernel
    exe = os.path.join(build_dir, "add_rms_norm_bias_custom")
    proc = subprocess.run([exe], cwd=build_dir, capture_output=True, text=True, timeout=300)
    if proc.returncode != 0:
        return {"name": name, "run_fail": True, "stderr": proc.stderr[-500:]}

    out_np = np.fromfile(os.path.join(out_dir, "output.bin"), dtype=dtype_np).reshape(shape)

    rtol, atol = TOL[dtype_np]
    rtol_t, atol_t = TOL_TORCH.get(dtype_np, TOL[dtype_np])
    res = {"name": name, "shape": shape, "dtype": str(dtype_np), "D": D, "eps": eps,
           "run_fail": False, "stage": {}, "probe": (float(out_np.reshape(-1)[0]), float(golden.reshape(-1)[0]))}
    if torch_out_np is not None:
        res["vs_torch"] = stats("kernel vs torch-native", out_np, torch_out_np, rtol_t, atol_t)
        res["torch_vs_np"] = stats("torch-native vs numpy-golden", torch_out_np, golden, rtol, atol)
    res["vs_numpy"] = stats("kernel vs numpy-golden", out_np, golden, rtol, atol)

    if stage:
        res["stage"]["add_max_abs"] = float(np.max(stage["add_diff"])) if stage["add_diff"].size else 0.0
        res["stage"]["rms_max_abs"] = float(np.max(stage["rms_diff"])) if stage["rms_diff"].size else 0.0
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=None, help="comma list of case names; default: all")
    ap.add_argument("--build", default=".", help="build dir")
    args = ap.parse_args()

    build_dir = os.path.abspath(args.build)
    selected = None
    if args.cases:
        selected = set(args.cases.split(","))
    cases = [c for c in CASES if selected is None or c[0] in selected]

    device = None
    if _HAS_NPU:
        try:
            device = "npu:0"
        except Exception:
            device = None
    print(f"torch={torch.__version__} npu={_HAS_NPU} device={device}")

    results = []
    for c in cases:
        print(f"--- case {c[0]} shape={c[1]} dtype={c[2].__name__} D={c[1][-1]} eps={c[3]} ---", flush=True)
        try:
            r = run_case(c, build_dir, device)
            results.append(r)
        except Exception as e:
            results.append({"name": c[0], "run_fail": True, "stderr": repr(e)})
            print("  ERROR:", e, flush=True)

    print("\n" + "=" * 130)
    hdr = f"{'case':26s} {'dtype':8s} {'D':>7s} {'#elem':>10s} | " \
          f"{'vs_torch':5s} {'max_abs':>10s} {'max_rel':>10s} {'frac':>8s} | " \
          f"{'vs_numpy':5s} {'max_abs':>10s} {'max_rel':>10s} {'frac':>8s}"
    print(hdr)
    print("-" * 130)
    for r in results:
        if r.get("run_fail"):
            print(f"{r['name']:26s} RUN FAILED: {r.get('stderr','')[:120]}")
            continue
        name = r["name"]
        nelem = int(np.prod(r["shape"]))
        row = f"{name:26s} {r['dtype']:8s} {r['D']:>7d} {nelem:>10d} | "
        for key in ("vs_torch", "vs_numpy"):
            s = r.get(key)
            if not s or s.get("fail"):
                row += f"{'n/a':5s} {'-':>10s} {'-':>10s} {'-':>8s} | "
            else:
                ok = "PASS" if s["frac_mis"] <= FRAC_ALLOWED else "FAIL"
                row += f"{ok:5s} {s['max_abs']:>10.3e} {s['max_rel']:>10.3e} {s['frac_mis']:>8.2e} | "
        print(row)
    print("=" * 130)
    # 解耦子步骤诊断：torch-native add / rms_norm 段相对 numpy fp32 参考的最大偏差
    print("\nstage diagnostics (torch-native sub-step vs numpy fp32 reference, max abs):")
    for r in results:
        if r.get("run_fail") or not r.get("stage"):
            continue
        st = r["stage"]
        print(f"  {r['name']:26s} add_max_abs={st.get('add_max_abs', float('nan')):.3e}  "
              f"rms_max_abs={st.get('rms_max_abs', float('nan')):.3e}")

    # 汇总
    n_run = sum(1 for r in results if not r.get("run_fail"))
    n_pass_t = sum(1 for r in results if not r.get("run_fail") and r.get("vs_torch") and not r["vs_torch"].get("fail") and r["vs_torch"]["frac_mis"] <= FRAC_ALLOWED)
    n_pass_n = sum(1 for r in results if not r.get("run_fail") and r.get("vs_numpy") and not r["vs_numpy"].get("fail") and r["vs_numpy"]["frac_mis"] <= FRAC_ALLOWED)
    print(f"ran={n_run}  pass(vs torch-native)={n_pass_t}  pass(vs numpy-golden)={n_pass_n}")
    if bfloat16 is not None:
        print("note: bf16 的 vs_torch 列使用 ~2ULP 容差（torch-native bf16 rms_norm 自身即偏离官方 golden）;")
    if n_run and n_pass_t == n_run and n_pass_n == n_run:
        print("ALL PASSED")
        return 0
    print("SOME FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
