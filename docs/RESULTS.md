# AddRmsNormBias 提交/测试结果记录

> 目的：把每一次 `git commit` 对应的精度与性能测试结果固定下来，方便回溯与对比。
>
> 约定：
> - 本地核时（`dev_avg_us`）：`cd build && ASCEND_VISIBLE_DEVICES=3 ARNBS_NOSYNC=1 ARNBS_TIMES=300 ./add_rms_norm_bias_custom`
>   或 `python3 /tmp|../scripts/runone` 生成指定 shape。300 次热迭代均值；注明 `reps` 若不同。
> - 精度回归：`cd build && ASCEND_VISIBLE_DEVICES=3 python3 ../scripts/run_accuracy.py`（16 case，
>   vs torch-native 与 vs numpy-golden，误差元素占比 <=1e-3 判 PASS）。
> - 重负荷回归：`cd build && ASCEND_VISIBLE_DEVICES=3 python3 ../scripts/run_heavy_accuracy.py`。
> - 榜单（官方评测系统）15 点成绩由用户回贴记录，未回贴的 commit 阶段标 `-`。

---

## ed5d384 / b056786 — 题目模板与说明
- 内容：初始模板 + PROBLEM.md。
- 状态：仅脚手架，无自研 kernel；无评测记录。

## dcdc538 — kernel version 1（初版）
- 内容：首版可跑 kernel（逐 chunk/逐行实现）。
- 状态：早期本地无系统性记录。

## b8bd2fe — npu on site（现场适配）
- 内容：切到本机 NPU 环境跑通。
- 状态：无系统性记录。

## a2f68fb — Fix AddRmsNormBias kernel（跨 pipe barrier，重写 TBuf-only）
- 内容：修正正确性问题，改为 TBuf-only 无 TQue。
- 状态：本地精度回归通过（此阶段开始有规律验证）。

## a177385 — Perf v2（chunked + tree-reduce + param cache + 计时 harness）
- 内容：v2 行内 1024/2048 大块、fp32 二叉归约、gamma/bias 小 D 常驻 UB；新增本地计时。
- 已知本地（v2 期历史记录，reps≈300）：
  - fp16 `[2048,64]` ~130µs →（合入 keepY 后）~43µs
  - fp16 `[2,32768]` ~598µs → ~25µs

## 4533c9d — Perf: keepY + tiling 上传只做一次
- 内容：单块行 pass1 把 y 留在 UB 省二次读；tiling 设备缓冲每进程只上传一次。
- 引入隐患：同进程多 case 复用旧 tiling → 评测 14/15 大面积 wrong answer（见 6ec695b）。

## 6ec695b — 修复 tiling 缓存跨 case 复用 bug + probe_multi
- 内容：tiling 设备缓冲按内容缓存；内容变化分配新缓冲，绝不覆盖可能被旧 kernel
  读取的缓冲。新增 `probe_multi.asc`（同进程顺序多 case 复现评测调用形态）。
- 精度：16 case 同进程回归全过；`run_accuracy.py` 16/16 ALL PASSED。
- 评测 15 点（此阶段，官方结果）：
  | # | 用时 | 最优 |
  |--|--|--|
  |1|6.88µs|1.47µs|
  |2|5.19µs|2.34µs|
  |3|7.50µs|2.56µs|
  |4|35.90µs|7.72µs|
  |5|29.46µs|5.70µs|
  |6|67.73µs|12.76µs|
  |7|128.48µs|17.09µs|
  |8|277.73µs|31.86µs|
  |9|145.68µs|68.58µs|
  |10|248.65µs|48.14µs|
  |11|434.93µs|154.07µs|
  |12|288.65µs|76.68µs|
  |13|1.21ms|419.91µs|
  |14|71.0ms|3.75ms|
  |15|19.0ms|8.66ms|

## 1cef497 — Perf v3/v4：批量行分组 kernel + 硬件 WholeReduceSum
- 内容：新增 `AddRmsNormBiasGroup`（D%64==0 且 D<=512 的海量行小 D 场景），
  tile 批量向量化、`WholeReduceSum` 行归约、单 V_S/S2V 每组、`FusedMulAdd`、去掉组内冗余 PIPE_V 屏障；
  新增 `scripts/run_heavy_accuracy.py`。
- 精度：`run_accuracy.py` 16/16 ALL PASSED；重负荷抽样 8/8 PASS。
- 本地热迭代（reps=300，稳态）：
  | shape | 上一阶段 | 本 commit | 提升 |
  |--|--|--|--|
  | fp16 `[2048,64]` | 42.9µs | ~10.2µs | 4.2x |
  | fp16 `[8192,64]` | ~300µs | ~23.4µs | ~13x |
  | fp16 `[32768,64]` | 748µs | ~77µs | ~10x |
  | fp16 `[2048,2048,64]`（重负荷 4M 行） | ~78ms | ~9.3ms | 8.4x |
  | fp16 `[1024,1024,64]` | ~19.6ms | ~2.35ms | 8.3x |
  | fp16 `[8192,64,128]` | ~10.7ms | ~1.54ms | ~7x |
- 评测 15 点（此阶段，官方结果）：
  | # | 用时 | 最优 | 较 6ec695b |
  |--|--|--|--|
  |1|4.25µs|1.47µs|6.88→4.25|
  |2|5.17µs|2.34µs|≈|
  |3|4.54µs|2.56µs|7.50→4.54|
  |4|36.18µs|7.72µs|≈|
  |5|29.64µs|5.70µs|≈|
  |6|67.89µs|12.76µs|≈|
  |7|128.88µs|17.09µs|≈|
  |8|84.72µs|31.86µs|277.7→84.7|
  |9|144.57µs|68.58µs|≈|
  |10|250.22µs|48.14µs|≈|
  |11|437.48µs|154.07µs|≈|
  |12|289.28µs|76.68µs|≈|
  |13|1.21ms|419.91µs|≈|
  |14|10.6ms|3.75ms|71ms→10.6ms|
  |15|19.0ms|8.66ms|≈|
- 小结：只吃到 v3 分组路径的点（#1/#3/#8/#14）；#4~#7、#9~#13、#15 等
  宽行/非 64 对齐/旧路径点基本未动 → 触发 v5（多核 split-D + repeat 批量归约）。

## <next> — v5：多核 split-D 两阶段 + repeat 批量行归约
- 内容：待记录（见 git log）。
- 精度：待记录。
- 评测：待用户回贴。
