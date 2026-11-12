# KL 消融：PTD-PO 的提升来自哪里

修复版 PTD-PO 在我们的配对评测中把九任务 avg@8 从 41.5675% 提到 54.9973%。这组消融把“是否开 PTD”和“reference KL 加在哪些轨迹上”拆开，看提升来自哪里。基座、数据、超参和评测与修复版完全相同（见仓库根目录 README）。

## 五个设置

| # | 设置 | arm | `algorithm.enable_ptd` | `algorithm.kl_mask_mode` |
| --- | --- | --- | --- | --- |
| 1 | GRPO，所有轨迹加 KL | `A1_grpo_kl` | false | `all` |
| 2 | GRPO，不加 KL | `A2_grpo_nokl` | false | `none` |
| 3 | GRPO，只对答对的轨迹加 KL | `A3_grpo_klcorrect` | false | `correct` |
| 4 | PTD-PO，答对的加 KL，答错的不加（答错的接受 PTD） | `A4_ptd_klcorrect` | true | `correct` |
| 5 | PTD-PO，PTD 生效的组内全部不加 KL（上游默认） | `A5`（即 `recorded_run/`） | true | `ptd_group` |

上游的 `compute_ptd_mask` 返回 `ptd_group_mask`，即组准确率 < `ptd_threshold`（1.0）的组里的**所有**轨迹，`dp_actor` 把这些轨迹的 ref KL 全部置零，包括组里答对的轨迹；只有 8 条全对的组保留 KL。所以已完成的修复版复现对应第 5 个设置。第 4 个设置需要本目录的补丁。

## 代码改动

`patch_kl_ablation.diff`（在仓库根目录执行 `git apply ablation/patch_kl_ablation.diff`）新增 `algorithm.kl_mask_mode`，默认 `ptd_group`，即上游原逻辑，不改变已有行为。

- `core_algos.compute_kl_keep_mask`：`all` / `none` / `correct`，其中 `correct` 指 `accuracy == 1.0`，与 `compute_ptd_mask` 的判定一致。
- `ray_trainer`：拿到逐轨迹 accuracy 后算出 `kl_keep_mask` 放进 batch，记录 `kl_ablation/keep_ratio`。
- `dp_actor`：非默认模式下 `kld *= kl_keep_mask`，取代 PTD 组置零；分母仍是全部 response token，与上游 PTD 组置零的写法一致。所有 arm 都额外记录未加 mask 的 `actor/kl_unmasked`。
- 非默认模式要求 `use_kl_loss=true`、`disable_kl=false`，否则报错，避免落到 reward 内 KL 的分支。“不加 KL”通过 mask 实现，ref log-prob 照常计算，四个 arm 的计算路径一致。

`ops/test_kl_mask.py` 是 CPU 单元测试（mask 语义、与上游语义对照、actor 算术、配置校验）。

## 运行

```bash
git apply ablation/patch_kl_ablation.diff
export PTD_ROOT="$(pwd)" PTD_MODEL_PATH=/path/to/Qwen3-VL-4B-Thinking
# venv、papo_eval、datasets、research 的准备同根目录 README；数据见 data/README.md
TS_MODE=smoke TS_JOB=n1 python3 ablation/ops/gpu_entry.py        # 每个 arm 1 步，检查 keep_ratio 等
TS_JOB=n1 TS_ARMS=A1_grpo_kl,A3_grpo_klcorrect python3 ablation/ops/gpu_entry.py
TS_JOB=n2 TS_ARMS=A4_ptd_klcorrect,A2_grpo_nokl python3 ablation/ops/gpu_entry.py   # 第二台机器，共享同一 PTD_ROOT
```

每个 arm 依次执行 train → 检查 step 198 完整 checkpoint → export + 重载生成 → 九任务评测。训练失败会从最近的 checkpoint 续训，最多 3 次，续训前归档 trainer 日志。`arms/<arm>/state/owner` 原子认领，两台机器不会跑同一个 arm。评测脚本与修复版相同，只改了 ROOT 和 GPU 型号检查（`PTD_EVAL_GPU`，默认 A100）。

## 实际执行

两台单机 8×A100-SXM4-80GB 并行，2026-11-10 开始：一台跑 A1 → A3，另一台跑 A4 → A2。A5 和原模型的分数来自 8×H100 上的修复版复现，跨硬件比较时会有生成层面的数值差。

GPU smoke：A1 step 1 的 keep=1.0，无 ptd_kl；A4 step 1 的 keep=0.5，等于 batch accuracy，ptd_kl=0.0056。正式运行的前几步：A1 每步 kl_loss 与 kl_unmasked 完全相等；A4 的 keep_ratio 与 accuracy 逐位相同，kl_loss 约为 kl_unmasked 的 24–31%。

### 训练过程（训练内指标，单 seed，不是最终结论）

| step | A5（H100）acc / 平均长度 | A4 acc / 长度 | A1 acc / 长度 |
| --- | --- | --- | --- |
| 10 | 0.487 / 2978 | 0.500 / 2960 | 0.609 / 2448 |
| 20 | 0.559 / 2676 | 0.563 / 2672 | 0.666 / 2190 |
| 29 | 0.581 / 2785 | 0.581 / 2806 | 0.682 / 2269 |
| 40 | 0.570 / 2783 | — | 0.725 / 2046 |

MMK12 内部验证（2000 题），格式为 accuracy / format。`format` 为 1 表示回答含 `</think>` 且其后有 `\boxed{}`，即在 4096 token 内结束思考并给出答案：

| step | A5（H100） | A4 | A1 |
| --- | --- | --- | --- |
| 40 | 0.353 / 0.334 | 0.359 / 0.341 | 0.460 / 0.493 |
| 80 | 0.392 / 0.473 | 0.403 / 0.458 | 0.686 / 0.949 |
| 120 | 0.407 / 0.523 | — | 0.706 / 0.970 |
| 198 | 0.428 / 0.581 | — | — |

到 2026-11-12 为止：A4 与 A5 几乎重合；A1 领先主要来自 format，即学会在预算内收尾（平均长度 3100 → 1700，PTD 下约 2600）。最终结论以九任务 avg@8 为准。

### 九任务结果

| 设置 | 九任务 avg@8 % | 五项通用 | 四项视觉 |
| --- | ---: | ---: | ---: |
| 原模型（H100） | 41.5675 | 38.6118 | 45.2622 |
| A5 修复版 PTD-PO（H100） | 54.9973 | 55.3956 | 54.4994 |
| A1 | 运行中 | | |
| A2 | 排队中 | | |
| A3 | 排队中 | | |
| A4 | 运行中 | | |

各 arm 完成后补充分数、逐任务结果、训练日志和截断比例。
