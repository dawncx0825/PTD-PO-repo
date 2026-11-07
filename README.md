# PTD-PO：Qwen3-VL-4B-Thinking 修复与复现

本仓库保存我们实际完成的 **PTD-PO 官方主实验配置 + 限定实现修复 + 自生成 hint** 版本，包括完整训练源码、启动配置、测试、评测、checkpoint 导出和审计脚本。

基于 [XszNeverSleep/PTD-PO](https://github.com/XszNeverSleep/PTD-PO) 的固定提交 [`9953ea2`](https://github.com/XszNeverSleep/PTD-PO/commit/9953ea2b5c68aa6454b2fd693d5f5ff0fe34ed97)，保留 Apache-2.0 许可。`source/` 下的训练 Python 源码与本次已完成实验的验收哈希一致；发布时只对外围脚本的机器路径做参数化。代码审查指出的是该公开提交的问题，不能据此断言论文报告使用了相同的执行路径。

## 我们修了什么

| 修复 | 原问题 | 本仓库的改动 | 验证方式 |
| --- | --- | --- | --- |
| JSD 学生梯度 | JSD 中学生概率权重被 `detach()`，损失值可能正常，但梯度与目标公式不一致 | 移除学生 Top-K、学生尾部及 full-vocabulary JSD 共三处学生权重 detach；教师仍 stop-gradient，Top-K 索引逻辑保持不变 | 独立 JSD 公式、FP64 有限差分／gradcheck、学生梯度非零、教师无梯度 |
| 教师图像链路 | 数据集先 pop 图像再构造 hint 消息；teacher forward 未完整传入多模态输入 | 在 pop 前构造 hint 消息；三个 teacher forward 分支传入对应 micro-batch 的像素与 grid；教师使用自己的 token／mRoPE 位置，学生与教师共享同一原图处理结果 | 真实 Qwen3-VL 单图／多图、记录像素与 grid、消融图像后的输出敏感性、视觉参数梯度 |
| 空 hint 与长度 | 空 hint 可能进入错误监督；hint 展开后可能静默截断 | 空 hint 用合法原始图文上下文参与同步 forward，并以 `hint_valid` 排除 PTD/OPSD 监督；hint 超长显式报错 | 空 hint／全通过／全失败／混合分组、全量长度与图像审计 |
| Teacher Top-K 数值精度 | 低精度 logsumexp 与相减造成概率质量偏差 | 在 logsumexp 和相减前转 FP32，按 64 token 分块限制临时显存；模型参数与前向仍为 BF16，不把 Top-K 重新归一化到 1 | FP32 概率质量、数值误差及显存检查 |
| PTD 全局有效 token 分母 | 各 micro-batch 局部均值会使不同失败 token 数的批次权重失真 | 每次 optimizer minibatch 跨 DP rank 统计有效 PTD token，统一分母累积 numerator，并补偿 FSDP 梯度平均；PG/reference KL 保留原 response-token 归一化 | 18 组 PTD/OPSD、micro-batch／rank 条件，覆盖动态 batching 与零有效监督分组，对照独立梯度 |
| 训练审计与错误停止 | 仅看到进程或 checkpoint 不能证明真实更新和可恢复性 | 非有限 grad norm 立即停止；记录 PTD numerator、有效 token、实际 optimizer updates；完整 checkpoint 和导出重载审计 | 八卡 NCCL/FSDP、优化器／scheduler／RNG 恢复等价性、完整训练器 smoke、最终权重与图像生成检查 |

核心修改位置：

- [`core_algos.py`](source/verl/trainer/core_algos.py)：JSD 梯度。
- [`dataset.py`](source/verl/utils/dataset.py)：图像生命周期、hint、位置编码与长度检查。
- [`dp_actor.py`](source/verl/workers/actor/dp_actor.py)：teacher 图像、FP32 分块、全局分母、非有限梯度停止。
- [`ray_trainer.py`](source/verl/trainer/ray_trainer.py)：有效 hint 与失败轨迹监督掩码。

完整逐行差异见 [`patch.diff`](patch.diff)，说明见 [`patch_notes.md`](patch_notes.md)。

## 保留的设置与差异

本次不增加新方法，不更换教师更新机制，不做超参数搜索：冻结 reference teacher、JSD Top-K=100、PTD 系数 0.5、阈值 1.0、reference KL 系数 0.01；2 epochs，rollout batch=384，actor 输入 batch=128，n=8，学习率 1e-6；prompt/hint/response 上限分别为 8192/4096/4096。

- 实际训练为单机 **8×H100 80GB**。OOM 后仅按显存适配条款使用 update micro-batch=2、experience micro-batch=4，有效全局 batch、数据、n 和长度预算保持不变。
- 首次正式运行在第 36 步 OOM，之前的 35 步没有正式 checkpoint；归档后从原始权重重新训练，这些失败步数不计入最终 198 步。
- checkpoint 配置为 `save_freq=40`、`save_limit=5`、`save_model_only=false`、`find_last_checkpoint=true`。保留上游保护 best checkpoint 的保留策略。曾准备的每 5 步切换已取消，从未在正式运行中生效；相关日志合并 helper 仅作历史测试，命令入口已禁用。
- 保留公开实现的**学生 Top-K 支持集和教师尾部近似**，包括 150000 常数（模型实际词表 151936）；没有改成论文描述的学生／教师 Top-K 并集。PTD 全局归一化验证范围为 SP=1，其他 SP 配置显式拒绝。
- 使用此前 QC 接受的 **38,156 条自生成 hint**，原始 38,870 条中另 714 条未纳入。因此不能称为作者官方 WSPH 数据的严格数值复现。

更多边界见 [`deviations.md`](deviations.md)。

## 实际完成结果

2026-11-07 已完成 **198 rollout steps、594 optimizer updates**。日志连续，每个 rollout 三次更新，198 步均有 PTD 监督；完整最终 checkpoint 审计与合并重载通过。609 个张量发生变化，其中视觉张量 259 个；重载后真实图像生成检查通过。

### 与论文 PTD-PO 的并列对照

论文列取自 [PTD-PO v2 Table 1](https://arxiv.org/html/2606.07000v2#S2.T1) 的 **Qwen3-VL-4B-Thinking / PTD-PO** 行；本次结果来自 `recorded_run/{initial,final}_summary.json`。各列均为 avg@8（%），差值为百分点（pp）。论文均值沿用表中报告的两位小数，最后一列据此计算；负值表示本次低于论文。

| 任务 | 本次题数 | 我们原模型 % | 我们训练后 % | 论文 PTD-PO % | 我们训练增益 pp | 我们训练后 − 论文 pp |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MMK12 | 2000 | 24.3375 | 40.3438 | 73.84 | +16.0063 | -33.4963 |
| Geo3K | 601 | 38.5607 | 51.6431 | 64.79 | +13.0824 | -13.1469 |
| MathVerse | 2180 | 39.5298 | 61.4736 | 80.45 | +21.9438 | -18.9764 |
| MathVista | 1000 | 55.5375 | 62.6125 | 69.81 | +7.0750 | -7.1975 |
| We-Math | 1740 | 35.0934 | 60.9052 | 81.67 | +25.8118 | -20.7648 |
| MMMU-Pro | 1730 | 24.1402 | 29.8555 | 41.20 | +5.7153 | -11.3445 |
| Counting | 200 | 91.3750 | 91.6250 | 92.88 | +0.2500 | -1.2550 |
| MathVerse V | 1308 | 36.5348 | 57.3394 | 76.15 | +20.8047 | -18.8106 |
| LogicVista | 447 | 28.9989 | 39.1779 | 60.32 | +10.1790 | -21.1421 |
| **五项通用推理均值** | **7521** | **38.6118** | **55.3956** | **74.11** | **+16.7838** | **-18.7144** |
| **四项视觉依赖均值** | **3685** | **45.2622** | **54.4994** | **67.64** | **+9.2372** | **-13.1406** |
| **九任务等权平均** | **11206** | **41.5675** | **54.9973** | **71.23** | **+13.4298** | **-16.2327** |

我们在自己的配对评测中提高 **13.4298 pp**，但最终 **54.9973%** 仍比论文报告的 **71.23%** 低约 **16.2327 pp**。与论文差距最小的是 Counting（−1.2550 pp），最大的是 MMK12（−33.4963 pp）。这不是严格同条件复现：论文采用 WSPH，本次采用 38,156 条 QC 后自生成 hint，另有样本集合、限定修复、公开 Top-K 近似和执行适配等差异；不能把跨实验差距归因于某一个因素，也不能把相对我们原模型的提升解释为修复相对未修复 PTD-PO 的增益。此处只对齐 Table 1，不混入其他设置的表或把论文 SFT 当作原始模型基线。

原模型、最终模型分别有 **11,206 题 × 8 = 89,648 条回答**，缺题 0、空回答 0；每题 draw、token 数、原始输出哈希、判分聚合与完整分母独立审计 PASS。使用 PAPO `compute_accuracy_boxed_math`，这里的 avg@8 是八条独立回答得分的平均，不是 pass@8 或投票准确率。

两侧使用相同原图、原生 Thinking 模板及逐题 seed；temperature=1、top_p=1、top_k=50、max_tokens=4096，不输入 hint/gold。原模型八个 TP=1 逻辑 shard 在四卡上调度，最终八个 shard 在单机八卡上运行。原模型／最终模型 `finish_reason=length` 分别为 **60.2624% / 43.6886%**，因此结果受 4096 token 上限和 boxed 答案提取影响。没有独立 LLM judge；不能将提升单独归因于某一项代码修复。

论文对应参考均值为 71.23%，与本次数据来源、样本集合及实现边界不同。训练器最后一次内部验证 accuracy reward 为 42.79375%，不是上表九任务宏平均。详见 [`docs/experiment_report.md`](docs/experiment_report.md)。

## 代码与记录目录

```text
source/             完整 PTD-PO/EasyR1/verl 训练源码、原始示例与模型合并工具
ops/                准备、验收、训练门控、评测、导出、步数及 checkpoint 审计
tests/             数值、图像、分组/分布式梯度、checkpoint 恢复测试
launch.sh           本次最终训练配置（机器路径参数化）
resolved_config.yaml  历史实际展开配置（路径以占位符展示）
requirements-repro.txt  关键环境版本；environment.txt 为历史完整环境快照
recorded_run/       历史验收、步数、权重变化、processor 一致性、分母与结果摘要
patch.diff          相对固定上游提交的训练修复差异
research/           论文参考表
```

`recorded_run/` 是已完成实验的**历史证据**，不能复制到新运行根目录作为验收通过凭据。未上传模型权重、数据/hint、原图、逐题回答、运行缓存、私有集群账号与调度脚本；这些不是训练方法代码。上游插图通过原仓库引用。检查点切换监督器和自动删除资源的脚本也不属于公开运行入口。

## 在自己的环境运行

需要 Linux、Python 3.12、单机八张 H100、已下载的 Qwen3-VL-4B-Thinking 原始权重以及合法准备的数据。原始模型 revision 为 `1de27d8c51f12e819435303b9e84c4e25ba8401e`。本次验证环境：torch **2.8.0+cu128**、vLLM **0.11.0**、Transformers **4.57.6**、FlashAttention **2.8.3**、Ray **2.48.0**。

```bash
export PTD_ROOT="$(pwd)"
export PTD_MODEL_PATH=/absolute/read-only/path/to/Qwen3-VL-4B-Thinking
export PTD_INPUT_BUNDLE=/absolute/path/to/accepted-input-bundle
export PTD_PAPO_SOURCE=/absolute/path/to/pinned/PAPO-Eval
mkdir -p evidence state logs config datasets/eval
python3.12 -m venv venv
source venv/bin/activate
# 安装 CUDA 12.8 版 torch、兼容的 vLLM/FlashAttention，再安装其余固定依赖。
# requirements-repro.txt 是关键版本约束；environment.txt 是历史快照而非跨平台锁文件。
export PYTHONPATH="$PTD_ROOT/source:$PTD_ROOT/papo_eval"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
```

`PTD_PAPO_SOURCE` 必须是 PAPO-Eval 固定提交 `bee18f0800cfd324f7a342053beea5806c473b32` 的本地源码。输入 bundle 需包含 `datasets/selected/train_with_hint.jsonl`、`datasets/eval/all.jsonl`、`research/paper_table1_4b.json`；所有图片路径必须可读。训练行包括 `sample_id`、`problem`、`images`、`image_sha256`、`answer`、`prompt_with_hint`、`hint_source`；评测行包括 `sample_id`、`benchmark`、`source_index`、`problem`、`images`、`image_sha256`、`answer`，不含 hint。

```bash
python ops/prepare_assets.py       # 检查、哈希并复制已准备的输入；不生成替代 hint
python ops/audit_eval_assets.py    # 图像/协议检查
python ops/length_audit.py         # 全量 token 长度与 smoke 子集
bash ops/run_acceptance.sh        # 数值、两 rank CPU、八卡、真实模型、完整 trainer smoke
bash ops/start_pipeline.sh        # 新运行验收门控 → 原模型评测 → 正式训练 → 导出 → 最终评测
```

这些命令会真实占用八卡并训练，不是本地静态检查。新运行必须重新生成本机 `test_results.json`、模型/数据 manifest 和协议哈希；pipeline 不接受以历史 PASS 替代当前验收。默认完整分母为本次固定的 38,156 训练样本与 11,206 评测题，换数据需要单独记录实验，不能套用本次结果。

可单独运行 `python ops/run_eval.py --label initial` 或 `--label final`；后者依赖本次运行生成的 `evidence/final_checkpoint.json`。结果写入 `results/<label>/`。

恢复前必须归档日志并核验完整模型、优化器、scheduler、RNG、dataloader；上游 file logger 会重写日志，不能仅凭 `find_last_checkpoint=true` 跳过检查。不要在 Bash 正执行训练时原地改写 launcher：本次训练完成保存后，外层 shell 曾因疑似读取偏移退出 127，已完整归档后仅续接后处理，无训练重跑。`ops/post_train_pipeline_20261107.py` 保留该次续接门控作历史参考，不能直接用于没有对应归档证据的新实验。

## 测试范围

`tests/` 包含独立目标/有限差分、概率质量、18 组 PTD/OPSD 分组与跨 rank/micro-batch 梯度、真实图像/视觉梯度、完整 checkpoint 状态恢复、奖励掩码与日志合并检查。历史最终环境验收及完整 smoke 均 PASS，见 [`recorded_run/test_results.json`](recorded_run/test_results.json)。

公开打包额外验证：训练源码哈希与实际验收记录一致，Python/嵌入式 Python 与 Shell 语法检查，日志合并单元测试；公开副本未再次启动 GPU 训练。发布文件清单与机器路径适配记录见 `publication_manifest.json`。
