# PTD-PO Qwen3-VL-4B-Thinking 复现报告

状态：两轮训练、最终模型重载、原模型与最终模型九任务 avg@8 已完成。

配置：官方主实验配置＋限定实现修复＋用户授权自生成 hint；1 nodes, 8 H100 GPUs per node。数据为先前 QC 接受的 38,156 条，原 ViRL39K 的另 714 条未纳入。不能称作者原始 WSPH 数据的严格数值复现。

|任务|题数|原模型 avg@8 %|最终 avg@8 %|变化 pp|论文参考 %|
|---|---:|---:|---:|---:|---:|
|PAPO_MMK12|2000|24.3375|40.3438|+16.0063|73.84|
|hiyouga_geometry3k|601|38.5607|51.6431|+13.0824|64.79|
|AI4Math_MathVerse|2180|39.5298|61.4736|+21.9438|80.45|
|AI4Math_MathVista|1000|55.5375|62.6125|+7.0750|69.81|
|We_Math|1740|35.0934|60.9052|+25.8118|81.67|
|MMMU_MMMU_Pro|1730|24.1402|29.8555|+5.7153|41.20|
|BUAADreamer_clevr_count_70k|200|91.3750|91.6250|+0.2500|92.88|
|AI4Math_MathVerse_vision_dependent|1308|36.5348|57.3394|+20.8047|76.15|
|lscpku_LogicVista|447|28.9989|39.1779|+10.1790|60.32|

五任务均值：38.6118% → 55.3956%。
四任务均值：45.2622% → 54.4994%。
九任务等权均值：41.5675% → 54.9973%；论文参考 71.23%。

正式训练完成 198 rollout steps、日志核验 594 optimizer updates；完整状态与 dataloader 文件在 checkpoints/global_step_198。合并权重：<PTD_ROOT>/checkpoints/global_step_198/actor/huggingface

逐题八条输出与每条判分分别在 results/{initial,final}/predictions 和 scores.jsonl。评测 temperature=1、top_p=1、top_k=50（PAPO 公开评测器默认）、max_tokens=4096，使用官方 compute_accuracy_boxed_math；不使用 LLM judge、pass@8 或多数投票。

源代码修复、模型/数据哈希、最终环境、全量长度检查与测试在 patch.diff、model_manifest.json、data_manifest.json、environment.txt、length_audit.json 和 evidence/。保留公开学生 Top-K 支持及 150000 常数近似，其与论文并集定义不同；详见 deviations.md。

结果差异同时受到 hint 数据来源、训练样本集合、限定修复、运行环境和随机采样影响，不能仅归因于任一修复。

## 验收与解释边界

正式训练的 198 条 rollout 记录连续，每步 3 次 optimizer updates，合计 594；全部 actor 数值有限且 198 步均有 PTD 监督。step 198 的 25 个状态文件通过 CRC/SHA256 审计，包含八 rank 模型、优化器、scheduler/RNG 与 dataloader 状态。合并权重重新加载后图像生成通过；609 个张量实际更新，其中视觉张量 259 个。原模型与最终模型的 tokenizer 后端、聊天模板、特殊 token 和图像处理器完全一致。证据：`evidence/training_completed.json`、`evidence/checkpoint_198_audit.json`、`evidence/final_checkpoint.json`、`evidence/final_processor_parity.json`。

原模型和最终模型各覆盖 11,206 题、89,648 条回答，九任务完整分母、每题八个 draw、输入 ID、输出 token 数、scores 与 summary 聚合由 `ops/audit_final_results_20261107.py` 独立复核；逐题输出的 SHA256 清单保留于 evidence。正式验收结论以 `evidence/final_eval_integrity.json` 为准。九任务宏平均提高 13.4298 个百分点，九项均提高；这里是按任务等权平均，不是把所有题直接混合后的准确率。

两侧生成上限均为 4096 token。原模型 `finish_reason=length` 为 54,024/89,648（60.2624%），最终为 39,166/89,648（43.6886%）；两侧空回答均为 0。长度截断比例仍高，因此这些结果衡量固定 4096 上限下的表现，不等同于更长推理预算下的能力。判分沿用官方 boxed-math 答案提取与匹配，答案格式、是否生成可提取答案以及截断都会影响得分；没有另用人工或 LLM 判分校正。分母审计复核保存输出和判分聚合，没有将其称为独立语义判分。

最终训练内验证 accuracy reward 42.79375% 来自训练器验证流程；九任务中的 MMK12 最终 avg@8 为 40.34375%，二者生成流程及汇总协议不同，不混用。

原模型基线在四卡上运行八个逻辑 shard，最终评测在单机八卡上运行相同八个逻辑 shard；每个 shard 都是 TP=1，采样参数及逐题 seed 保持一致。原模型首轮评测耗时约 4.65 小时（含评分），最终约 2.07 小时（含评分）；因并发卡数和输出长度不同，不能直接据此声称模型吞吐提升。原基线 timing 保留在 `evidence/initial_eval_first_pass/`。

2026-11-07 训练器已完成最终验证与 step 198 保存，随后外层 shell 因疑似运行中脚本更新导致的读取偏移退出 127。完整 4.64 GB train.log、指标和失败状态已归档于 `evidence/post_train_shell_exit_20261107/`。完整 checkpoint 审计通过后，使用同一 dlm80 八卡布局仅续接步数核验、export、final_eval 和 report；没有重跑训练、修改数值设置或启用已取消的 checkpoint 频率切换。后处理 pipeline 退出 0。

独立全量审计最终 PASS：两侧均完整覆盖 11,206 题、89,648 回答，每题八个 draw、输出 token 数和评分聚合一致；pipeline.lock 已释放。预测文件哈希清单、完整分母证据和评分已同步本地。资源清理与监控关闭的最终回执在本地 `evidence/completion_20261107.json`；远端大体积原始输出、完整 checkpoint 和日志保留于本任务私有目录。

本地最终清理确认：生产 Pod、零 GPU 审计 Pod 已按名称、实验标签及 UID 前置条件删除，实时 GET 均为 404，原 n0/n1 也为 404；监控 `ptd-po` 已删除。

公开版说明：本文所述 evidence/、checkpoint、完整原始输出及资源回执为私有实验归档；公开摘要见 ../recorded_run/，本仓库不附带权重与原始数据。
