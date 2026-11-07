# 实验差异

2026-11-05，首次正式运行完成 35 rollout 后在 experience log-prob forward OOM，无正式 checkpoint。失败现场已归档，experience microbatch 从 8 降为 4，update microbatch 为 2。完整八卡验收及 smoke 重新通过后，从原始权重重启正式流水线，有效批量和训练协议保持固定。旧运行步数不计入新运行。

2026-11-05 曾按用户要求准备每 5 步保存及 step 40 受控恢复，配置差异检查和四项日志合并测试通过；但用户随后恢复每 40 步保存，该切换从未在运行中生效。11:15 已停止等待中的监督进程，未重启或删除 Pod，也没有新增训练段。最新 launcher 显式设置 save_freq=40、save_limit=5、save_model_only=false、find_last_checkpoint=true；相对于已经验收的配置，只有启动时查找 checkpoint 的 false→true 是实际配置差异，其他三项与原默认值一致。原 test_results.json 与验收 hash 不重写，新 launcher 的严格文本差异及用户决定单独存证。此前每 5 步的脚本、测试与记录仅作历史证据，不得重启。

用户于 2026-11-03 授权使用此前自行生成的 hint，并指定共享模型路径。本次为「官方主实验配置＋限定实现修复＋自生成 hint」，不是作者官方 WSPH 数据的严格数值复现。

- 模型只读加载 `<PTD_MODEL_PATH>`。
- 产物写入本人 CephFS `<PTD_ROOT>`，不写共享模型目录。
- 保留公开学生 Top-K 支持集和教师尾部近似；`_topk_match_and_gather` 中常数为 150000，实际模型词表为 151936（已核实共享模型 config）。
- PTD 全局归一化限制为 Ulysses=1，其他 SP 配置显式拒绝；此限制与本次主配置相容。
- 教师概率归一化采用 64 token 分块 FP32，权重与前向保持 BF16。
- 空 hint 使用合法原始图文上下文同步 forward，`hint_valid=false` 排除监督。
- 调用方原有 student truncation=right；本次 hint 独立设置为 error，避免静默截断。
- 单节点 8 H100 与脚本默认双节点 16 GPU 不同，有效全局 batch 保持不变。
- 2026-11-04 完整训练器 smoke 在 update micro-batch=4 时于 `loss.backward()` 显存不足，单次额外申请为 19.16–22.34 GiB。依据任务书显存适配条款，update micro-batch 降为 2，experience micro-batch 为 8。动态 batching 及经八卡验证的梯度累计用于维持有效更新批量，2 epochs、n=8、rollout batch 384、actor batch 输入 128、响应长度 4096 保持原设置。失败日志、配置和 smoke 目录归档于 `evidence/acceptance_update_micro4_oom_20261104/`，调整后重新执行完整验收。

测试与训练结果只在实际完成后填入报告。

update micro-batch=2 的最终环境完整验收已于 2026-11-04 15:59 通过。完整训练器 smoke 完成 1 个 rollout step 和 2 次优化器更新，最大 allocated memory 为 46.37 GiB，完整八卡 checkpoint 已保存。正式训练从共享原始权重重新初始化，验收证据见 `test_results.json` 和 `evidence/integrated_smoke.json`。

- 评测沿用 PAPO 题目和官方 boxed-math 判分，生成使用 Qwen3-VL 原生 processor/chat template、n=8 批量采样；与 PAPO LLaMA-Factory 循环八次的包装方式不同。保留其 top_k=50 及像素预算，但按用户偏好将原图直接交给官方 processor，不先进行额外 resize。原模型与最终模型采用完全相同适配器；详见 evaluation_protocol.json。

2026-11-07：训练完成 198/594 和最终验证、checkpoint 保存后，外层 launch.sh 报 `ainer.find_last_checkpoint=true: command not found` 并退出 127，现象与运行中原地修改脚本后的 Bash 文件读取偏移一致。完整日志和状态归档、checkpoint 完整性审计 PASS 后，仅在同一 dlm80 八卡布局续接 export/final_eval/report，退出 0；未重跑训练，未改变模型或评测协议。详情见 evidence/post_train_recovery_20261107.json。原模型四卡/最终八卡运行同样八个 TP=1 逻辑评测 shard，已记录执行并发差异。
