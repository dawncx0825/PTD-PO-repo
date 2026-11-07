# 限定修复与验证

基线：XszNeverSleep/PTD-PO `9953ea2b5c68aa6454b2fd693d5f5ff0fe34ed97`。保留独立干净快照；完整差异见 patch.diff。

1. `core_algos.py`：仅移除 JSD 三处学生概率权重 detach，教师分布仍无梯度。保留学生 Top-K 支持集、150000 常数及尾部近似。
2. `dataset.py`、`dp_actor.py`：在图像字段 pop 前构造 hint 消息，教师采用相同图像处理结果、独立 mRoPE 和对应 micro-batch 的像素/grid。三个 teacher 输出分支一致传图像。空 hint 使用合法原始上下文参加同步 forward，同时以 hint_valid 排除监督；ray_trainer.py 对应使用该标记。
3. `dp_actor.py`：teacher Top-K logsumexp 和减法采用 FP32，以 64 token 分块限制临时显存；模型前向仍为 BF16。
4. `dp_actor.py`：每个 optimizer minibatch 跨 DP rank 统计有效 PTD token，以同一分母累计 micro-batch numerator，并补偿 FSDP 梯度平均。PG/ref KL 仍按上游全部 response token 归一化。SP=1 为已验证范围；其他 SP 明确报错。
5. Hint 展开长度超限显式报错，不静默截断。非有限 grad norm 显式停止。增加 PTD numerator 和实际 optimizer update 计数，仅供审计；启用原生 file logger 在私有目录记录完整数值。

测试包括独立公式/有限差分、FP32 概率质量、18 种 PTD/OPSD 分组与跨卡梯度条件（含动态 batching）、真实 Qwen3-VL 图像路径及视觉参数梯度、8 卡完整状态恢复等价性。测试状态与所用环境必须以 evidence 和 test_results.json 为准；早期环境通过不代替最终训练环境的验收。

完整训练流程的 smoke 与正式训练隔离，正式运行从共享原模型初始化。测试使用的小型 SGD 更新仅验证梯度，不更改正式 adamw_bf16 优化器。
