# 原模型与论文基线差距：逐项审计（2026-11-07）

**结论：已定位一处真实的图像输入流程差异，排除了模型误用、公开题集错配和常见提取问题；尚未完成 GPU 单变量重测，不能把全部 10.7625 pp 归因于图像处理或截断。**

比较的是论文 v2 Appendix E.1 / Table 3 的 Qwen3-VL-4B-Thinking 原模型行 **52.33%**，与本次原模型九任务等权 avg@8 **41.567531%**。差值沿用论文四舍五入后的均值。与训练后分数、Table 1 的 71.23% 分开处理。

## 证据范围

此次以只读方式检查实际保存的 11,206 题、89,648 条原模型回答、逐条 token ID、评分、源 parquet、图像和模型权重；仅使用 CPU，没有重跑训练或改写原分数。审计结果见 `research/baseline_gap_audit/`。公开结果不含逐题私有回答、原图、权重或账号信息。

论文没有提供 Table 3 该行对应的完整评测命令、逐题输出、权重 revision、processor 配置及环境锁文件。公开 PTD-PO `ptd-grpo` 分支可见训练代码，未找到完整九任务评测入口。因此以下严格区分“与论文文字对齐”“与 PAPO 公开代码对齐”和“与作者实际执行对齐”。后者不能仅凭前两项推定。Table 3 后的说明还明确提醒跨模型比较包含评测协议等差异。

## 逐项核对

| 项目 | 核查结果 | 对分差的判断 |
| --- | --- | --- |
| 原模型身份 | 实际读取的两个 safetensors 分片重新计算 SHA256，与官方 Qwen/Qwen3-VL-4B-Thinking 固定 revision 的 HF LFS 哈希一致 | 排除本次误用 Instruct 或换了模型；作者权重 revision 未公开 |
| Thinking 输入 | 原生模板确实以 `assistant\n<think>\n` 结尾；不是仅设置了一个无效开关 | 已开启 Thinking；不应为了追分随意关闭 |
| 公开题集版本 | 九个 parquet SHA256 全部与 PAPO_eval 公开 revision 一致；11,206 行的题干、gold、sample ID、顺序和图像路径逐行匹配 | 排除相对该公开题集的错题/错 gold/顺序错配；作者实际题目 ID 清单仍缺失 |
| 图像文件 | 11,206 次图像引用均重新核对本次固定清单中的 SHA256，且对应公开 parquet 图像路径 | 排除本次记录与当前图像文件不一致；未把此次检查称为对公开 ZIP 所有字节的重新下载比对 |
| 答案格式指令 | 九任务原始题干均已有 think/boxed 指令，原样保留 | 排除没有要求输出 boxed 的假设 |
| 提示词模板 | 本次为 Qwen3-VL 原生模板；PAPO 示例 `qwen2_vl` 有默认 system，且不预开 `<think>`；九个样本模板 token 长度差为 -9 | 确认与公开示例不同；示例针对旧模型，不能认定作者 Qwen3 也用该模板 |
| 实际 prompt | 每任务首题共九题，按本次图像参数重建后的 prompt token 数与保存记录完全一致 | 确认本次所审计处理路线与真实输入长度吻合；不是全量 token ID 逐字节重放 |
| 图像预处理 | 本次将 1024–589824 直接作为模型 processor 的 size；PAPO 在外层先缩放，再由 vLLM 使用模型默认 processor | **已确认的输入流程差异，见下节；分数影响待测** |
| n / 指标 | 每题八个独立得分平均，再九任务等权；完整分母 11206×8，非 pass@8、投票或合并题量加权 | 与论文指标文字一致 |
| temperature / top_p / 长度 | 本次 1 / 1 / 4096；论文主实验文字和 Table 2 声明相同 | 文字一致；Table 3 行级运行记录缺失，不能补造其实际配置 |
| top_k / 随机种子 / 实现 | 本次 top_k=50 确实是固定 PAPO 代码默认值；本次每题固定 seed、n=8 同批，PAPO 用循环生成 | 作者具体种子、top_k、库版本未公开；不是已证实的单独原因 |
| EOS / token 解码 | 全量输出中没有 `<\|endoftext\|>`；没有在 `<\|im_end\|>` 后继续生成；`</think>` 未被 skip-special 解码吞掉 | 本次 `generation_config='vllm'` 与自动配置的额外 EOS 差异，没有在这些已存回答中表现为漏停 |
| 判分 | 与 PAPO 相同的 boxed 提取及 mathruler 规则；解析器取最后 boxed，不是“错取第一个 boxed” | 常见规则调整几乎不能填补分差，详见下节 |
| 训练修复 / hint | 原模型在 PTD-PO 训练前评测，不输入 hint/gold | 自生成 hint、JSD 梯度和 teacher 修复不能解释这段原模型差距 |

模型 revision：`1de27d8c51f12e819435303b9e84c4e25ba8401e`。数据 revision：`6f55c6a9582bd307cfcafac0b0c668bad5341461`。PAPO-Eval commit：`bee18f0800cfd324f7a342053beea5806c473b32`。

## 图像处理的具体差异

本次路径：原图转 RGB → 模型 processor，显式 `shortest_edge=1024, longest_edge=589824`。

PAPO 公开默认路径：`_regularize_images` 按 1024–589824 外部 PIL 缩放（含 Qwen2VLPlugin 最小边与宽高比保护）→ 转 RGB → vLLM 模型 processor 默认配置。本模型默认 `shortest_edge=65536, longest_edge=16777216`。这里两个 size 值是像素面积预算，不是字面边长。

将预算放在不同处理层，不保证结果等价：小图是否继续放大、大图是否经历两次插值、32 像素对齐后的 grid 都可能不同。公开代码允许通过 `vllm_config` 覆盖默认值，作者未提供该参数，因此这里比较的是**公开 PAPO 默认路线**，不是已确认的作者内部路线。

- 全量 grid 通过已安装 processor 的 `smart_resize` 计算，并用 54 张实际 processor 输出校验公式。
- 11,206 张图中，**5,617 张（50.12%）**的 grid 不同，且本次视觉 token 更少；不是说 50.12% 的题一定被影响了答案。
- 每任务固定六个位置的 54 张样本中，**25 张**像素张量不同。该抽样不是随机抽样，不用它估计总体像素差异比例。
- MathVerse 首题原图 137×93，本次 grid `[1,6,8]`，合并后 **12 个视觉 token**；公开默认路线 grid `[1,14,20]`，合并后 **70 个视觉 token**。
- 图像大小统计：3,463 张原图低于模型默认 65,536 像素，2,727 张高于本次上限 589,824；不是所有差异都来自最小像素设置。

此前协议说明“保留 PAPO 像素预算”只描述两个数值，不足以表示端到端处理一致。本次明确更正这一解释。历史 `evaluation_protocol.json` 及分数保持原样，避免破坏既有审计哈希；原/最终模型之间使用同一路线的配对结果仍成立，但与论文严格等价的说法不成立。

## 89,648 条回答的失败位置

下列比例按全部回答计数，**不是九任务宏平均分数**。

| 回答类别 | 条数 |
| --- | ---: |
| 正确 | 31,633 |
| 错误 | 58,015 |
| 错误且无完整 boxed | 52,611 |
| 有完整 boxed 但错误 | 5,404 |
| 达到 4096 上限 | 54,024 |
| 达到上限、仍未输出 `</think>`、无完整 boxed | 49,668 |
| 达到上限但仍按既定规则判对 | 1,007 |

**90.6852% 的错答没有完整 boxed；55.4034% 的全部回答在预算耗尽时仍处于未结束的思考阶段，且无完整 boxed。** 这定位了本次大量失分发生的阶段，但不能据此认定这些回答延长后就会答对，更不能用其比例直接解释 10.76 pp 跨论文差距。对照论文原始回答缺失，无法比较作者的截断比例。

提取反事实只重判已有文本，不生成新答案：

- 对最后一个 boxed 不完整的情况，改取此前最后一个完整 boxed：8 条提取变化，只额外判对 **2 条**，九任务均值 **+0.002948 pp**，约从 41.567531% 到 41.570479%。这还是改变规则后的诊断值，正式分数不改。
- 单字母选项的有限规范化：额外判对 **0 条**。
- PTD 训练奖励所做的 `< > /` 周边空白清理：2 条提取变化，得分变化 **0**。
- 强制只读取 `</think>` 之后的 boxed 并不能提高本次分数，反而会丢掉部分已得分答案。未做基于 gold 的任意答案搜索，也没有用 LLM judge 替换规则。

这些结果只排除已检查的提取差异，不能证明所有语义等价答案都被完美判分。

## 因果归因尚缺的实验

优先做两臂配对：A 重跑本次处理，B 仅替换为 PAPO 外部缩放 + 默认模型 processor。相同官方原模型、题目、原生 Thinking 模板、逐题 seed、n=8、1/1/50、4096、环境和判分；两臂都新生成，避免把调度变化误当图像收益。先用每任务固定随机 64 题（576 题、每臂 4,608 回答）做定位，再决定是否全量验证。报告逐题配对差、九任务宏平均、按题 bootstrap 区间、截断率和无 boxed 率；小样本不冒充完整九任务结果。

若只改图像仍有大差，再单独检验模板差异；扩展 4096 长度只能作为预算敏感性分析，不能冒充论文协议。需要作者提供 Table 3 该行的权重 revision、评测命令/环境、实际 processor/template、题目 ID 清单和原始输出，才能把剩余差距与其实际运行闭合。

本轮 CPU 审计完成时 H100 所属资源额度为 24/24 已占用，没有启动 GPU 对照、没有抢占其他任务，也没有创建排队训练。**当前状态：CPU_AUDIT_COMPLETE / CAUSAL_GPU_TEST_PENDING。** 不把“输入不同”“截断很多”或“CPU 审计通过”写成已解释完整分差。

## 来源

重做 CPU 审计时，须准备对应的完整历史运行目录和固定依赖环境。设置 `PTD_ROOT`、`PTD_MODEL_PATH`、`PTD_PUBLIC_EVAL_SOURCE`（含 `data/*.parquet` 的 PAPO 数据目录），然后依次执行 `ops/audit_baseline_{gap,tokens,images,grid,prompt}_20261107.py`，各自传入同一个新建审计输出目录。先执行 images，再执行依赖其样本输出的 grid。公开仓库不含原图、权重或逐题回答，不能仅凭公开摘要重新完成全量检查。

这些脚本不改写原始评分；gap 脚本输出包含私有错答片段的 `response_forensics.json` 只能留在私有目录。公开紧凑副本已移除 `examples`。历史检查使用实际私有环境运行，公开代码副本仅将输入路径参数化并做语法检查。

- [PTD-PO v2：§3.1、Appendix B/Table 2、Appendix E.1/Table 3](https://arxiv.org/html/2606.07000v2)
- [固定 PAPO 推理入口：外层图像缩放、SamplingParams、vLLM 配置](https://github.com/xhguo7/PAPO-Eval/blob/bee18f0800cfd324f7a342053beea5806c473b32/papo_eval/run_infer.py)
- [固定 PAPO 多模态插件：像素面积缩放及 Qwen2VL 保护](https://github.com/xhguo7/PAPO-Eval/blob/bee18f0800cfd324f7a342053beea5806c473b32/src/llamafactory/data/mm_plugin.py)
- [固定 PAPO 模板](https://github.com/xhguo7/PAPO-Eval/blob/bee18f0800cfd324f7a342053beea5806c473b32/src/llamafactory/data/template.py)
- [官方 Thinking 模型 revision](https://huggingface.co/Qwen/Qwen3-VL-4B-Thinking/tree/1de27d8c51f12e819435303b9e84c4e25ba8401e)
- [PAPO 公开数据 revision](https://huggingface.co/datasets/PAPO-Galaxy/PAPO_eval/tree/6f55c6a9582bd307cfcafac0b0c668bad5341461)
