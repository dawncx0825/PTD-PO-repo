# 数据与 hint

本目录是修复版 PTD-PO 复现（`recorded_run/`）和 KL 消融（`ablation/`）实际使用的全部训练、验证、评测数据，包括逐题 hint。图像约 5.2 GB，放在本仓库的 GitHub Release 资产里，见下文。

| 文件 | 行数 | 用途 | 原始文件 SHA256（重写图像路径之前） |
| --- | ---: | --- | --- |
| `train_with_hint.jsonl.gz` | 38,156 | 训练集（ViRL39K 题目 + 自生成 hint） | `cb8296874611f8b851b9bef06afc2664fe6441603797b37a71ad4e69ff5ad105` |
| `mmk12_val.jsonl.gz` | 2,000 | 训练中的周期验证（MMK12） | `28c4a74e0a7b71b74db43cd7253b19e105090c19dae8bf806b3113c48584011f` |
| `eval_all.jsonl.gz` | 11,206 | 最终九任务评测（PAPO 公开题集），不含 hint | `6126127c43b4b050ac5881d7938f9e606365b555faf6dbccc0ec9ea587ab193a` |

与原文件的唯一区别是 `images` 改成了相对路径（`images/train/...`、`images/papo_eval/<子集>/...`），其他字段和行顺序不变。每张图的 SHA256 记录在 `image_sha256` 中。

## 字段

- 训练：`sample_id`、`source_index`（ViRL39K 原始行号）、`problem`、`answer`、`images`、`image_sha256`、`hint`、`hint_source`、`prompt_with_hint`（`problem` + 空行 + `hint`，即教师输入）。
- 验证/评测：`sample_id`、`source_index`、`benchmark`、`problem`、`answer`、`images`、`image_sha256`。

学生（被训练的策略）只看 `problem`；`prompt_with_hint` 只进冻结 reference teacher 的前向。评测不使用 hint 或 gold。

## Hint 是怎么来的

论文的 WSPH hint 未公开，这里的 hint 是我们自己按以下流程生成的，因此不能称为作者官方数据的严格数值复现。

1. 对 ViRL39K 全部 38,870 条题目，主生成模型 **Qwen3.5-122B-A10B**（revision `b5890131…`，thinking，temperature 1.0、top_p 0.95、top_k 20、presence_penalty 1.5，max 16384 tokens）生成一次，输入为图像、题目和 gold。系统提示见 `hint_generation/generator_system_prompt.txt`。
2. 独立质检模型 **Qwen3.5-27B**（贪心，非 thinking）判断：是否泄露答案或中间结果、是否给出完整解答、是否基于图像证据、是否与 gold 一致。提示见 `hint_generation/qc_system_prompt.txt`。只有 `valid=true`、三个泄露项都为 false、另两项都为 true 时才通过。
3. 主生成未通过的题目最多再试 3 个 Gemini 候选（temperature 1.0，其余为供应商默认），由同一个质检模型筛选，取第一个通过的。
4. 结果：通过 38,156 条，拒绝 714 条（不在本数据中）。

| `hint_source` | 条数 |
| --- | ---: |
| Qwen3.5-122B-A10B | 34,027 |
| gemini-3.1-pro-preview | 1,384 |
| gemini-3.7-flash | 1,083 |
| gemini-3.8-flash | 699 |
| gemini-3.5-flash | 555 |
| gemini-3-flash-preview | 408 |

## 图像（Release 资产）

Release `data-v1` 中有：

- `images_train.part1of2.tar`、`images_train.part2of2.tar`：训练图像 42,062 张，约 2.9 GB
- `images_papo_eval.part*.tar`：验证和评测图像，验证集与 MMK12 共用同一套文件，约 2.4 GB
- `SHA256SUMS`

```bash
export PTD_ROOT="$(pwd)"
# 把 Release 里的 images_*.tar 下载到某个目录，例如 ./release_assets
python data/prepare_data.py --tars ./release_assets
# 得到 datasets/train_with_hint.jsonl、datasets/mmk12_val.jsonl、datasets/eval/all.jsonl（绝对图像路径，逐图校验 SHA256）
```

## 来源与许可

题目、答案和图像来自公开数据集，著作权与许可归原作者：训练集为 [ViRL39K](https://huggingface.co/datasets/TIGER-Lab/ViRL39K)；验证和评测集为 PAPO 公开的九个评测子集（MMK12、Geo3K、MathVerse、MathVista、We-Math、MMMU-Pro、Counting、MathVerse Vision、LogicVista）。这里只为可复现性原样转存，使用时请遵守各数据集自身的许可。hint 文本由上述模型生成。
