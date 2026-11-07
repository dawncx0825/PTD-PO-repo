import os
from pathlib import Path
import json
R=Path(str(Path(os.environ['PTD_ROOT'])))
a=json.loads((R/'results/initial/summary.json').read_text());b=json.loads((R/'results/final/summary.json').read_text());cp=json.loads((R/'evidence/final_checkpoint.json').read_text());assert a['complete'] and b['complete'] and cp['verified'] and cp['step']==198
training=json.loads((R/'evidence/training_completed.json').read_text());assert training['status']=='PASS' and training['optimizer_updates']==594
runtime=json.loads((R/'checkpoints/experiment_config.json').read_text())
hardware=f"{runtime['trainer']['nnodes']} nodes, {runtime['trainer']['n_gpus_per_node']} H100 GPUs per node"
order=['PAPO_MMK12','hiyouga_geometry3k','AI4Math_MathVerse','AI4Math_MathVista','We_Math','MMMU_MMMU_Pro','BUAADreamer_clevr_count_70k','AI4Math_MathVerse_vision_dependent','lscpku_LogicVista'];paper=[73.84,64.79,80.45,69.81,81.67,41.20,92.88,76.15,60.32]
lines=['# PTD-PO Qwen3-VL-4B-Thinking 复现报告','','状态：两轮训练、最终模型重载、原模型与最终模型九任务 avg@8 已完成。','','配置：官方主实验配置＋限定实现修复＋用户授权自生成 hint；两节点各 4 张 H100，共 8 张。数据为先前 QC 接受的 38,156 条，原 ViRL39K 的另 714 条未纳入。不能称作者原始 WSPH 数据的严格数值复现。','','|任务|题数|原模型 avg@8 %|最终 avg@8 %|变化 pp|论文参考 %|','|---|---:|---:|---:|---:|---:|']
lines=[line.replace('两节点各 4 张 H100，共 8 张',hardware) for line in lines]
for task,reference in zip(order,paper):
 x=a['benchmarks'][task];y=b['benchmarks'][task];assert x['questions']==y['questions'] and x['responses']==y['responses']==8*x['questions'];xx=x['avg_at_8_percent'];yy=y['avg_at_8_percent'];lines.append(f'|{task}|{x["questions"]}|{xx:.4f}|{yy:.4f}|{yy-xx:+.4f}|{reference:.2f}|')
lines+=['',f'五任务均值：{a["general_avg"]:.4f}% → {b["general_avg"]:.4f}%。',f'四任务均值：{a["vision_avg"]:.4f}% → {b["vision_avg"]:.4f}%。',f'九任务等权均值：{a["overall_macro_avg"]:.4f}% → {b["overall_macro_avg"]:.4f}%；论文参考 71.23%。','','正式训练完成 198 rollout steps、日志核验 594 optimizer updates；完整状态与 dataloader 文件在 checkpoints/global_step_198。合并权重：'+cp['hf_path'],'','逐题八条输出与每条判分分别在 results/{initial,final}/predictions 和 scores.jsonl。评测 temperature=1、top_p=1、top_k=50（PAPO 公开评测器默认）、max_tokens=4096，使用官方 compute_accuracy_boxed_math；不使用 LLM judge、pass@8 或多数投票。','','源代码修复、模型/数据哈希、最终环境、全量长度检查与测试在 patch.diff、model_manifest.json、data_manifest.json、environment.txt、length_audit.json 和 evidence/。保留公开学生 Top-K 支持及 150000 常数近似，其与论文并集定义不同；详见 deviations.md。','','结果差异同时受到 hint 数据来源、训练样本集合、限定修复、运行环境和随机采样影响，不能仅归因于任一修复。']
(R/'REPORT.md').write_text('\n'.join(lines)+'\n')
