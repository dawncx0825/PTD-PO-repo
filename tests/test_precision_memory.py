import os
import torch,json,gc
from pathlib import Path
R=Path(__file__).resolve().parents[1];torch.manual_seed(17)
x=torch.randn(8,128,151936,device='cuda',dtype=torch.bfloat16);ids=x.topk(100,-1).indices
results={}
for name in ['original_bf16','fixed_fp32_chunk64']:
 gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();start=torch.cuda.memory_allocated()
 if name=='original_bf16':lp=(x.gather(-1,ids)-x.logsumexp(-1,keepdim=True)).float()
 else:lp=torch.cat([x[:,t:t+64].gather(-1,ids[:,t:t+64]).float()-x[:,t:t+64].float().logsumexp(-1,keepdim=True) for t in range(0,128,64)],1)
 torch.cuda.synchronize();peak=torch.cuda.max_memory_allocated();results[name]={'incremental_peak_bytes':peak-start,'total_peak_bytes':peak}
 if name=='fixed_fp32_chunk64':
  ref=x.float().log_softmax(-1).gather(-1,ids);torch.testing.assert_close(lp,ref,atol=3e-6,rtol=1e-6);results[name]['max_logprob_error']=(lp-ref).abs().max().item()
 del lp
results['added_incremental_peak_bytes']=results['fixed_fp32_chunk64']['incremental_peak_bytes']-results['original_bf16']['incremental_peak_bytes'];results['shape']=[8,128,151936];(R/'evidence/precision_memory.json').write_text(json.dumps(results,indent=2));print(results)
