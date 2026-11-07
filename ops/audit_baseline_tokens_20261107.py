import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
"""Token/EOS and alternative extraction audit, never changes official scores."""
import collections,json,sys,re
from pathlib import Path
from mathruler.grader import extract_boxed_content,grade_answer
R=Path(os.environ.get('PTD_ROOT', Path(__file__).resolve().parents[1]));O=Path(sys.argv[1])
from transformers import AutoTokenizer
tok=AutoTokenizer.from_pretrained(os.environ['PTD_MODEL_PATH'])
special={t:tok.convert_tokens_to_ids(t) for t in ['<think>','</think>','<|im_end|>','<|endoftext|>','<|im_start|>']}
rows=[json.loads(x) for x in (R/'datasets/eval/all.jsonl').read_text().splitlines()];scores=[json.loads(x) for x in (R/'results/initial/scores.jsonl').read_text().splitlines()]
agg=collections.defaultdict(collections.Counter)
for i,(row,score) in enumerate(zip(rows,scores)):
 d=json.loads((R/'results/initial/predictions'/f'{i:06d}.json').read_text());c=agg[row['benchmark']]
 for o,s in zip(sorted(d['outputs'],key=lambda x:x['draw']),score['scores']):
  ids=o['token_ids'];text=o['text'];c['responses']+=1
  for t,k in special.items():
   if k in ids:c[t]+=1
  if special['</think>'] in ids and '</think>' not in text:c['think_close_stripped_in_decode']+=1
  if special['<|endoftext|>'] in ids and ids.index(special['<|endoftext|>'])<len(ids)-1:c['continued_after_endoftext']+=1
  if special['<|im_end|>'] in ids and ids.index(special['<|im_end|>'])<len(ids)-1:c['continued_after_im_end']+=1
  # Same strict parser, official PTD reward whitespace cleanup, check only differing extraction.
  cleaned=re.sub(r'\s*(<|>|/)\s*',r'\1',text)
  a=extract_boxed_content(text);b=extract_boxed_content(cleaned)
  if a!=b:c['ptd_cleanup_extraction_changed']+=1;c['ptd_cleanup_score_delta']+=float(grade_answer(b,str(row['answer'])))-s
  if '</think>' in text:
   final=extract_boxed_content(text.rsplit('</think>',1)[1])
   if final!=a:c['final_only_score_delta']+=float(grade_answer(final,str(row['answer'])))-s
 if i%2000==0:print(i,flush=True)
(O/'token_checks.json').write_text(json.dumps({'special_ids':special,'benchmarks':agg},indent=2));print('TOKENS_DONE',flush=True)
