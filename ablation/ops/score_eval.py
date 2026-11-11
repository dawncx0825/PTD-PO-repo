import argparse,collections,concurrent.futures,hashlib,json,time
from pathlib import Path
from papo_eval.eval_utils import compute_accuracy_boxed_math
ROOT=Path(__import__('os').environ['PTD_ROOT'])
def grade(arg):
 row,record=arg
 return {'sample_id':row['sample_id'],'benchmark':row['benchmark'],'global_index':record['global_index'],'scores':[compute_accuracy_boxed_math(o['text'],str(row['answer'])) for o in sorted(record['outputs'],key=lambda o:o['draw'])],'finish_reasons':[o['finish_reason'] for o in record['outputs']],'empty':sum(not o['text'].strip() for o in record['outputs']),'tokens':sum(o['tokens'] for o in record['outputs'])}
def main(label):
 started=time.time();clock_start=time.monotonic()
 rows=[json.loads(x) for x in (ROOT/'datasets/eval/all.jsonl').read_text().splitlines()];run=ROOT/'results'/label;records=[]
 for idx,row in enumerate(rows):
  p=run/'predictions'/f'{idx:06d}.json';assert p.exists(),p
  d=json.loads(p.read_text());assert (d['sample_id'],d['benchmark'])==(row['sample_id'],row['benchmark']);assert sorted(o['draw'] for o in d['outputs'])==list(range(8));records.append(d)
 assert len(list((run/'predictions').glob('*.json')))==len(rows)
 with concurrent.futures.ProcessPoolExecutor(max_workers=16) as ex:graded=list(ex.map(grade,zip(rows,records),chunksize=8))
 out=run/'scores.jsonl';out.write_text(''.join(json.dumps(g)+'\n' for g in graded))
 by=collections.defaultdict(list)
 for g in graded:by[g['benchmark']].append(g)
 benchmarks={}
 for name,rr in by.items():
  scores=[s for r in rr for s in r['scores']];benchmarks[name]={'questions':len(rr),'responses':len(scores),'correct':sum(scores),'avg_at_8_percent':100*sum(scores)/len(scores),'empty':sum(r['empty'] for r in rr),'finish_reasons':dict(collections.Counter(f for r in rr for f in r['finish_reasons'])),'output_tokens':sum(r['tokens'] for r in rr)}
 order=json.loads((ROOT/'research/paper_table1_4b.json').read_text())['benchmark_order'];assert set(benchmarks)==set(order)
 summary={'label':label,'complete':True,'questions':len(rows),'responses':len(rows)*8,'missing':0,'benchmarks':benchmarks,'general_avg':sum(benchmarks[n]['avg_at_8_percent'] for n in order[:5])/5,'vision_avg':sum(benchmarks[n]['avg_at_8_percent'] for n in order[5:])/4,'overall_macro_avg':sum(benchmarks[n]['avg_at_8_percent'] for n in order)/9,'scores_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'scorer':'official PAPO-Eval compute_accuracy_boxed_math; avg@8, not pass@8','scorer_commit':'bee18f0800cfd324f7a342053beea5806c473b32'}
 (run/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
 timing={'label':label,'started_epoch':started,'finished_epoch':time.time(),'elapsed_s':time.monotonic()-clock_start,'questions':len(rows),'responses':len(rows)*8}
 (ROOT/'evidence'/f'score_{label}_timing.json').write_text(json.dumps(timing,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--label',required=True);main(p.parse_args().label)
