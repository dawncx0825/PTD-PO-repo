"""Rebuild $PTD_ROOT/datasets from this repo's data/*.jsonl.gz and the Release image tars.

  python data/prepare_data.py --tars DIR_WITH_DOWNLOADED_TARS
Extracts images to $PTD_ROOT/datasets/images/{train,papo_eval}/..., rewrites the relative image paths to
absolute ones, and checks every image against the recorded image_sha256. Output layout matches launch.sh,
ablation/ops/launch_arm.sh and the eval scripts:
  datasets/train_with_hint.jsonl  datasets/mmk12_val.jsonl  datasets/eval/all.jsonl
"""
import argparse, gzip, hashlib, json, os, tarfile
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--tars', required=True)
ap.add_argument('--skip-hash', action='store_true')
a = ap.parse_args()
root = Path(os.environ['PTD_ROOT']).resolve()
here = Path(__file__).resolve().parent
ds = root / 'datasets'
ds.mkdir(exist_ok=True)
for t in sorted(Path(a.tars).glob('images_*.tar')):
    with tarfile.open(t) as tf:
        for m in tf.getmembers():
            assert m.isfile() and m.name.startswith('images/') and '..' not in Path(m.name).parts, m.name
        tf.extractall(ds)
    print('extracted', t.name)
for name, dst in [('train_with_hint', 'train_with_hint.jsonl'), ('mmk12_val', 'mmk12_val.jsonl'), ('eval_all', 'eval/all.jsonl')]:
    out = ds / dst
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with gzip.open(here / f'{name}.jsonl.gz', 'rt', encoding='utf-8') as f, out.open('w', encoding='utf-8') as w:
        for line in f:
            r = json.loads(line)
            r['images'] = [str(ds / p) for p in r['images']]
            if not a.skip_hash:
                for p, h in zip(r['images'], r['image_sha256']):
                    assert hashlib.sha256(Path(p).read_bytes()).hexdigest() == h, p
            w.write(json.dumps(r, ensure_ascii=False) + '\n'); n += 1
    print(dst, n, 'rows')
