"""Download the datasets from Hugging Face (public, no login), convert them to the MedQA schema MDAgents
expects, and freeze the evaluation subsets.

  data/<name>/test.jsonl   evaluation subset: {id, question, options{A..}, answer_idx, answer, meta}
  data/<name>/train.jsonl  few-shot exampler pool
  data/subsets/<name>.txt  subset ids, so every run and every paper number uses the same questions

  medqa       MedQA US test (5 options, the MDAgents setting)        300 random      pool: MedQA train
  medmcqa     MedMCQA validation (NEET-PG, labels public)            500 by subject  pool: 2,000 from train
  medmcqa_hi  MedMCQA-Indic 'hi' = same questions, Hindi (Llama-4    same 500 ids    pool: 2,000 other Hindi
              Maverick translation by Eka Care; licence field empty)                  validation items
"""
import io
import json
import os
import random
import urllib.request
from collections import defaultdict

import pyarrow.parquet as pq

SEED, N_MEDQA, N_MEDMCQA, POOL = 0, 300, 500, 2000
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data')


def rows(dataset, config, split):
    """Every row of one HF split, via the Hub's auto-converted parquet files (cached under data/raw)."""
    api = f'https://datasets-server.huggingface.co/parquet?dataset={dataset}'
    files = [f for f in json.load(urllib.request.urlopen(api))['parquet_files']
             if f['config'] == config and f['split'] == split]
    assert files, f'no parquet files for {dataset} {config} {split}'
    out = []
    for f in files:
        path = os.path.join(DATA, 'raw', dataset.replace('/', '__'), config, split, os.path.basename(f['url']))
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            print('download', f['url'])
            urllib.request.urlretrieve(f['url'], path)
        out += pq.read_table(path).to_pylist()
    return out


def medqa(r, i, split):
    options = {o['key']: o['value'] for o in r['options']}
    return {'id': f'medqa-{split}-{i}', 'question': r['question'], 'options': options,
            'answer_idx': r['answer_idx'], 'answer': options[r['answer_idx']], 'meta': {'step': r['meta_info']}}


def medmcqa(r):
    options = {'A': r['opa'], 'B': r['opb'], 'C': r['opc'], 'D': r['opd']}
    idx = 'ABCD'[int(r['cop'])]  # HF version is 0-indexed (the original JSON was 1-indexed)
    return {'id': r['id'], 'question': r['question'], 'options': options, 'answer_idx': idx, 'answer': options[idx],
            'meta': {'subject': r['subject_name'], 'topic': r['topic_name'], 'choice_type': r['choice_type']}}


def stratified(items, n, key, rng):
    """n items, allocated to each stratum in proportion to its size (largest remainder)."""
    groups = defaultdict(list)
    for it in items:
        groups[key(it)].append(it)
    quota = {k: n * len(v) / len(items) for k, v in groups.items()}
    take = {k: int(q) for k, q in quota.items()}
    for k in sorted(quota, key=lambda k: quota[k] - take[k], reverse=True)[:n - sum(take.values())]:
        take[k] += 1
    out = [it for k in sorted(groups) for it in rng.sample(groups[k], take[k])]
    rng.shuffle(out)  # so any prefix (--num_samples 20/200) is a random sample, not a few subjects
    return out


def write(name, test, pool):
    os.makedirs(os.path.join(DATA, name), exist_ok=True)
    os.makedirs(os.path.join(DATA, 'subsets'), exist_ok=True)
    for fname, items in [('test.jsonl', test), ('train.jsonl', pool)]:
        with open(os.path.join(DATA, name, fname), 'w') as f:
            f.writelines(json.dumps(x, ensure_ascii=False) + '\n' for x in items)
    with open(os.path.join(DATA, 'subsets', f'{name}.txt'), 'w') as f:
        f.writelines(x['id'] + '\n' for x in test)
    print(f'{name}: {len(test)} test, {len(pool)} few-shot pool')


if __name__ == '__main__':
    mq_test = [medqa(r, i, 'test') for i, r in enumerate(rows('bigbio/med_qa', 'med_qa_en_source', 'test'))]
    mq_train = [medqa(r, i, 'train') for i, r in enumerate(rows('bigbio/med_qa', 'med_qa_en_source', 'train'))]
    assert len(mq_test) == 1273, len(mq_test)
    write('medqa', random.Random(SEED).sample(mq_test, N_MEDQA), mq_train)  # sample order is already random

    mm_dev = [medmcqa(r) for r in rows('openlifescienceai/medmcqa', 'default', 'validation')]
    mm_train = [medmcqa(r) for r in rows('openlifescienceai/medmcqa', 'default', 'train')]
    assert len(mm_dev) == 4183, len(mm_dev)
    mm_test = stratified(mm_dev, N_MEDMCQA, lambda x: x['meta']['subject'], random.Random(SEED))
    write('medmcqa', mm_test, random.Random(SEED).sample(mm_train, POOL))

    hi = {x['id']: x for x in (medmcqa(r) for r in rows('ekacare/MedMCQA-Indic', 'hi', 'test'))}
    ids = [x['id'] for x in mm_test]
    assert all(i in hi for i in ids), 'Hindi set is missing some subset ids'
    assert all(hi[i]['answer_idx'] == x['answer_idx'] for i, x in zip(ids, mm_test)), 'Hindi labels differ'
    rest = sorted(set(hi) - set(ids))
    write('medmcqa_hi', [hi[i] for i in ids], [hi[i] for i in random.Random(SEED).sample(rest, POOL)])
