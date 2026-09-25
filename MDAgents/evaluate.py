"""Score result files written by main.py / baselines.py.

  python evaluate.py output/*.jsonl                 one summary row per run (shards .sIofN are merged)
  python evaluate.py compare 'output/A*.jsonl' 'output/B*.jsonl'
                                                    paired: McNemar exact test + bootstrap 95% CI of the accuracy gap
  python evaluate.py selftest
"""
import glob
import json
import math
import os
import random
import re
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
PRICES = json.load(open(os.path.join(HERE, 'models.json')))

# ponytail: regex-only extraction; unparsed answers count as wrong and are reported. Add an LLM fallback if that rate is high.
PATTERNS = [r'(?i:answer|उत्तर)\s*(?i:is|:|：)?\s*(?i:option\s*)?\**\s*\(?([A-E])\b',  # "Answer: (C)", "answer is C"
            r'\(([A-E])\)',                                                           # "(C)"
            r'\b([A-E])\)',                                                           # "C) ..."
            r'^\s*\(?([A-E])\)?\.?\s*$']                                              # a line that is only "C"


def text_of(resp):
    """MDAgents stores {0.0: text} or {'majority': {0.0: text}}; baselines store plain text."""
    while isinstance(resp, dict):
        resp = next(iter(resp.values()), None)
    return resp or ''


def extract(resp):
    t = text_of(resp)
    for p in PATTERNS:
        m = re.findall(p, t, flags=re.M)
        if m:
            return m[-1]
    return None


def load(patterns):
    """{run name: [records]} with shards of the same run merged."""
    runs = defaultdict(list)
    for pat in patterns:
        for path in sorted(glob.glob(pat)):
            run = re.sub(r'\.s\d+of\d+$', '', os.path.basename(path)[:-len('.jsonl')])
            calls = defaultdict(list)
            cpath = os.path.join(os.path.dirname(path), 'calls', os.path.basename(path))
            if os.path.exists(cpath):
                for line in open(cpath):
                    c = json.loads(line)
                    calls[c['qid']].append(c)
            for line in open(path):
                r = json.loads(line)
                r['pred'], r['calls'] = extract(r['response']), calls.get(r['id'], [])
                runs[run].append(r)
    return runs


def macro_f1(gold, pred):
    f1s = []
    for c in sorted(set(gold)):
        tp = sum(g == c and p == c for g, p in zip(gold, pred))
        fp = sum(g != c and p == c for g, p in zip(gold, pred))
        fn = sum(g == c and p != c for g, p in zip(gold, pred))
        f1s.append(2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0)
    return sum(f1s) / len(f1s)


def cost(calls):
    total = defaultdict(float)
    for c in calls:
        p = PRICES.get(c['model'], {})
        if 'price_in' in p and c['in'] is not None:
            total[p['currency']] += (c['in'] * p['price_in'] + c['out'] * p['price_out']) / 1e6
    return total


def summarize(name, recs):
    n = len(recs)
    ok = [r['pred'] == r['label'] for r in recs]
    calls = [c for r in recs for c in r['calls']]
    inter = [r for r in recs if r['difficulty'] == 'intermediate' and r['trace'].get('num_yes')]
    by_diff = defaultdict(list)
    for r, o in zip(recs, ok):
        by_diff[r['difficulty']].append(o)
    print(f'\n== {name}  (n={n})')
    print(f'  accuracy {sum(ok) / n:.3f} | macro-F1 {macro_f1([r["label"] for r in recs], [r["pred"] for r in recs]):.3f}'
          f' | unparsed {sum(r["pred"] is None for r in recs) / n:.1%} | errors {sum(bool(r["error"]) for r in recs) / n:.1%}'
          f' | complexity fallback {sum(r.get("difficulty_fallback", False) for r in recs)}')
    print('  by complexity: ' + ', '.join(f'{d} {sum(v) / len(v):.3f} (n={len(v)})' for d, v in sorted(by_diff.items(), key=str)))
    if inter:
        print(f'  intermediate: silent (nobody spoke in round 1) {sum(r["trace"]["num_yes"][0][2] == 0 for r in inter) / len(inter):.1%}'
              f' | final answers reached moderator {sum(r["trace"].get("finals_collected", False) for r in inter) / len(inter):.1%}')
    if calls:
        tok = lambda k: sum(c[k] or 0 for c in calls) / n
        spend = cost(calls)
        print(f'  per question: {len(calls) / n:.1f} calls, {tok("in"):.0f} in / {tok("out"):.0f} out tokens, '
              f'{sum(r["sec"] for r in recs) / n:.1f}s' + ''.join(f', {v / n:.4f} {k}' for k, v in spend.items())
              + ''.join(f' | total {v:.2f} {k}' for k, v in spend.items()))


def compare(a, b, B=2000):
    ra, rb = {r['id']: r for r in a}, {r['id']: r for r in b}
    ids = sorted(set(ra) & set(rb))
    x = [ra[i]['pred'] == ra[i]['label'] for i in ids]
    y = [rb[i]['pred'] == rb[i]['label'] for i in ids]
    b_, c_ = sum(p and not q for p, q in zip(x, y)), sum(q and not p for p, q in zip(x, y))
    k, m = min(b_, c_), b_ + c_
    p = min(1.0, 2 * sum(math.comb(m, j) for j in range(k + 1)) / 2 ** m) if m else 1.0
    rng = random.Random(0)
    diffs = sorted(sum(x[j] - y[j] for j in idx) / len(ids)
                   for idx in ([rng.randrange(len(ids)) for _ in ids] for _ in range(B)))
    print(f'paired n={len(ids)}: A {sum(x) / len(ids):.3f} vs B {sum(y) / len(ids):.3f} | diff {(sum(x) - sum(y)) / len(ids):+.3f} '
          f'95% CI [{diffs[int(.025 * B)]:+.3f}, {diffs[int(.975 * B)]:+.3f}] | McNemar b={b_} c={c_} p={p:.4f}')


def selftest():
    cases = {'Answer: (C) 3rd pharyngeal arch': 'C', 'Answer: C) 2th pharyngeal arch': 'C', '**Answer:** (B)': 'B',
             'Step 1 rules out (A). The answer is D': 'D', 'Final answer: E': 'E', 'उत्तर: (B)': 'B',
             'A 45-year-old man most likely has (D) sarcoidosis': 'D', 'B': 'B', 'A patient with fever': None,
             'The answer is a combination of factors': None, '': None,
             "{0.0: 'x'}": None}
    for text, want in cases.items():
        assert extract(text) == want, (text, extract(text), want)
    assert extract({'majority': {0.0: 'Answer: (A)'}}) == 'A' and extract({'0.0': 'Answer: B'}) == 'B'
    assert abs(macro_f1(list('AABB'), list('AABA')) - (0.8 + 2 / 3) / 2) < 1e-9
    print('evaluate selftest ok')


if __name__ == '__main__':
    if sys.argv[1:2] == ['selftest']:
        selftest()
    elif sys.argv[1:2] == ['compare']:
        (_, a), (_, b) = sorted(load([sys.argv[2]]).items())[:1] + sorted(load([sys.argv[3]]).items())[:1]
        compare(a, b)
    else:
        for name, recs in load(sys.argv[1:]).items():
            summarize(name, recs)
