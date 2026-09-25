"""Single-agent baselines on the same frozen questions as main.py (RQ1): zero-shot, CoT, CoT-SC.
MDAgents' own 'basic' path (main.py --difficulty basic) is the few-shot single agent.
Output has the same schema as main.py, so evaluate.py scores both."""
import argparse
import json
import os
import random
import time
from collections import Counter

from tqdm import tqdm

import llm
from evaluate import extract
from utils import load_data, create_question

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='medqa')
parser.add_argument('--model', type=str, default='gpt-4o-mini', help='name in models.json')
parser.add_argument('--method', choices=['zeroshot', 'cot', 'cotsc'], default='zeroshot')
parser.add_argument('--sc_samples', type=int, default=5)
parser.add_argument('--sc_temperature', type=float, default=0.7)
parser.add_argument('--num_samples', type=int, default=100)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--shard', type=str, default='0/1')
parser.add_argument('--tag', type=str, default='')
args = parser.parse_args()
shard_i, shard_n = map(int, args.shard.split('/'))

SYSTEM = 'You are a helpful assistant that answers multiple choice questions about medical knowledge.'  # MDAgents' single-agent prompt
ZERO = "{q}\n\nReply with only the correct option, formatted as 'Answer: (X)'."
COT = "{q}\n\nLet's think step by step. Finish with the final answer formatted as 'Answer: (X)'."

test_qa, _ = load_data(args.dataset)
name = f'{args.model}_{args.dataset}_{args.method}{args.tag}' + (f'.s{shard_i}of{shard_n}' if shard_n > 1 else '')
os.makedirs('output/calls', exist_ok=True)
out_path, llm.CALL_LOG = f'output/{name}.jsonl', f'output/calls/{name}.jsonl'
done = {json.loads(l)['id'] for l in open(out_path)} if os.path.exists(out_path) else set()

for no, sample in enumerate(tqdm(test_qa[:args.num_samples])):
    if no % shard_n != shard_i or sample['id'] in done:
        continue
    llm.QID = sample['id']
    random.seed(f"{args.seed}-{sample['id']}")  # same option order as main.py
    t0, question, response, trace, error = time.time(), None, None, {}, None
    try:
        question, _ = create_question(sample, args.dataset)
        msgs = [{'role': 'system', 'content': SYSTEM},
                {'role': 'user', 'content': (ZERO if args.method == 'zeroshot' else COT).format(q=question)}]
        if args.method == 'cotsc':
            samples = [llm.chat(args.model, msgs, temperature=args.sc_temperature, role='cotsc')
                       for _ in range(args.sc_samples)]
            votes = [extract(s) for s in samples]
            top = Counter(v for v in votes if v).most_common(1)
            response, trace = (f'Answer: ({top[0][0]})' if top else ''), {'votes': votes, 'samples': samples}
        else:
            response = llm.chat(args.model, msgs, temperature=0.0, role=args.method)
    except Exception as e:
        error = repr(e)
        print(f'[ERROR] {error}')

    with open(out_path, 'a') as f:
        f.write(json.dumps({'id': sample['id'], 'question': question,
                            'label': sample['answer_idx'], 'answer': sample['answer'], 'options': sample['options'],
                            'response': response, 'difficulty': 'single', 'trace': trace, 'error': error,
                            'sec': round(time.time() - t0, 2), 'meta': sample.get('meta')}, ensure_ascii=False) + '\n')
