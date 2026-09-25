import os
import json
import time
import random
import argparse
from tqdm import tqdm
from termcolor import cprint
from pptree import print_tree
from prettytable import PrettyTable
import llm
import utils
from utils import (
    Agent, Group, parse_hierarchy, parse_group_info,
    load_data, create_question, determine_difficulty,
    process_basic_query, process_intermediate_query, process_advanced_query
)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='medqa')
parser.add_argument('--model', type=str, default='gpt-4o-mini', help='name in models.json')
parser.add_argument('--aux_model', type=str, default=None,
                    help='complexity checker + intermediate recruiter (upstream: gpt-3.5-turbo); default = --model')
parser.add_argument('--difficulty', type=str, default='adaptive')
parser.add_argument('--num_samples', type=int, default=100)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--shard', type=str, default='0/1', help='i/n: this worker runs questions i, i+n, ... (parallel runs)')
parser.add_argument('--tag', type=str, default='')
args = parser.parse_args()
args.aux_model = args.aux_model or args.model
shard_i, shard_n = map(int, args.shard.split('/'))

test_qa, examplers = load_data(args.dataset)

name = f'{args.model}_{args.dataset}_{args.difficulty}{args.tag}' + (f'.s{shard_i}of{shard_n}' if shard_n > 1 else '')
os.makedirs('output/calls', exist_ok=True)
out_path, llm.CALL_LOG = f'output/{name}.jsonl', f'output/calls/{name}.jsonl'
done = {json.loads(l)['id'] for l in open(out_path)} if os.path.exists(out_path) else set()  # resume after a crash

for no, sample in enumerate(tqdm(test_qa[:args.num_samples])):
    if no % shard_n != shard_i or sample['id'] in done:
        continue

    print(f"\n[INFO] no: {no}")
    llm.QID = sample['id']
    random.seed(f"{args.seed}-{sample['id']}")  # same option order for a question in every run/shard
    utils.trace.clear()
    t0, question, difficulty, fallback, final_decision, error = time.time(), None, None, False, None, None

    try:
        question, img_path = create_question(sample, args.dataset)
        difficulty = determine_difficulty(question, args.difficulty, args.aux_model)
        if difficulty is None:  # checker named no level; upstream silently reused the previous answer
            difficulty, fallback = 'basic', True

        print(f"difficulty: {difficulty}")

        if difficulty == 'basic':
            final_decision = process_basic_query(question, examplers, args.model, args)
        elif difficulty == 'intermediate':
            final_decision = process_intermediate_query(question, examplers, args.model, args)
        elif difficulty == 'advanced':
            final_decision = process_advanced_query(question, args.model, args)
    except Exception as e:  # keep going; the error is recorded and scored as wrong
        error = repr(e)
        cprint(f"[ERROR] {error}", 'red')

    with open(out_path, 'a') as file:
        file.write(json.dumps({
            'id': sample['id'],
            'question': question,
            'label': sample['answer_idx'],
            'answer': sample['answer'],
            'options': sample['options'],
            'response': final_decision,
            'difficulty': difficulty,
            'difficulty_fallback': fallback,
            'trace': dict(utils.trace),
            'error': error,
            'sec': round(time.time() - t0, 2),
            'meta': sample.get('meta'),
        }, ensure_ascii=False) + '\n')
