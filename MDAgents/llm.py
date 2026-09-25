"""Single entry point for every LLM call in MDAgents: chat(model_name, messages, temperature, role) -> str.

Models are listed in models.json; anything with an OpenAI-compatible /chat/completions endpoint works
(OpenAI, Sarvam /v2, OpenRouter, a local vLLM / llama.cpp server). The "mock" model needs no key and
exists only so the whole pipeline can be tested offline. Every call is appended to CALL_LOG as JSONL.
"""
import hashlib
import json
import os
import re
import time

from openai import OpenAI

MODELS = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models.json')))
CALL_LOG = None   # set by the runner
QID = None        # current question id, set by the runner
_clients = {}


def _client(cfg):
    key = (cfg['base_url'], cfg.get('api_key_env'))
    if key not in _clients:
        env = cfg.get('api_key_env')
        api_key = os.environ.get(env) if env else 'local'
        if not api_key:
            raise RuntimeError(f"Set the environment variable {env} (see RUN.md)")
        _clients[key] = OpenAI(base_url=cfg['base_url'], api_key=api_key, max_retries=8, timeout=900,
                               default_headers=cfg.get('headers'))
    return _clients[key]


def chat(name, messages, temperature=None, role=None):
    cfg = MODELS[name]
    t0 = time.time()
    if cfg.get('mock'):
        text = _mock(messages)
        usage = (sum(len(m['content']) for m in messages) // 4, len(text) // 4)
    else:
        kw = {} if temperature is None else {'temperature': temperature}
        r = _client(cfg).chat.completions.create(model=cfg['model'], messages=messages,
                                                 extra_body=cfg.get('extra'), **kw)
        text = r.choices[0].message.content or ''
        usage = (r.usage.prompt_tokens, r.usage.completion_tokens) if r.usage else (None, None)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.S).strip()   # local reasoning models
    if CALL_LOG:
        with open(CALL_LOG, 'a') as f:
            f.write(json.dumps({'qid': QID, 'model': name, 'role': role, 'in': usage[0], 'out': usage[1],
                                'sec': round(time.time() - t0, 3)}) + '\n')
    return text


def _mock(messages):
    """Canned replies in the formats MDAgents' parsers expect, so every code path runs without a model."""
    last = messages[-1]['content']
    h = int(hashlib.md5(''.join(m['content'] for m in messages).encode()).hexdigest(), 16)
    if 'difficulty/complexity of the medical query among' in last:
        return ['1) basic', '2) intermediate', '3) advanced'][int(hashlib.md5(last.encode()).hexdigest(), 16) % 3]
    if 'Hierarchy: Independent' in last:
        return '\n'.join(f'{i}. {r} - Specializes in {r.lower()} care. - Hierarchy: {hier}' for i, (r, hier) in enumerate(
            [('Internist', 'Independent'), ('Pathologist', 'Internist > Pathologist'), ('Pharmacologist', 'Independent'),
             ('Radiologist', 'Independent'), ('Surgeon', 'Independent')], 1))
    if 'organize' in last and 'MDTs' in last:
        return '\n\n'.join(f'Group {g} - {name}\n' + '\n'.join(
            f'Member {m}: {role}{" (Lead)" if m == 1 else ""} - Handles {role.lower()} review.'
            for m, role in enumerate(['Internist', 'Pathologist', 'Pharmacologist'], 1))
            for g, name in enumerate(['Initial Assessment Team (IAT)', 'Diagnostic Evidence Team (DET)',
                                      'Final Review and Decision Team (FRDT)'], 1))
    if 'indicate whether you want to talk' in last:
        return 'yes' if h % 3 == 0 else 'no'
    if 'Enter the number of the expert' in last:
        return '1'
    return f"Answer: ({'ABCD'[h % 4]}) mock answer"
