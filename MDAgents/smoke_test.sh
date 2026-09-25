#!/usr/bin/env bash
# Offline end-to-end check with the mock model: no API key, no GPU. Run prepare_data.py once first.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python3}
rm -f output/mock_*_smoke* output/calls/mock_*_smoke*
$PY evaluate.py selftest
for d in basic intermediate advanced adaptive; do
  $PY main.py --model mock --dataset medmcqa --difficulty $d --num_samples 6 --tag _smoke > /dev/null
done
$PY main.py --model mock --dataset medqa --num_samples 6 --tag _smoke --shard 0/2 > /dev/null
$PY main.py --model mock --dataset medqa --num_samples 6 --tag _smoke --shard 1/2 > /dev/null
$PY main.py --model mock --dataset medqa --num_samples 6 --tag _smoke --shard 1/2 > /dev/null   # resume: adds nothing
$PY main.py --model mock --dataset medmcqa_hi --num_samples 6 --tag _smoke > /dev/null
for m in zeroshot cot cotsc; do
  $PY baselines.py --model mock --dataset medmcqa --method $m --num_samples 6 --tag _smoke > /dev/null
done
$PY evaluate.py output/mock_*_smoke*.jsonl
$PY evaluate.py compare 'output/mock_medmcqa_adaptive_smoke.jsonl' 'output/mock_medmcqa_cot_smoke.jsonl'
[ "$(cat output/mock_medqa_adaptive_smoke.s*of2.jsonl | wc -l)" -eq 6 ] || { echo 'FAIL: shard/resume'; exit 1; }
if grep -l '"error": "' output/mock_*_smoke*.jsonl; then echo 'FAIL: errors in files above'; exit 1; fi
echo 'SMOKE TEST PASSED'
