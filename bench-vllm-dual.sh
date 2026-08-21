#!/bin/bash
# bench-vllm-dual.sh — Benchmark aggregate throughput on 2× B70 dual-instance setup
#
# Fires concurrent requests to both vLLM instances and measures aggregate tok/s.
# Usage: ./bench-vllm-dual.sh
set -euo pipefail

PORT0=8020
PORT1=8021
MODEL="/model"

echo "=== Warming up both instances ==="
curl -s http://localhost:$PORT0/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"'"$MODEL"'","prompt":"Hello","max_tokens":5,"temperature":0}' > /dev/null 2>&1 &
curl -s http://localhost:$PORT1/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"'"$MODEL"'","prompt":"Hello","max_tokens":5,"temperature":0}' > /dev/null 2>&1 &
wait
sleep 1

echo "=== Concurrent benchmark (3 runs, 512 tokens each) ==="
for run in 1 2 3; do
  python3 -c "
import time, json, threading, urllib.request

results = {}

def bench(port, name):
    data = json.dumps({
        'model': '$MODEL',
        'prompt': 'Count from 1 to 100. Just numbers separated by commas.',
        'max_tokens': 512,
        'temperature': 0,
        'ignore_eos': True
    }).encode()
    req = urllib.request.Request(f'http://localhost:{port}/v1/completions',
                                 data=data,
                                 headers={'Content-Type': 'application/json'})
    start = time.time()
    resp = urllib.request.urlopen(req)
    elapsed = time.time() - start
    d = json.loads(resp.read())
    toks = d['usage']['completion_tokens']
    results[name] = (toks, elapsed, toks / elapsed)

t0 = threading.Thread(target=bench, args=($PORT0, 'gpu0'))
t1 = threading.Thread(target=bench, args=($PORT1, 'gpu1'))
t0.start()
t1.start()
t0.join()
t1.join()

total_toks = results['gpu0'][0] + results['gpu1'][0]
max_elapsed = max(results['gpu0'][1], results['gpu1'][1])
agg_tps = total_toks / max_elapsed

print(f'Run $run:')
print(f'  GPU0: {results[\"gpu0\"][0]} tok in {results[\"gpu0\"][1]*1000:.0f}ms = {results[\"gpu0\"][2]:.1f} tok/s')
print(f'  GPU1: {results[\"gpu1\"][0]} tok in {results[\"gpu1\"][1]*1000:.0f}ms = {results[\"gpu1\"][2]:.1f} tok/s')
print(f'  AGGREGATE: {total_toks} tok / {max_elapsed*1000:.0f}ms = {agg_tps:.1f} tok/s')
" 2>&1
done

echo ""
echo "=== Coherence check ==="
ANS0=$(curl -s http://localhost:$PORT0/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"'"$MODEL"'","prompt":"What is 17*3? Answer with just the number.","max_tokens":64,"temperature":0}' \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['text'].strip())")
ANS1=$(curl -s http://localhost:$PORT1/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"'"$MODEL"'","prompt":"What is 17*3? Answer with just the number.","max_tokens":64,"temperature":0}' \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['text'].strip())")
echo "GPU0: 17*3 = $ANS0"
echo "GPU1: 17*3 = $ANS1"

echo ""
echo "=== Spec metrics ==="
echo "GPU0:"
docker logs vllm-dual-gpu0 2>&1 | grep -i "accept" | tail -2
echo "GPU1:"
docker logs vllm-dual-gpu1 2>&1 | grep -i "accept" | tail -2
