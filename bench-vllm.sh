#!/usr/bin/env bash
# bench-vllm.sh — reproducible throughput + coherence gate for vLLM/XPU.
# Usage: ./bench-vllm.sh [host] [port] [model_name]
# Uses vLLM /metrics endpoint for precise timing.
# Prints one JSON line.
set -u
HOST="${1:-localhost}"
PORT="${2:-8011}"
MODEL="${3:-/model}"
URL="http://$HOST:$PORT"

# Wait for health (up to 180s)
for i in $(seq 1 60); do
  curl -sf "$URL/health" >/dev/null 2>&1 && break
  sleep 3
done
curl -sf "$URL/health" >/dev/null 2>&1 || { echo '{"error":"server not healthy"}'; exit 1; }

# Build a ~3.7k-token prompt
PARA="The Gated Delta Network is a linear-attention variant that replaces softmax with a gated delta rule update over a recurrent state. Each layer maintains a fixed-size state matrix that is updated in O(1) per token, independent of sequence length, which makes decode throughput nearly flat with context. The delta rule is h = h + g * (v - h . k) where g is a data-dependent sigmoid gate, k a key, and v a value. Aggressive negative g values cause rapid forgetting of old state, keeping the state bounded. Full attention is used every fourth layer to recover exact long-range tokens that the linear path would otherwise forget."
PROMPT=""
for i in $(seq 1 28); do PROMPT="$PROMPT$PARA "; done

# Get a prometheus metric value
get_metric() {
  curl -s "$URL/metrics" | grep "^$1" | tail -1 | awk '{print $NF}'
}

# --- All benchmarking done in Python to avoid bash float/quoting issues ---
python3 << PYEOF
import json, urllib.request, time, sys, re

HOST = "$HOST"
PORT = "$PORT"
MODEL = "$MODEL"
URL = f"http://{HOST}:{PORT}"
PROMPT = """$PROMPT"""

def get_metric(name):
    """Get the last value of a prometheus metric (handles label suffix)."""
    try:
        resp = urllib.request.urlopen(f"{URL}/metrics")
        val = 0.0
        for line in resp.read().decode().split("\n"):
            if line.startswith(name) and not line.startswith("#"):
                val = float(line.split()[-1])
        return val
    except:
        pass
    return 0.0

def get_metric_count(name):
    """Get the count value of a prometheus metric (last matching line)."""
    try:
        resp = urllib.request.urlopen(f"{URL}/metrics")
        lines = resp.read().decode().split("\n")
        val = 0
        for line in lines:
            if line.startswith(name) and not line.startswith("#"):
                val = int(float(line.split()[-1]))
        return val
    except:
        pass
    return 0

def post_chat(messages, max_tokens=256, temp=0):
    payload = json.dumps({
        "model": MODEL, "max_tokens": max_tokens, "temperature": temp,
        "stream": False, "messages": messages
    }).encode()
    req = urllib.request.Request(
        f"{URL}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"}
    )
    return json.loads(urllib.request.urlopen(req).read())

# Baseline metrics
base_tpot_sum = get_metric("vllm:inter_token_latency_seconds_sum")
base_tpot_count = get_metric_count("vllm:inter_token_latency_seconds_count")
base_ttft_sum = get_metric("vllm:time_to_first_token_seconds_sum")
base_ttft_count = get_metric_count("vllm:time_to_first_token_seconds_count")

# 3 hard-task runs
hard_prompt = f"Answer in one sentence: what is the mechanism described in the following text?\n\n{PROMPT}"
last_resp = None
for i in range(3):
    last_resp = post_chat([{"role": "user", "content": hard_prompt}], max_tokens=256)

# Post-hard metrics
new_tpot_sum = get_metric("vllm:inter_token_latency_seconds_sum")
new_tpot_count = get_metric_count("vllm:inter_token_latency_seconds_count")
new_ttft_sum = get_metric("vllm:time_to_first_token_seconds_sum")
new_ttft_count = get_metric_count("vllm:time_to_first_token_seconds_count")

tpot_delta = new_tpot_count - base_tpot_count
ttft_delta = new_ttft_count - base_ttft_count

if tpot_delta > 0:
    avg_tpot = (new_tpot_sum - base_tpot_sum) / tpot_delta
    decode_tps = 1.0 / avg_tpot if avg_tpot > 0 else 0
else:
    decode_tps = 0

n_in = last_resp.get("usage", {}).get("prompt_tokens", 0)
n_out = last_resp.get("usage", {}).get("completion_tokens", 0)

if ttft_delta > 0 and n_in > 0:
    avg_ttft = (new_ttft_sum - base_ttft_sum) / ttft_delta
    prefill_tps = n_in / avg_ttft if avg_ttft > 0 else 0
else:
    prefill_tps = 0

# Easy task: count 1 to 100 (2 runs)
base_tpot2_sum = get_metric("vllm:inter_token_latency_seconds_sum")
base_tpot2_count = get_metric_count("vllm:inter_token_latency_seconds_count")
easy_resp = None
for i in range(2):
    easy_resp = post_chat([{"role": "user", "content": "Count from 1 to 100, one number per line."}], max_tokens=256)
new_tpot2_sum = get_metric("vllm:inter_token_latency_seconds_sum")
new_tpot2_count = get_metric_count("vllm:inter_token_latency_seconds_count")
tpot2_delta = new_tpot2_count - base_tpot2_count
if tpot2_delta > 0:
    avg_tpot2 = (new_tpot2_sum - base_tpot2_sum) / tpot2_delta
    easy_tps = 1.0 / avg_tpot2 if avg_tpot2 > 0 else 0
else:
    easy_tps = 0
easy_out = easy_resp.get("usage", {}).get("completion_tokens", 0) if easy_resp else 0

# Coherence gate
a1_resp = post_chat([{"role": "user", "content": "What is 17 times 3? Answer with just the number, no explanation."}], max_tokens=512)
a2_resp = post_chat([{"role": "user", "content": "Name the capital of France. One word."}], max_tokens=512)
a1 = a1_resp["choices"][0]["message"]["content"]
a2 = a2_resp["choices"][0]["message"]["content"]

# Garbage detection
combined = a1 + a2
garbage = 1 if re.search(r'!{3,}', combined) or re.search(r'(.)\1{19,}', combined) else 0
c1 = 1 if re.search(r'51', a1, re.IGNORECASE) else 0
c2 = 1 if re.search(r'paris', a2, re.IGNORECASE) else 0
coherent = 1 if (garbage == 0 and c1 == 1 and c2 == 1) else 0

result = {
    "decode_tok_s": round(decode_tps, 2),
    "prefill_tok_s": round(prefill_tps, 2),
    "coherent": coherent,
    "answers": {"math": a1.strip()[:200], "capital": a2.strip()[:200]},
    "tokens": {"in": n_in, "out": n_out},
    "easy": {"decode_tok_s": round(easy_tps, 2), "out": easy_out},
}
print(json.dumps(result))
PYEOF
