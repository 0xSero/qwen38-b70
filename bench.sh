#!/usr/bin/env bash
# bench.sh — reproducible throughput + coherence gate for qwen38-b70.
# Usage: ./bench.sh [host]   (default host: localhost, run ON the host)
# Runs 3 throughput measurements and reports median for stability.
# Prints one JSON line: {"decode_tok_s":..,"prefill_tok_s":..,"coherent":..,...}
set -u
HOST="${1:-localhost}"
PORT=8010
URL="http://$HOST:$PORT"

# Wait for health (up to 180s)
for i in $(seq 1 60); do
  curl -sf "$URL/health" >/dev/null 2>&1 && break
  sleep 3
done
curl -sf "$URL/health" >/dev/null 2>&1 || { echo '{"error":"server not healthy"}'; exit 1; }

# Build a ~2.5k-token prompt by repeating a dense paragraph.
PARA="The Gated Delta Network is a linear-attention variant that replaces softmax with a gated delta rule update over a recurrent state. Each layer maintains a fixed-size state matrix that is updated in O(1) per token, independent of sequence length, which makes decode throughput nearly flat with context. The delta rule is h = h + g * (v - h . k) where g is a data-dependent sigmoid gate, k a key, and v a value. Aggressive negative g values cause rapid forgetting of old state, keeping the state bounded. Full attention is used every fourth layer to recover exact long-range tokens that the linear path would otherwise forget."
PROMPT=""
for i in $(seq 1 28); do PROMPT="$PROMPT$PARA "; done

# 3 throughput runs (hard task), collect decode tok/s
HARD_TPS=()
for run in 1 2 3; do
  RESP=$(curl -s "$URL/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "$(jq -nc --arg p "$PROMPT" '{model:"qwen3.8-27b",max_tokens:256,temperature:0,messages:[{role:"user",content:("Answer in one sentence: what is the mechanism described in the following text?\n\n"+$p)}]}')")
  TPS=$(echo "$RESP" | jq -r '.timings.predicted_per_second // 0')
  HARD_TPS+=("$TPS")
done

# Median of 3 (sort, pick middle)
HARD_MEDIAN=$(printf '%s\n' "${HARD_TPS[@]}" | sort -n | sed -n 2p)

# Prefill from last run
PREFILL_TPS=$(echo "$RESP" | jq -r '.timings.prompt_per_second // 0')
N_OUT=$(echo "$RESP" | jq -r '.usage.completion_tokens // 0')
N_IN=$(echo "$RESP" | jq -r '.usage.prompt_tokens // 0')

# Easy-task decode (high MTP acceptance): count from 1 to 100. 2 runs, median.
EASY_TPS=()
for run in 1 2; do
  EASY_RESP=$(curl -s "$URL/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"model":"qwen3.8-27b","max_tokens":256,"temperature":0,"messages":[{"role":"user","content":"Count from 1 to 100, one number per line."}]}')
  ETPS=$(echo "$EASY_RESP" | jq -r '.timings.predicted_per_second // 0')
  EASY_TPS+=("$ETPS")
done
EASY_MEDIAN=$(printf '%s\n' "${EASY_TPS[@]}" | sort -n | sed -n 1p)

# Coherence gate: two factual questions.
A1=$(curl -s "$URL/v1/chat/completions" -H "Content-Type: application/json" \
  -d '{"model":"qwen3.8-27b","max_tokens":16,"temperature":0,"messages":[{"role":"user","content":"What is 17 times 3? Answer with just the number."}]}' \
  | jq -r '.choices[0].message.content')
A2=$(curl -s "$URL/v1/chat/completions" -H "Content-Type: application/json" \
  -d '{"model":"qwen3.8-27b","max_tokens":16,"temperature":0,"messages":[{"role":"user","content":"Name the capital of France. One word."}]}' \
  | jq -r '.choices[0].message.content')

# Garbage detection: !!! runs (3+ bangs) or 20+ repeated same char.
GARBAGE=0
printf '%s%s' "$A1" "$A2" | grep -qE '(!{3,}|(.){20}\2*)' && GARBAGE=1
echo "$A1" | grep -qi '51' && C1=1 || C1=0
echo "$A2" | grep -qi 'paris' && C2=1 || C2=0
COHERENT=$(( GARBAGE==0 && C1==1 && C2==1 ))

jq -nc --argjson decode "$HARD_MEDIAN" --argjson prefill "$PREFILL_TPS" \
  --argjson coherent "$COHERENT" --arg a1 "$A1" --arg a2 "$A2" \
  --argjson n_in "$N_IN" --argjson n_out "$N_OUT" \
  --argjson easy_tps "$EASY_MEDIAN" \
  --argjson hard_all "$(printf '%s\n' "${HARD_TPS[@]}" | jq -cs '.')" \
  '{decode_tok_s:$decode, prefill_tok_s:$prefill, coherent:$coherent, answers:{math:$a1, capital:$a2}, tokens:{in:$n_in,out:$n_out}, easy:{decode_tok_s:$easy_tps}, hard_runs:$hard_all}'
