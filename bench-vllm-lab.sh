#!/usr/bin/env bash
# bench-vllm-lab.sh — benchmark vLLM lab image with MTP
# Usage: bench-vllm-lab.sh [host] [port] [model_name]
set -u
HOST="${1:-localhost}"
PORT="${2:-8015}"
MODEL="${3:-/model}"
URL="http://$HOST:$PORT"

# Wait for health
for i in $(seq 1 60); do
  curl -sf "$URL/health" >/dev/null 2>&1 && break
  sleep 3
done
curl -sf "$URL/health" >/dev/null 2>&1 || { echo '{"error":"server not healthy"}'; exit 1; }

# Build a ~2.5k-token prompt by repeating a dense paragraph
PARA="The Gated Delta Network is a linear-attention variant that replaces softmax with a gated delta rule update over a recurrent state. Each layer maintains a fixed-size state matrix that is updated in O(1) per token, independent of sequence length, which makes decode throughput nearly flat with context. The delta rule is h = h + g * (v - h . k) where g is a data-dependent sigmoid gate, k a key, and v a value. Aggressive negative g values cause rapid forgetting of old state, keeping the state bounded. Full attention is used every fourth layer to recover exact long-range tokens that the linear path would otherwise forget."
PROMPT=""
for i in $(seq 1 28); do PROMPT="$PROMPT$PARA "; done

# Warmup
curl -s "$URL/v1/chat/completions" -H "Content-Type: application/json" \
  -d "$(jq -nc --arg p "$PROMPT" --arg m "$MODEL" '{model:$m,max_tokens:32,temperature:0,messages:[{role:"user",content:("Answer in one sentence: what is the mechanism described in the following text?\n\n"+$p)}]}')" >/dev/null 2>&1

# 3 throughput runs (hard task)
HARD_TPS=()
for run in 1 2 3; do
  START=$(python3 -c "import time; print(int(time.time()*1000))")
  RESP=$(curl -s "$URL/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "$(jq -nc --arg p "$PROMPT" --arg m "$MODEL" '{model:$m,max_tokens:256,temperature:0,messages:[{role:"user",content:("Answer in one sentence: what is the mechanism described in the following text?\n\n"+$p)}]}')")
  END=$(python3 -c "import time; print(int(time.time()*1000))")
  N_OUT=$(echo "$RESP" | jq -r '.usage.completion_tokens // 0')
  ELAPSED_MS=$((END - START))
  TPS=$(python3 -c "print(round($N_OUT * 1000.0 / max($ELAPSED_MS,1), 1))")
  HARD_TPS+=("$TPS")
  echo "Hard run $run: $N_OUT tokens in ${ELAPSED_MS}ms = $TPS tok/s" >&2
done

HARD_MEDIAN=$(printf '%s\n' "${HARD_TPS[@]}" | sort -n | sed -n 2p)

# Easy-task decode (high MTP acceptance): count from 1 to 100
EASY_TPS=()
for run in 1 2; do
  START=$(python3 -c "import time; print(int(time.time()*1000))")
  EASY_RESP=$(curl -s "$URL/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "$(jq -nc --arg m "$MODEL" '{model:$m,max_tokens:256,temperature:0,messages:[{role:"user",content:"Count from 1 to 100, one number per line."}]}')")
  END=$(python3 -c "import time; print(int(time.time()*1000))")
  N_OUT=$(echo "$EASY_RESP" | jq -r '.usage.completion_tokens // 0')
  ELAPSED_MS=$((END - START))
  ETPS=$(python3 -c "print(round($N_OUT * 1000.0 / max($ELAPSED_MS,1), 1))")
  EASY_TPS+=("$ETPS")
  echo "Easy run $run: $N_OUT tokens in ${ELAPSED_MS}ms = $ETPS tok/s" >&2
done
EASY_MEDIAN=$(printf '%s\n' "${EASY_TPS[@]}" | sort -n | sed -n 1p)

# Coherence gate
A1=$(curl -s "$URL/v1/chat/completions" -H "Content-Type: application/json" \
  -d "$(jq -nc --arg m "$MODEL" '{model:$m,max_tokens:16,temperature:0,messages:[{role:"user",content:"What is 17 times 3? Answer with just the number."}]}')" \
  | jq -r '.choices[0].message.content')
A2=$(curl -s "$URL/v1/chat/completions" -H "Content-Type: application/json" \
  -d "$(jq -nc --arg m "$MODEL" '{model:$m,max_tokens:16,temperature:0,messages:[{role:"user",content:"Name the capital of France. One word."}]}')" \
  | jq -r '.choices[0].message.content')

GARBAGE=0
printf '%s%s' "$A1" "$A2" | grep -qE '(!{3,}|(.){20}\2*)' && GARBAGE=1
echo "$A1" | grep -qi '51' && C1=1 || C1=0
echo "$A2" | grep -qi 'paris' && C2=1 || C2=0
COHERENT=$(( GARBAGE==0 && C1==1 && C2==1 ))

# Spec metrics
DRAFT_TOTAL=$(curl -s "$URL/metrics" | grep "spec_decode_num_draft_tokens_total" | grep -v "HELP\|TYPE\|created" | awk -F' ' '{print $NF}')
ACCEPTED_TOTAL=$(curl -s "$URL/metrics" | grep "spec_decode_num_accepted_tokens_total" | grep -v "HELP\|TYPE\|created" | awk -F' ' '{print $NF}')
DRAFTS_TOTAL=$(curl -s "$URL/metrics" | grep "spec_decode_num_drafts_total" | grep -v "HELP\|TYPE\|created" | awk -F' ' '{print $NF}')

echo ""
echo "=== RESULTS ==="
echo "Hard median: $HARD_MEDIAN tok/s"
echo "Easy median: $EASY_MEDIAN tok/s"
echo "Coherent: $COHERENT (math=$A1, capital=$A2)"
echo "Spec: drafts=$DRAFTS_TOTAL, draft_tokens=$DRAFT_TOTAL, accepted=$ACCEPTED_TOTAL"
if [ "$DRAFTS_TOTAL" -gt 0 ] 2>/dev/null; then
  echo "Acceptance: $(python3 -c "print(round($ACCEPTED_TOTAL/$DRAFTS_TOTAL, 2))") tokens/draft, $(python3 -c "print(round(100*$ACCEPTED_TOTAL/$DRAFT_TOTAL, 1))")% of draft tokens"
fi
