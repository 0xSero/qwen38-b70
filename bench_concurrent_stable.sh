#!/usr/bin/env bash
# Stable concurrent benchmark with warmup and shorter prompts
set -u
URL="http://localhost:8020"
CONCURRENCY="${1:-8}"
ROUNDS="${2:-5}"

for i in $(seq 1 60); do
  curl -sf "$URL/health" >/dev/null 2>&1 && break
  sleep 3
done
curl -sf "$URL/health" >/dev/null 2>&1 || { echo '{"error":"server not healthy"}'; exit 1; }

PARA="The Gated Delta Network is a linear-attention variant that replaces softmax with a gated delta rule update over a recurrent state. Each layer maintains a fixed-size state matrix that is updated in O(1) per token, independent of sequence length, which makes decode throughput nearly flat with context. The delta rule is h = h + g * (v - h . k) where g is a data-dependent sigmoid gate, k a key, and v a value. Aggressive negative g values cause rapid forgetting of old state, keeping the state bounded. Full attention is used every fourth layer to recover exact long-range tokens that the linear path would otherwise forget."
PROMPT=""
for i in $(seq 1 3); do PROMPT="$PROMPT$PARA "; done

build_payload() {
  python3 -c "
import json,sys
p=sys.argv[1]
mt=int(sys.argv[2])
print(json.dumps({'model':'/model','max_tokens':mt,'temperature':0,'top_p':1,'messages':[{'role':'user','content':'Answer in one sentence: what is the mechanism described in the following text?\n\n'+p}]}))
" "$1" "$2"
}

PAYLOAD=$(build_payload "$PROMPT" 256)

# Warmup round (discard)
echo "Warming up..." >&2
for i in $(seq 1 "$CONCURRENCY"); do
  curl -s "$URL/v1/chat/completions" -H "Content-Type: application/json" -d "$PAYLOAD" -o "/tmp/warmup_$i.json" &
done
wait

# Benchmark rounds
AGG_TPS=()
for round in $(seq 1 "$ROUNDS"); do
  T0=$(python3 -c "import time; print(time.time())")
  TOTAL_TOKENS=0

  for i in $(seq 1 "$CONCURRENCY"); do
    curl -s "$URL/v1/chat/completions" -H "Content-Type: application/json" -d "$PAYLOAD" -o "/tmp/resp_$i.json" &
  done
  wait

  T1=$(python3 -c "import time; print(time.time())")
  for i in $(seq 1 "$CONCURRENCY"); do
    N=$(jq -r '.usage.completion_tokens // 0' "/tmp/resp_$i.json")
    TOTAL_TOKENS=$((TOTAL_TOKENS + N))
  done

  WALL=$(python3 -c "print($T1-$T0)")
  TPS=$(python3 -c "print(round($TOTAL_TOKENS/$WALL, 1))")
  AGG_TPS+=("$TPS")
  echo "  round $round: ${TPS} tok/s aggregate, ${TOTAL_TOKENS} tokens in ${WALL}s" >&2
done

# Median
AGG_MEDIAN=$(printf '%s\n' "${AGG_TPS[@]}" | sort -n | sed -n $(((${#AGG_TPS[@]}+1)/2))p)

# Coherence
A1=$(curl -s "$URL/v1/chat/completions" -H "Content-Type: application/json" \
  -d '{"model":"/model","max_tokens":64,"temperature":0,"top_p":1,"messages":[{"role":"user","content":"What is 17 times 3? Answer with just the number."}]}' \
  | jq -r '.choices[0].message.content')
A2=$(curl -s "$URL/v1/chat/completions" -H "Content-Type: application/json" \
  -d '{"model":"/model","max_tokens":64,"temperature":0,"top_p":1,"messages":[{"role":"user","content":"Name the capital of France. One word."}]}' \
  | jq -r '.choices[0].message.content')

echo "$A1" | grep -qi '51' && C1=1 || C1=0
echo "$A2" | grep -qi 'paris' && C2=1 || C2=0
COHERENT=$(( C1==1 && C2==1 ))

jq -nc --argjson agg_tps "$AGG_MEDIAN" --argjson concurrent "$CONCURRENCY" \
  --argjson coherent "$COHERENT" --arg a1 "$A1" --arg a2 "$A2" \
  --argjson all_runs "$(printf '%s\n' "${AGG_TPS[@]}" | jq -cs '.')" \
  '{aggregate_tok_s:$agg_tps, concurrency:$concurrent, coherent:$coherent, answers:{math:$a1, capital:$a2}, all_runs:$all_runs}'
