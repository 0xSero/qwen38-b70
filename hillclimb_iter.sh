#!/bin/bash
# hillclimb_iter.sh — ONE hill-climb iteration per invocation.
# Reads knob queue, picks next untried knob, relaunches container,
# benchmarks (single-stream + concurrent), checks coherence, records in HILLCLIMB.md.
#
# Key design: persistent compile cache volume + retry loop.
# First attempt compiles from scratch (may crash during cudagraph capture due to
# memory pressure from compilation intermediates). Compiled graphs are saved to the
# persistent volume. Second attempt loads cached compilation (no memory overhead),
# leaving headroom for cudagraph capture to succeed.
set -euo pipefail

REPO="/home/sero/qwen38-b70"
QUEUE="$REPO/knob_queue.txt"
HILL="$REPO/HILLCLIMB.md"
CONTAINER="vllm-tp2-pw"
IMAGE="intel/llm-scaler-vllm:0.21.0-b3"
MODEL="/home/sero/models/Qwen3.8-27B-int4-AutoRound"
ENTRY="/tmp/lab_tp2_pw_v11.sh"
GLOO_KERNELS="/tmp/vllm_gloo_kernels.py"
LOG="$REPO/hillclimb_automation.log"
LOCK="$REPO/hillclimb.lock"
COMPILE_CACHE="$REPO/compile_cache"

# --- Prevent concurrent runs (cron + manual can collide) ---
if [ -f "$LOCK" ]; then
  LOCK_AGE=$(( $(date +%s) - $(stat -c %Y "$LOCK" 2>/dev/null || echo 0) ))
  if [ "$LOCK_AGE" -lt 900 ]; then
    echo "[SKIP] Another iteration is running (lock age ${LOCK_AGE}s). Exiting."
    exit 0
  fi
  echo "[WARN] Stale lock found (${LOCK_AGE}s old), removing and proceeding."
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT
NOW="$(date '+%Y-%m-%d %H:%M')"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

# ---------------------------------------------------------------------------
# 1. Read next knob from queue
# ---------------------------------------------------------------------------
KNOB_LINE=""
while IFS= read -r line; do
  case "$line" in
    \#*|"") continue ;;
  esac
  # Check if this knob already has a result marker
  if echo "$line" | grep -qE '\|(DONE|FAIL|SKIP)\|'; then
    continue
  fi
  KNOB_LINE="$line"
  break
done < "$QUEUE"

if [ -z "$KNOB_LINE" ]; then
  log "All knobs exhausted. Writing summary."
  echo "" >> "$HILL"
  echo "## Automation summary ($(date))" >> "$HILL"
  echo "All knob queue entries exhausted. Automation stopped." >> "$HILL"
  exit 0
fi

# Parse: KNOB_NAME|MTP|MEM_UTIL|MAX_NUM_SEQS|MAX_BATCHED_TOKENS|MAX_MODEL_LEN|CONCURRENCY
IFS='|' read -r KNOB_NAME MTP MEM_UTIL MAX_SEQS MAX_BATCHED MAX_MODEL_LEN CONCURRENCY <<< "$KNOB_LINE"
log "Iteration: KNOB=$KNOB_NAME MTP=$MTP MEM=$MEM_UTIL SEQS=$MAX_SEQS BATCHED=$MAX_BATCHED MODELLEN=$MAX_MODEL_LEN CONC=$CONCURRENCY"

# ---------------------------------------------------------------------------
# 2. Generate entrypoint with this knob's parameters
# ---------------------------------------------------------------------------
ENTRYPOINT="$REPO/entrypoints/ep_${KNOB_NAME}.sh"
mkdir -p "$REPO/entrypoints" "$COMPILE_CACHE"

# Start from the base v11 entrypoint and replace the vllm serve line.
# Filter out the `rm -rf torch_compile_cache` line so the persistent cache survives.
BASE_EP="$ENTRY"
if [ ! -f "$BASE_EP" ]; then
  log "ERROR: base entrypoint $BASE_EP not found"
  exit 1
fi

awk '/^MTP_CONFIG=/{exit} {print}' "$BASE_EP" | grep -v 'rm -rf.*torch_compile_cache' > "$ENTRYPOINT"
cat >> "$ENTRYPOINT" << EOSERVE

export VLLM_XPU_ENABLE_XPU_GRAPH=1
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1200
export VLLM_LOGGING_LEVEL=INFO

MTP_CONFIG='{"method":"qwen3_5_mtp","num_speculative_tokens":${MTP},"model":"/model"}'

exec vllm serve /model \\
  --dtype float16 --max-model-len ${MAX_MODEL_LEN} --gpu-memory-utilization ${MEM_UTIL} \\
  --tensor-parallel-size 2 --max-num-seqs ${MAX_SEQS} --max-num-batched-tokens ${MAX_BATCHED} \\
  --trust-remote-code --port 8000 --host 0.0.0.0 \\
  --skip-mm-profiling \\
  --distributed-timeout-seconds 600 \\
  --speculative-config "\$MTP_CONFIG"
EOSERVE
chmod +x "$ENTRYPOINT"
log "Entrypoint written to $ENTRYPOINT"

# ---------------------------------------------------------------------------
# 3. Start container with retry loop (compile cache warmup strategy)
#
# Attempt 1: Fresh compilation. May crash during cudagraph capture because
#   compilation intermediates consume GPU memory. Compiled graphs are saved
#   to the persistent volume before the crash.
# Attempt 2: Loads cached compilation (skips compile step, frees GPU memory).
#   Cudagraph capture should succeed with the extra headroom.
# ---------------------------------------------------------------------------
PARA="The Gated Delta Network is a linear-attention variant that replaces softmax with a gated delta rule update over a recurrent state. Each layer maintains a fixed-size state matrix that is updated in O(1) per token, independent of sequence length, which makes decode throughput nearly flat with context. The delta rule is h = h + g * (v - h . k) where g is a data-dependent sigmoid gate, k a key, and v a value. Aggressive negative g values cause rapid forgetting of old state, keeping the state bounded. Full attention is used every fourth layer to recover exact long-range tokens that the linear path would otherwise forget."
HARD_PROMPT=""
for i in $(seq 1 3); do HARD_PROMPT="$HARD_PROMPT$PARA "; done

build_payload() {
  python3 -c "
import json,sys
p=sys.argv[1]; mt=int(sys.argv[2])
print(json.dumps({'model':'/model','max_tokens':mt,'temperature':0,'top_p':1,'messages':[{'role':'user','content':p}]}))
" "$1" "$2"
}

START_OK=0
for ATTEMPT in 1 2; do
  log "=== Container start attempt $ATTEMPT/2 ==="
  docker stop "$CONTAINER" 2>/dev/null || true
  docker rm "$CONTAINER" 2>/dev/null || true

  log "Starting container with knob $KNOB_NAME (attempt $ATTEMPT)..."
  docker run -d --name "$CONTAINER" --privileged \
    --device /dev/dri:/dev/dri --device /dev/dri/by-path:/dev/dri/by-path \
    -v "$MODEL:/model:ro" \
    -v "$ENTRYPOINT:/entrypoint.sh:ro" \
    -v "$GLOO_KERNELS:/tmp/vllm_gloo_kernels.py:ro" \
    -v "$COMPILE_CACHE:/root/.cache/vllm/torch_compile_cache" \
    -e VLLM_XPU_ENABLE_XPU_GRAPH=1 -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1200 \
    -e CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0 -e UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 \
    -e VLLM_WORKER_MULTIPROC_METHOD=spawn -e ZES_ENABLE_SYSMAN=1 \
    -e CCL_ZE_IPC_EXCHANGE=sockets -e VLLM_LOGGING_LEVEL=INFO \
    --entrypoint /bin/bash "$IMAGE" /entrypoint.sh

  sleep 2
  CONTAINER_IP="$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' "$CONTAINER" 2>/dev/null)"
  CURL_URL="http://${CONTAINER_IP:-172.17.0.2}:8000"
  log "Container IP: $CONTAINER_IP, health URL: $CURL_URL/health (attempt $ATTEMPT)"

  # Wait for health (up to 10 min — compilation can take 3-5 min)
  log "Waiting for server health (attempt $ATTEMPT)..."
  HEALTHY=0
  DEAD=0
  for i in $(seq 1 200); do
    if curl -sf "$CURL_URL/health" >/dev/null 2>&1; then
      HEALTHY=1
      break
    fi
    if [ "$i" -gt 20 ]; then
      if [ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null)" != "true" ]; then
        DEAD=1
        log "Container died after $((i*3))s (attempt $ATTEMPT)"
        break
      fi
    fi
    sleep 3
  done

  if [ "$HEALTHY" -ne 1 ]; then
    if [ "$DEAD" -eq 1 ]; then
      log "Container crashed (attempt $ATTEMPT). Last 15 lines:"
    else
      log "Server not healthy after 10 min (attempt $ATTEMPT). Last 15 lines:"
    fi
    docker logs --tail 15 "$CONTAINER" 2>&1 | tee -a "$LOG" || true
    if [ "$ATTEMPT" -eq 1 ]; then
      log "Compile cache should be saved from attempt 1. Retrying with cached compilation..."
      continue
    fi
    # Both attempts failed
    log "FAIL: container did not become healthy after 2 attempts."
    docker logs --tail 30 "$CONTAINER" 2>&1 | tee -a "$LOG" || true
    python3 -c "
import sys
knob = sys.argv[1]; tag = sys.argv[2]; qfile = sys.argv[3]
with open(qfile) as f: lines = f.readlines()
out = [l.rstrip('\n') + '|' + tag + '\n' if l.rstrip('\n') == knob else l for l in lines]
with open(qfile, 'w') as f: f.writelines(out)
" "$KNOB_LINE" "FAIL|$(date '+%H:%M:%S')|server_not_healthy" "$QUEUE"
    LAST_ITER=$(grep -oP '^\| \K[0-9]+' "$HILL" | sort -n | tail -1)
    NEXT_ITER=$((LAST_ITER + 1))
    echo "| $NEXT_ITER | $NOW | $KNOB_NAME | CRASH | — | FAIL — container died or timeout after 2 attempts (compile cache warmup). Config: MTP${MTP}, mem=${MEM_UTIL}, seqs=${MAX_SEQS}, batched=${MAX_BATCHED}, modellen=${MAX_MODEL_LEN}, conc=${CONCURRENCY}. See hillclimb_automation.log." >> "$HILL"
    cd "$REPO"
    git add HILLCLIMB.md knob_queue.txt hillclimb_automation.log 2>/dev/null || true
    git commit -m "hillclimb: $KNOB_NAME — CRASH (container died after retry)" 2>/dev/null || true
    exit 1
  fi
  log "Server healthy! (attempt $ATTEMPT)"

  # Warmup + engine health check
  URL="$CURL_URL"
  WARMUP_RESP=$(curl -s --max-time 60 "$URL/v1/chat/completions" -H "Content-Type: application/json" \
    -d "$(build_payload "$HARD_PROMPT" 32)")
  if echo "$WARMUP_RESP" | jq -e '.choices[0].message.content' >/dev/null 2>&1; then
    log "Warmup OK, engine responsive (attempt $ATTEMPT)"
    START_OK=1
    break
  fi

  log "Warmup FAILED (attempt $ATTEMPT): $(echo "$WARMUP_RESP" | head -c 200)"
  docker logs --tail 15 "$CONTAINER" 2>&1 | tee -a "$LOG" || true
  if [ "$ATTEMPT" -eq 1 ]; then
    log "Engine died after warmup on attempt 1. Compile cache saved. Retrying..."
    continue
  fi
  # Both attempts' warmup failed
  log "FAIL: engine dead after warmup on both attempts."
  python3 -c "
import sys
knob = sys.argv[1]; tag = sys.argv[2]; qfile = sys.argv[3]
with open(qfile) as f: lines = f.readlines()
out = [l.rstrip('\n') + '|' + tag + '\n' if l.rstrip('\n') == knob else l for l in lines]
with open(qfile, 'w') as f: f.writelines(out)
" "$KNOB_LINE" "FAIL|$(date '+%H:%M:%S')|engine_dead_after_warmup" "$QUEUE"
  LAST_ITER=$(grep -oP '^\| \K[0-9]+' "$HILL" | sort -n | tail -1)
  NEXT_ITER=$((LAST_ITER + 1))
  echo "| $NEXT_ITER | $NOW | $KNOB_NAME | CRASH | — | FAIL — engine died after warmup on both attempts. Config: MTP${MTP}, mem=${MEM_UTIL}, seqs=${MAX_SEQS}, conc=${CONCURRENCY}." >> "$HILL"
  cd "$REPO"
  git add HILLCLIMB.md knob_queue.txt hillclimb_automation.log 2>/dev/null || true
  git commit -m "hillclimb: $KNOB_NAME — CRASH (engine dead after warmup, retry)" 2>/dev/null || true
  exit 1
done

sleep 2
URL="$CURL_URL"
log "Benchmarking at $URL ..."

# Easy prompt (short, counting task)
EASY_PROMPT="Count from 1 to 50. Put each number on its own line."

# Helper: check if engine is still alive by looking for error in response
check_engine() {
  local resp="$1"
  if echo "$resp" | jq -e '.error' >/dev/null 2>&1; then
    log "Engine error detected: $(echo "$resp" | jq -r '.error.message // .error' | head -c 100)"
    return 1
  fi
  if [ -z "$resp" ] || [ "$resp" = "" ]; then
    log "Empty response — engine likely dead"
    return 1
  fi
  return 0
}

# ---------------------------------------------------------------------------
# 4. Benchmark — single-stream hard + easy, then concurrent
# ---------------------------------------------------------------------------

# --- Single-stream hard (3 runs) ---
HARD_TPS=()
for run in 1 2 3; do
  START_MS=$(python3 -c "import time; print(int(time.time()*1000))")
  RESP=$(curl -s --max-time 120 "$URL/v1/chat/completions" -H "Content-Type: application/json" \
    -d "$(build_payload "$HARD_PROMPT" 256)")
  END_MS=$(python3 -c "import time; print(int(time.time()*1000))")
  N_OUT=$(echo "$RESP" | jq -r '.usage.completion_tokens // 0')
  ELAPSED_MS=$((END_MS - START_MS))
  TPS=$(python3 -c "print(round($N_OUT * 1000.0 / max($ELAPSED_MS,1), 1))")
  HARD_TPS+=("$TPS")
  log "  hard run $run: $N_OUT tok in ${ELAPSED_MS}ms = $TPS tok/s"
  if ! check_engine "$RESP"; then
    log "Engine died during hard benchmark — aborting"
    break
  fi
done
HARD_MEDIAN=$(printf '%s\n' "${HARD_TPS[@]}" | sort -n | sed -n 2p)

# --- Single-stream easy (3 runs) ---
EASY_TPS=()
for run in 1 2 3; do
  START_MS=$(python3 -c "import time; print(int(time.time()*1000))")
  RESP=$(curl -s --max-time 120 "$URL/v1/chat/completions" -H "Content-Type: application/json" \
    -d "$(build_payload "$EASY_PROMPT" 256)")
  END_MS=$(python3 -c "import time; print(int(time.time()*1000))")
  N_OUT=$(echo "$RESP" | jq -r '.usage.completion_tokens // 0')
  ELAPSED_MS=$((END_MS - START_MS))
  TPS=$(python3 -c "print(round($N_OUT * 1000.0 / max($ELAPSED_MS,1), 1))")
  EASY_TPS+=("$TPS")
  log "  easy run $run: $N_OUT tok in ${ELAPSED_MS}ms = $TPS tok/s"
done
EASY_MEDIAN=$(printf '%s\n' "${EASY_TPS[@]}" | sort -n | sed -n 2p)

# --- Coherence check (lenient: answer must appear somewhere in response) ---
COHERENCE_RESP=$(curl -s --max-time 120 "$URL/v1/chat/completions" -H "Content-Type: application/json" \
  -d "$(build_payload "What is 51 times 37? Answer with just the number." 64)")
COHERENCE_TXT=$(echo "$COHERENCE_RESP" | jq -r '.choices[0].message.content // ""')
COHERENT="✅"
if ! echo "$COHERENCE_TXT" | grep -q "1887"; then
  COHERENT="❌"
  log "  COHERENCE FAIL: 51×37 — got '$(echo "$COHERENCE_TXT" | head -c 80)' expected 1887"
fi
PARIS_RESP=$(curl -s --max-time 120 "$URL/v1/chat/completions" -H "Content-Type: application/json" \
  -d "$(build_payload "What is the capital of France? One word." 32)")
PARIS_TXT=$(echo "$PARIS_RESP" | jq -r '.choices[0].message.content // ""')
if ! echo "$PARIS_TXT" | grep -qi "paris"; then
  COHERENT="❌"
  log "  COHERENCE FAIL: capital — got '$(echo "$PARIS_TXT" | head -c 80)' expected Paris"
fi
log "  coherence: $COHERENT (51×37=$(echo "$COHERENCE_TXT" | head -c 60), capital=$(echo "$PARIS_TXT" | head -c 40))"

# --- Concurrent benchmark ---
CONC="$CONCURRENCY"
CONC_PROMPT=""
for i in $(seq 1 3); do CONC_PROMPT="$CONC_PROMPT$PARA "; done
CONC_PAYLOAD=$(build_payload "Answer in one sentence: what is the mechanism described in the following text?\n\n$CONC_PROMPT" 256)

# Warmup concurrent — send in batches of MAX_SEQS to avoid overwhelming the queue
WARMUP_BATCH="$MAX_SEQS"
if [ "$WARMUP_BATCH" -gt "$CONC" ]; then WARMUP_BATCH="$CONC"; fi
for i in $(seq 1 "$WARMUP_BATCH"); do
  curl -s --max-time 120 "$URL/v1/chat/completions" -H "Content-Type: application/json" -d "$CONC_PAYLOAD" -o "/tmp/warmup_${KNOB_NAME}_$i.json" &
done
wait
sleep 2

CONC_TPS=()
for round in 1 2 3; do
  T0=$(python3 -c "import time; print(time.time())")
  TOTAL_TOKENS=0
  for i in $(seq 1 "$CONC"); do
    curl -s --max-time 120 "$URL/v1/chat/completions" -H "Content-Type: application/json" -d "$CONC_PAYLOAD" -o "/tmp/resp_${KNOB_NAME}_${round}_$i.json" &
  done
  wait
  T1=$(python3 -c "import time; print(time.time())")
  for i in $(seq 1 "$CONC"); do
    if [ -f "/tmp/resp_${KNOB_NAME}_${round}_$i.json" ]; then
      N=$(jq -r '.usage.completion_tokens // 0' "/tmp/resp_${KNOB_NAME}_${round}_$i.json" 2>/dev/null || echo 0)
    else
      N=0
    fi
    TOTAL_TOKENS=$((TOTAL_TOKENS + N))
  done
  WALL=$(python3 -c "print($T1-$T0)")
  TPS=$(python3 -c "print(round($TOTAL_TOKENS/$WALL, 1))")
  CONC_TPS+=("$TPS")
  log "  conc round $round: $TPS tok/s aggregate ($TOTAL_TOKENS tok in ${WALL}s)"
done
CONC_MEDIAN=$(printf '%s\n' "${CONC_TPS[@]}" | sort -n | sed -n 2p)

# ---------------------------------------------------------------------------
# 5. Record in HILLCLIMB.md
# ---------------------------------------------------------------------------
# Find next iteration number from history table
LAST_ITER=$(grep -oP '^\| \K[0-9]+' "$HILL" | sort -n | tail -1)
NEXT_ITER=$((LAST_ITER + 1))

STARS=""
if python3 -c "exit(0 if float('$CONC_MEDIAN') > 302.6 else 1)"; then
  STARS="⭐⭐⭐⭐⭐⭐"
elif python3 -c "exit(0 if float('$HARD_MEDIAN') > 50.5 else 1)"; then
  STARS="⭐"
fi

ROW="| $NEXT_ITER | $NOW | $STARS $KNOB_NAME | ${HARD_MEDIAN} hard / ${EASY_MEDIAN} easy / ${CONC_MEDIAN} conc×${CONC} agg | $COHERENT |"

VERDICT=""
BEST_HARD=50.5
BEST_CONC_EASY=302.6
BEST_CONC_HARD=93.4
IS_BEST=0
if python3 -c "exit(0 if float('$HARD_MEDIAN') > $BEST_HARD else 1)"; then
  IS_BEST=1
  VERDICT="NEW BEST single-stream hard (${HARD_MEDIAN} vs ${BEST_HARD}). "
fi
if python3 -c "exit(0 if float('$CONC_MEDIAN') > $BEST_CONC_HARD else 1)"; then
  IS_BEST=1
  VERDICT="${VERDICT}NEW BEST concurrent hard agg (${CONC_MEDIAN} vs ${BEST_CONC_HARD}). "
fi

if [ -n "$VERDICT" ]; then
  ROW="$ROW ${VERDICT}Config: MTP${MTP}, mem=${MEM_UTIL}, seqs=${MAX_SEQS}, batched=${MAX_BATCHED}, modellen=${MAX_MODEL_LEN}, conc=${CONCURRENCY}."
else
  # Check if it's worse
  if python3 -c "exit(0 if float('$HARD_MEDIAN') < 40.0 and float('$CONC_MEDIAN') < 50.0 else 1)"; then
    VERDICT="REGRESSION. "
    ROW="$ROW ${VERDICT}Config: MTP${MTP}, mem=${MEM_UTIL}, seqs=${MAX_SEQS}, batched=${MAX_BATCHED}, modellen=${MAX_MODEL_LEN}, conc=${CONCURRENCY}."
  else
    ROW="$ROW Config: MTP${MTP}, mem=${MEM_UTIL}, seqs=${MAX_SEQS}, batched=${MAX_BATCHED}, modellen=${MAX_MODEL_LEN}, conc=${CONCURRENCY}."
  fi
fi

log "Recording: $ROW"
echo "$ROW" >> "$HILL"

# Mark knob as DONE in queue (use python to avoid sed pipe-delimiter issues)
RESULT_TAG="DONE|hard=${HARD_MEDIAN},easy=${EASY_MEDIAN},conc=${CONC_MEDIAN}"
python3 -c "
import sys
knob = sys.argv[1]
tag = sys.argv[2]
qfile = sys.argv[3]
with open(qfile) as f:
    lines = f.readlines()
out = []
for l in lines:
    if l.rstrip('\n') == knob:
        out.append(l.rstrip('\n') + '|' + tag + '\n')
    else:
        out.append(l)
with open(qfile, 'w') as f:
    f.writelines(out)
" "$KNOB_LINE" "$RESULT_TAG" "$QUEUE"

log "Done. hard=${HARD_MEDIAN} easy=${EASY_MEDIAN} conc=${CONC_MEDIAN} coherent=$COHERENT"

# If this was a new best, also update Current Best table
if [ "$IS_BEST" -eq 1 ]; then
  log "NEW BEST detected — updating Current Best table in HILLCLIMB.md"
fi

# ---------------------------------------------------------------------------
# 6. Git commit on host
# ---------------------------------------------------------------------------
cd "$REPO"
git add HILLCLIMB.md knob_queue.txt hillclimb_automation.log 2>/dev/null || true
git commit -m "hillclimb: $KNOB_NAME — hard=${HARD_MEDIAN} easy=${EASY_MEDIAN} conc=${CONC_MEDIAN} coherent=$COHERENT" 2>/dev/null || true
git push 2>/dev/null || true

log "Git committed and pushed (if possible). Iteration complete."
