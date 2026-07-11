#!/usr/bin/env bash
# Ghost Runner stack launcher/stopper.
#
#   ./start-stack.sh          start everything that isn't already up
#   ./start-stack.sh stop     stop everything this script started
#   ./start-stack.sh status   health of all four components
#
# Components (in dependency order):
#   llama-server  :8080  Qwen3.6-27B  (LOCAL build — the system llama-server
#                        cannot load the MTP gguf; --reasoning off, or voice
#                        latency eats ~11 s of hidden thinking tokens)
#   whisper-server:8081  CPU (GPU whisper + 27B oversubscribes 32 GB VRAM)
#   irodori-tts   :8088  Irodori-TTS-500M-v3 (bf16 ~3.5 GB VRAM, preloads)
#   core          :8790  ghost-runner-core (fail-fast: refuses to start unless
#                        every dependency above answers)
#
# Fail-loud: a component that doesn't become healthy within its timeout aborts
# the launch with its log tail. Already-healthy components are left untouched,
# so the script is safe to re-run.

set -euo pipefail

LLAMA_BIN="$HOME/llama.cpp/build/bin/llama-server"
LLAMA_MODEL="$HOME/Qwen3.6-27B-UD-Q4_K_XL_MTP.gguf"
WHISPER_BIN="$HOME/whisper.cpp/build/bin/whisper-server"
WHISPER_MODEL="$HOME/models/whisper/ggml-large-v3-turbo.bin"
IRODORI_DIR="$HOME/irodori-tts-server"
CORE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="$CORE_DIR/config.toml"

RUN_DIR="$HOME/.ghost-runner"
LOG_DIR="$RUN_DIR/logs"
mkdir -p "$LOG_DIR"

say() { printf '[stack] %s\n' "$*"; }
die() { printf '[stack] ERROR: %s\n' "$*" >&2; exit 1; }

healthy() { # healthy <name>
    case "$1" in
        llama)   curl -sf -o /dev/null --max-time 2 http://127.0.0.1:8080/v1/models ;;
        whisper) curl -s  -o /dev/null --max-time 2 http://127.0.0.1:8081/ ;;
        irodori) curl -sf --max-time 2 http://127.0.0.1:8088/health 2>/dev/null \
                     | grep -q '"loaded":true' ;;
        core)    # the WS port answers HTTP with 426 Upgrade Required when alive
                 [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 \
                      http://127.0.0.1:8790/ 2>/dev/null)" = "426" ] ;;
    esac
}

wait_healthy() { # wait_healthy <name> <timeout_s>
    local name=$1 timeout=$2 waited=0
    until healthy "$name"; do
        pid=$(cat "$RUN_DIR/$name.pid" 2>/dev/null || true)
        if [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null; then
            printf '\n'; tail -15 "$LOG_DIR/$name.log" >&2
            die "$name exited during startup — full log: $LOG_DIR/$name.log"
        fi
        [ "$waited" -ge "$timeout" ] && {
            printf '\n'; tail -15 "$LOG_DIR/$name.log" >&2
            die "$name not healthy after ${timeout}s — full log: $LOG_DIR/$name.log"
        }
        sleep 1; waited=$((waited + 1)); printf '.'
    done
    printf '\n'
}

launch() { # launch <name> <timeout_s> <cmd...>
    local name=$1 timeout=$2; shift 2
    if healthy "$name"; then say "$name: already up"; return; fi
    say "$name: starting -> $LOG_DIR/$name.log"
    setsid "$@" >"$LOG_DIR/$name.log" 2>&1 < /dev/null &
    echo $! > "$RUN_DIR/$name.pid"
    wait_healthy "$name" "$timeout"
    say "$name: healthy"
}

start() {
    [ -x "$LLAMA_BIN" ]    || die "llama-server not found at $LLAMA_BIN"
    [ -f "$LLAMA_MODEL" ]  || die "chat model not found at $LLAMA_MODEL"
    [ -x "$WHISPER_BIN" ]  || die "whisper-server not found at $WHISPER_BIN"
    [ -f "$WHISPER_MODEL" ] || die "whisper model not found at $WHISPER_MODEL"
    [ -d "$IRODORI_DIR" ]  || die "irodori-tts-server not found at $IRODORI_DIR"
    [ -f "$CONFIG" ]       || die "core config not found at $CONFIG (copy config.toml.example)"

    launch llama 300 "$LLAMA_BIN" -m "$LLAMA_MODEL" -c 32768 -ngl 99 \
        --host 127.0.0.1 --port 8080 --reasoning off
    launch whisper 120 "$WHISPER_BIN" --host 127.0.0.1 --port 8081 \
        -m "$WHISPER_MODEL" --no-gpu -t 16 -l ja
    launch irodori 300 env -C "$IRODORI_DIR" uv run --no-sync irodori-openai-tts
    launch core 60 env -C "$CORE_DIR" uv run python -m ghost_runner_core.main \
        --config "$CONFIG"

    say "stack is up — client connects to ws://localhost:8790"
}

stop() {
    for name in core irodori whisper llama; do
        pid=$(cat "$RUN_DIR/$name.pid" 2>/dev/null || true)
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            say "$name: stopping (pid $pid)"
            kill "$pid"
        else
            say "$name: not running (no live pid)"
        fi
    done
}

status() {
    for name in llama whisper irodori core; do
        if healthy "$name"; then say "$name: healthy"; else say "$name: DOWN"; fi
    done
}

case "${1:-start}" in
    start)  start ;;
    stop)   stop ;;
    status) status ;;
    *) die "usage: $0 [start|stop|status]" ;;
esac
