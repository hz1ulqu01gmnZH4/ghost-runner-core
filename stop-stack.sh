#!/usr/bin/env bash
# Stop the Ghost Runner stack (core, irodori-tts, whisper, llama).
# Thin alias for `start-stack.sh stop`, which owns the logic — it stops
# script-started components by pidfile and hand-started ones by command
# pattern, SIGTERM only.
exec bash "$(dirname "${BASH_SOURCE[0]}")/start-stack.sh" stop
