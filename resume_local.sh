#!/bin/bash
# Restart a local run on the current harness code, continuing its stored response chain.
# usage: resume_local.sh <name> <task> [extra args]
cd "$(dirname "$0")"
set -a; . ./.env; set +a
name=$1; task=$2; shift 2
dir=/home/user/runs/$name
pid=$(pgrep -f "rundir $dir " | head -1)
dump=$(py-spy dump --pid $pid --locals 2>/dev/null)
rid=$(echo "$dump" | grep -oE '(prev|previous_response_id): "resp_[0-9a-f]+"' | head -1 | grep -oE 'resp_[0-9a-f]+')
[ -z "$rid" ] && { echo "no response id for $name"; exit 1; }
start=$(head -1 $dir/events.jsonl | python3 -c "import json,sys;print(json.loads(sys.stdin.read())['t'])")
kill $pid
nohup python3 squeeze/agent.py --rundir $dir --task-file tasks/$task.md --start "$start" --resume-response $rid --max-timeout 900 "$@" >> $dir/agent.out 2>&1 &
echo "$name resumed from $rid (start $start) pid $!"
