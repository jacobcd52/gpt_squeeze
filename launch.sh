#!/bin/bash
# usage: launch.sh <name> <task> [extra agent args...]
cd "$(dirname "$0")"
set -a; . ./.env; set +a
name=$1; task=$2; shift 2
mkdir -p /home/user/runs/$name
nohup python3 squeeze/agent.py --rundir /home/user/runs/$name --task-file tasks/$task.md "$@" > /home/user/runs/$name/agent.out 2>&1 &
echo "$name pid $!"
