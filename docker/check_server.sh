#!/usr/bin/env bash
# 서버 상태 확인: 결과를 그대로 복사해 공유하면 됨
echo "== host"; hostname; uname -r; whoami; id
echo "== gpu"; nvidia-smi --query-gpu=index,name,memory.total,memory.used,driver_version --format=csv 2>&1
echo "== docker"; docker --version 2>&1; docker info 2>/dev/null | grep -iE "runtimes|default runtime" ; docker ps --format '{{.Names}}\t{{.Image}}\t{{.Status}}' 2>&1 | head -20
echo "== nvidia container toolkit"; which nvidia-container-toolkit nvidia-ctk 2>&1
echo "== disk"; df -h ~ | tail -1
