#!/usr/bin/env bash
# 사용법: GPU=0 bash docker/run.sh        (저장소 루트에서 실행)
#   NAME  컨테이너 이름 (기본 ami-dl)      IMG   이미지 태그 (기본 ami-dl:0.1)
#   GPU   사용할 GPU 번호 (기본 0)          DATA  읽기 전용으로 마운트할 데이터 폴더
#   RUNS  실험 산출물 폴더 (쓰기 가능)
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
NAME=${NAME:-ami-dl}; GPU=${GPU:-0}; IMG=${IMG:-ami-dl:0.1}
DATA=${DATA:-$HOME/ami/data}; RUNS=${RUNS:-$HOME/ami/runs}
mkdir -p "$DATA" "$RUNS"

docker build -t "$IMG" -f "$ROOT/docker/Dockerfile" "$ROOT/docker"
docker rm -f "$NAME" 2>/dev/null || true
# --user 로 호스트 UID/GID 를 넘겨 컨테이너가 만든 파일을 호스트에서 그대로 다룰 수 있게 한다.
docker run -d --name "$NAME" --gpus "\"device=$GPU\"" --shm-size 8g \
  --user "$(id -u):$(id -g)" -e HOME=/workspace \
  -v "$ROOT":/workspace -v "$DATA":/data:ro -v "$RUNS":/runs \
  "$IMG"
docker exec "$NAME" python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
docker exec "$NAME" python -c "from pyspark.sql import SparkSession;s=SparkSession.builder.getOrCreate();print('spark',s.version);s.stop()"
echo "접속: docker exec -it $NAME bash"
