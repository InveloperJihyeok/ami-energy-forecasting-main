#!/bin/bash
# 자기지도 사전학습 3종(온수·수도·가스) 순차 실행. 저장소 루트에서: bash paper_code/self-supervised/exp.sh
cd "$(dirname "$0")"
python rae_hw.py 1
python rae_water.py 1
python rae_gas.py 1
