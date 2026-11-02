"""UCI 'Individual household electric power consumption' 데이터 전처리 + 크롤링 기상 병합.

- 1분 단위 유효전력(Global_active_power)을 15분 합으로 리샘플링
- 시간 특징(분·시·일·월 sin/cos, 요일 더미) 생성
- weather_crawler.py 결과(일별 최고·최저 기온·습도)를 날짜 기준으로 병합

데이터: https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption
사용 예
    python preprocess_uci.py --input household_power_consumption.txt \
        --weather weather_data.csv --out dataset.csv
"""
import argparse

import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="household_power_consumption.txt")
    ap.add_argument("--weather", default=None, help="weather_crawler.py 출력 CSV (선택)")
    ap.add_argument("--out", default="dataset.csv")
    ap.add_argument("--freq", default="15min")
    args = ap.parse_args()

    data = pd.read_csv(args.input, sep=";", low_memory=False)
    data["Global_active_power"] = pd.to_numeric(data["Global_active_power"], errors="coerce")
    data["electricity_consumption"] = data["Global_active_power"]
    data["DateTime"] = pd.to_datetime(data["Date"] + " " + data["Time"], format="%d/%m/%Y %H:%M:%S")
    data["DateTime"] += pd.to_timedelta("1 minute")      # 구간 끝 시각 기준으로 정렬
    data = data.set_index("DateTime")[["electricity_consumption"]]
    data = data.resample(args.freq, closed="left", label="right").sum()

    idx = data.index
    for i, dow in enumerate(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]):
        data[dow] = (idx.dayofweek == i).astype(int)

    f = lambda v: (np.sin(2 * np.pi * v), np.cos(2 * np.pi * v))
    data["minute_x"], data["minute_y"] = f(idx.minute / 60.0)
    data["hour_x"], data["hour_y"] = f((idx.hour + idx.minute / 60.0) / 24.0)
    data["day_x"], data["day_y"] = f(idx.day / idx.days_in_month)
    data["month_x"], data["month_y"] = f(idx.month / 12.0)

    if args.weather:
        w = pd.read_csv(args.weather, index_col=0, parse_dates=True)
        w = w[~w.index.duplicated(keep="first")].sort_index()   # 크롤링 재시도로 생긴 중복 날짜 제거
        w = w.apply(pd.to_numeric, errors="coerce")
        w = w.reindex(idx.normalize())                   # 일별 값을 각 15분 구간에 복사
        for col in w.columns:
            data[col] = w[col].to_numpy()
    data.to_csv(args.out, index_label="DateTime")
    print(f"saved {data.shape} -> {args.out}")


if __name__ == "__main__":
    main()
