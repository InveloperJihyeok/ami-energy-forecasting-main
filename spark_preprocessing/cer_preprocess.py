"""CER 스마트미터(아일랜드 CER Smart Metering Project) 원시 데이터 전처리 — PySpark + Spark SQL.

입력
    <raw-dir>/power_raw/*        공백 구분 텍스트 (id, 시간코드, 사용량[kWh/30min]) — 약 1.6억 행
    <raw-dir>/allocations.csv    가구 id 별 참여 그룹 코드 (1: 주거, 2: 소상공인 …)
    <climate-file>               climate_crawler.py 로 수집·병합한 일별 기상 pickle (date 컬럼 포함)
출력
    <out-dir>/values/H<id>.bin   가구별 30분 단위 시계열 (선형 보간 · bfill)
    <out-dir>/master.bin         시간 특징(사인·코사인, 요일, 공휴일) + 기상 + 전 가구 사용량 (wide)

CER 데이터는 ISSDA(Irish Social Science Data Archive) 에 신청해 받아야 하며 재배포할 수 없다.
    https://www.ucd.ie/issda/data/commissionforenergyregulationcer/

사용 예
    spark-submit cer_preprocess.py --raw-dir /data/CER --out-dir data \
        --climate-file data/raw/climate_history/ireland.bin
"""
import argparse
import pathlib
import pickle
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession

CER_EPOCH = datetime(2009, 1, 1)
POWER_RAW_SCHEMA = "id string, code string, value float"
WEEKDAY_DICT = {0: "mon", 1: "tue", 2: "wed", 3: "thu", 4: "fri", 5: "sat", 6: "sun"}

# https://www.timeanddate.com/holidays/ireland
IRELAND_HOLIDAYS = pd.to_datetime([
    "2009-01-01", "2009-03-17", "2009-04-10", "2009-04-12", "2009-04-13", "2009-05-04", "2009-06-01",
    "2009-08-03", "2009-10-26", "2009-12-24", "2009-12-25", "2009-12-26", "2009-12-31",
    "2010-01-01", "2010-03-17", "2010-04-02", "2010-04-04", "2010-04-05", "2010-05-03", "2010-06-07",
    "2010-08-02", "2010-10-25", "2010-12-24", "2010-12-25", "2010-12-26", "2010-12-31",
    "2011-01-01", "2011-03-17", "2011-04-22", "2011-04-24", "2011-04-25", "2011-05-02", "2011-06-06",
    "2011-08-01", "2011-10-31", "2011-12-24", "2011-12-25", "2011-12-26", "2011-12-31",
])


# ----------------------------------------------------------------------------- CER 시간 코드
def parse_cer_time_code(code: str) -> datetime:
    """CER 시간코드 'DDDTT' → datetime. DDD: 2009-01-01 기준 일수(1부터), TT: 30분 틱(1~48)."""
    code_s = str(code)
    days, tick = int(code_s[:3]) - 1, int(code_s[3:]) - 1
    return CER_EPOCH + timedelta(days=days) + timedelta(minutes=30 * tick)


def generate_cer_time_code(start_time: datetime, end_time: datetime) -> pd.DataFrame:
    """[start, end] 구간의 30분 간격 시각과 CER 시간코드 매핑표를 만든다."""
    t_seq = pd.date_range(start_time, end_time, freq="30min").to_series()
    t_df = (t_seq - CER_EPOCH).dt.components
    t_df.index.name = "time"
    t_df["days_s"] = (t_df["days"] + 1).astype(str)
    t_df["tick_s"] = (t_df["hours"] * 2 + t_df["minutes"] / 30 + 1).astype(int).astype(str).str.zfill(2)
    t_df["code"] = t_df["days_s"] + t_df["tick_s"]
    return t_df.reset_index()[["time", "code"]]


# ----------------------------------------------------------------------------- 시간 특징
def create_time_feature(x: pd.Series) -> pd.DataFrame:
    """주기 특징(분·시·일·월의 sin/cos)과 요일 더미를 만든다."""
    df = pd.DataFrame({"date": x.dt.date, "weekday_": x.dt.weekday, "year_": x.dt.year, "month_": x.dt.month,
                       "day_": x.dt.day, "hour_": x.dt.hour, "minute_": x.dt.minute}, index=x.index)
    d_s = pd.Series(pd.date_range(datetime(df["year_"].min(), 1, 1), datetime(df["year_"].max(), 12, 31), freq="1D"))
    d_df = d_s.groupby([d_s.dt.year, d_s.dt.month]).count().reset_index()
    d_df.columns = ["year_", "month_", "daysofmonth_"]
    df = pd.merge(df, d_df, on=["year_", "month_"], how="inner")
    df = df.astype({"month_": float, "day_": float, "hour_": float, "minute_": float, "daysofmonth_": float})

    f = lambda v: (np.sin(2 * np.pi * v), np.cos(2 * np.pi * v))
    df["minute_x"], df["minute_y"] = f(df["minute_"] / 60.0)
    df["hour_x"], df["hour_y"] = f((df["hour_"] + df["minute_"] / 60.0) / 24.0)
    df["day_x"], df["day_y"] = f(df["day_"] / df["daysofmonth_"])
    df["month_x"], df["month_y"] = f(df["month_"] / 12.0)

    # 기간이 짧아 일부 요일이 없어도 열이 항상 7개가 되도록 reindex (원본은 전체 요일이 있다고 가정했음)
    w_df = pd.get_dummies(df["weekday_"].map(WEEKDAY_DICT)).reindex(columns=list(WEEKDAY_DICT.values()), fill_value=0).astype(int)
    w_df = w_df.rename(columns=lambda c: c + "_yn").drop(columns=["sun_yn"])
    df = pd.concat([df, w_df], axis=1)
    df["time"] = x.values
    return df[["time", "date", "minute_x", "minute_y", "hour_x", "hour_y", "day_x", "day_y", "month_x", "month_y",
               "mon_yn", "tue_yn", "wed_yn", "thu_yn", "fri_yn", "sat_yn"]]


# ----------------------------------------------------------------------------- 메인
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", required=True, help="power_raw/ 와 allocations.csv 가 있는 폴더")
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--climate-file", default="data/raw/climate_history/ireland.bin")
    ap.add_argument("--groups", default="1,2", help="allocations.csv 의 참여 그룹 코드 (쉼표 구분)")
    args = ap.parse_args()

    raw_dir = pathlib.Path(args.raw_dir)
    out_dir = pathlib.Path(args.out_dir)
    values_dir = out_dir / "values"
    values_dir.mkdir(parents=True, exist_ok=True)

    spark = SparkSession.builder.appName("cer_preprocess").getOrCreate()

    # 1) 원시 사용량 + 참여 그룹 로드 → Spark SQL 로 대상 가구만 캐시
    power_files = [p.as_posix() for p in (raw_dir / "power_raw").glob("*")]
    spark.read.csv(power_files, sep=" ", schema=POWER_RAW_SCHEMA).createOrReplaceTempView("raw_pwr_view")
    spark.read.csv((raw_dir / "allocations.csv").as_posix(), header=True)[["id", "code"]] \
        .createOrReplaceTempView("allo_view")

    group_list = ", ".join(f"'{g.strip()}'" for g in args.groups.split(","))
    spark.sql("DROP TABLE IF EXISTS raw_pwr_tbl")
    spark.sql(f"""
        CACHE TABLE raw_pwr_tbl AS
        SELECT t1.id, t1.code, t1.value
        FROM (SELECT id, code, value FROM raw_pwr_view) AS t1
        JOIN (SELECT DISTINCT id FROM allo_view WHERE code IN ({group_list})) AS t2
          ON t1.id = t2.id
    """)

    # 2) 전체 기간의 30분 시각표 생성
    code_range = spark.sql("SELECT MIN(code) AS start_code, MAX(code) AS end_code FROM raw_pwr_tbl").toPandas()
    start_code, end_code = code_range.loc[0, ["start_code", "end_code"]]
    t_df = generate_cer_time_code(parse_cer_time_code(start_code), parse_cer_time_code(end_code))

    # 3) 가구별 시계열 추출 (중복 제거 → 시각표에 정렬 → 결측 보간) 후 pickle 저장
    ids = spark.sql("SELECT DISTINCT id FROM raw_pwr_tbl ORDER BY id").toPandas()["id"].to_list()
    for target_id in ids:
        save_file = values_dir / f"H{target_id}.bin"
        if save_file.exists():
            continue
        id_df = spark.sql(f"SELECT code, value FROM raw_pwr_tbl WHERE id == '{target_id}'").toPandas()
        id_df = id_df.drop_duplicates("code")
        id_df = pd.merge(t_df, id_df, on="code", how="left")[["value"]]
        id_df["value"] = id_df["value"].interpolate().bfill()
        id_df = id_df.reset_index(drop=True).rename(columns={"value": f"h{target_id}"})
        id_df.to_pickle(save_file)
        print("saved", save_file)

    # 4) 시간 특징 + 기상 + 공휴일 → 전 가구 wide 테이블(master.bin)
    time_df = create_time_feature(t_df["time"])
    with open(args.climate_file, "rb") as f:
        c_df = pickle.load(f)
    time_df = pd.merge(time_df, c_df, on="date", how="left").sort_values("time").reset_index(drop=True)
    time_df["holiday_yn"] = np.where(pd.to_datetime(time_df["date"]).isin(IRELAND_HOLIDAYS), 1, 0)

    h_df = pd.concat([pd.read_pickle(p) for p in sorted(values_dir.glob("*.bin"))], axis=1)
    master = pd.concat([time_df, h_df], axis=1)
    master.to_pickle(out_dir / "master.bin")
    print("saved", out_dir / "master.bin", master.shape)
    spark.stop()


if __name__ == "__main__":
    main()
