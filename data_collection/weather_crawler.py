"""Weather Underground 월별 기상 이력 크롤러 (Selenium headless Chrome).

관측소 페이지(history/monthly)를 월 단위로 열어 일별 최고·최저 기온과 습도 표를 수집한다.
ICESI 실험(UCI 가정 전력 데이터 + 파리 오를리 LFPO 기상 병합)에 사용했다.

사용 예
    python weather_crawler.py --station fr/villeneuve-le-roi/LFPO \
        --start 2006-12-16 --end 2010-11-26 --out weather_data.csv
    # chromedriver 를 직접 지정하려면 --chromedriver /path/to/chromedriver
    # (생략하면 Selenium Manager 가 브라우저에 맞는 드라이버를 내려받는다)

주의: Weather Underground 이용약관상 수집 데이터의 재배포에는 제약이 있으므로
수집 스크립트만 공개하고 결과 CSV 는 저장소에 포함하지 않는다.
"""
from __future__ import annotations

import argparse
import time
from io import StringIO

import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from tqdm import tqdm

USER_AGENT = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36")


def build_driver(chromedriver: str | None = None, headless: bool = True) -> webdriver.Chrome:
    opts = webdriver.ChromeOptions()
    opts.add_argument(f"user-agent={USER_AGENT}")
    if headless:
        opts.add_argument("--headless=new")
    service = Service(chromedriver) if chromedriver else None
    return webdriver.Chrome(options=opts, service=service)


def _col(table: pd.DataFrame, prefix: str, stat: str) -> pd.Series:
    """MultiIndex 헤더에서 level-0 이 prefix 로 시작하고 level-1 이 stat 인 열을 찾는다 (단위 표기 변화에 강건)."""
    for c in table.columns:
        if str(c[0]).startswith(prefix) and str(c[1]) == stat:
            return pd.to_numeric(table[c], errors="coerce")
    raise KeyError(f"{prefix}/{stat} 열을 찾지 못함: {list(table.columns)[:8]}")


def parse_daily_table(html: str) -> pd.DataFrame:
    """페이지의 표 중 일별 요약표(Date 열 + 항목별 Max/Avg/Min)를 골라 기온·습도 최고·최저를 뽑는다."""
    tables = pd.read_html(StringIO(html))
    daily = [t for t in tables if isinstance(t.columns, pd.MultiIndex) and t.shape[0] >= 20]
    if not daily:
        raise ValueError("일별 요약표가 아직 렌더링되지 않음")
    t = daily[0]
    date_col = [c for c in t.columns if str(c[0]).startswith("Date")][0]
    out = pd.DataFrame({
        "temp_high": _col(t, "Temperature", "Max"),
        "temp_low": _col(t, "Temperature", "Min"),
        "humid_high": _col(t, "Humidity", "Max"),
        "humid_low": _col(t, "Humidity", "Min"),
    })
    out.index = pd.to_datetime(t[date_col].astype(str), format="%m/%d/%Y")
    return out


def crawl_monthly(driver: webdriver.Chrome, station: str, start: str, end: str,
                  max_retry: int = 5, wait_sec: int = 8) -> pd.DataFrame:
    """월별 페이지를 순회하며 일별 기온·습도 표를 하나의 DataFrame 으로 합친다."""
    months = pd.date_range(start=pd.Timestamp(start).replace(day=1), end=end, freq="MS")
    frames = []
    for month in tqdm(months, desc=station):
        ym = month.strftime("%Y-%m")
        url = f"https://www.wunderground.com/history/monthly/{station}/date/{ym}"
        part = None
        for _ in range(max_retry):
            try:
                driver.get(url)
                time.sleep(wait_sec)                      # 표가 JS 로 늦게 렌더링됨
                part = parse_daily_table(driver.page_source)
                break
            except Exception:
                time.sleep(2)
        if part is None:
            print(f"[skip] {ym}: 재시도 {max_retry}회 실패")
            continue
        frames.append(part)
    df = pd.concat(frames).sort_index()
    return df.loc[start:end]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--station", default="fr/villeneuve-le-roi/LFPO", help="Weather Underground 관측소 경로")
    ap.add_argument("--start", default="2006-12-16")
    ap.add_argument("--end", default="2010-11-26")
    ap.add_argument("--out", default="weather_data.csv")
    ap.add_argument("--chromedriver", default=None, help="chromedriver 경로 (생략 가능)")
    ap.add_argument("--no-headless", action="store_true")
    args = ap.parse_args()

    driver = build_driver(args.chromedriver, headless=not args.no_headless)
    try:
        df = crawl_monthly(driver, args.station, args.start, args.end)
    finally:
        driver.quit()
    df.to_csv(args.out, index_label="date")
    print(f"saved {len(df)} rows -> {args.out}")


if __name__ == "__main__":
    main()
