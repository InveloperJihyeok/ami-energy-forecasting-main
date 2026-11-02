"""Weather Underground 일별 기상 이력 크롤러 (Selenium) — `climate_downloader.downloader` 사용.

아일랜드 CER 스마트미터 실험(2009-07 ~ 2010-12)에 맞춰 섀넌 공항(EINN) 일별 기상을 수집했다.
날짜별로 pickle 로 저장해 중단 후 재개가 가능하고, 마지막에 하나의 DataFrame 으로 병합한다.

사용 예
    python climate_crawler.py --station ie/clenagh/EINN --start 2009-01-01 --end 2010-12-31 \
        --out-dir data/raw/climate_history/ireland
"""
import argparse
import pathlib
import pickle

import pandas as pd

import climate_downloader as cd


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--station", default="ie/clenagh/EINN")
    ap.add_argument("--start", default="2009-01-01")
    ap.add_argument("--end", default="2010-12-31")
    ap.add_argument("--out-dir", default="data/raw/climate_history/ireland")
    ap.add_argument("--chromedriver", default=None, help="chromedriver 경로 (생략 가능)")
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    example_url = f"https://www.wunderground.com/history/daily/{args.station}/date/2020-3-14"
    dwr = cd.downloader(args.chromedriver, example_url)

    try:
        for day in pd.date_range(args.start, args.end, freq="1D"):
            ds = day.date().isoformat()
            save_path = out_dir / f"{ds}.bin"
            if save_path.exists():          # 재개 지원
                continue
            data_df = dwr.get_data(ds)
            with open(save_path, "wb") as f:
                pickle.dump(data_df, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(ds)
    finally:
        dwr.close()

    frames = []
    for p in sorted(out_dir.glob("*.bin")):
        with open(p, "rb") as f:
            frames.append(pickle.load(f))
    merged = pd.concat(frames).reset_index(drop=True)
    merged_path = out_dir.parent / f"{out_dir.name}.bin"
    merged.to_pickle(merged_path)
    print(f"merged {len(merged)} rows -> {merged_path}")


if __name__ == "__main__":
    main()
