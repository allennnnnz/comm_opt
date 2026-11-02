"""Load a run directory produced by ncclrun.run_all into one row per (job, size, place)."""
import json
import sys
from pathlib import Path

import pandas as pd


def load_run(run_dir: Path) -> pd.DataFrame:
    sched = pd.read_csv(run_dir / "schedule.tsv", sep="\t", comment="#", keep_default_na=False)
    rows = []
    for job in sched.itertuples():
        f = run_dir / f"{job.tag}.json"
        try:
            results = json.load(open(f))["results"]
        except (OSError, json.JSONDecodeError, KeyError) as e:
            print(f"skip {f.name}: {e}", file=sys.stderr)
            continue
        seen = set()
        for r in results:
            if r["size"] == 0 or r["size"] in seen:  # sub-element rounding: 0-byte rows and repeated sizes
                continue
            seen.add(r["size"])
            t = r["tuning"]
            for place in ("out_of_place", "in_place"):
                it = r[f"{place}_per_iter"]
                rows.append(dict(
                    tag=job.tag, order=job.order, coll=job.coll, ngpus=job.ngpus, gpuset=job.gpuset,
                    variant=job.variant, rep=job.rep, size=r["size"], place=place,
                    time_us=r[place]["time"], busbw=r[place]["bus_bw"], nwrong=r[place]["nwrong"],
                    p50_us=it["p50_us"], p99_us=it["p99_us"], cv_pct=it["cv_pct"],
                    impl=t["implementation"], algo=t["algo"], proto=t["proto"], channels=t["#channels"]))
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, keys: list, value: str = "busbw") -> pd.DataFrame:
    """Median and [min, max] across independent launches (~94% nonparametric CI of the median for n=5)."""
    return df.groupby(keys)[value].agg(med="median", lo="min", hi="max", n="count").reset_index()


def disjoint(a, b) -> bool:
    """[min, max] intervals do not overlap (R5)."""
    return a.hi < b.lo or b.hi < a.lo
