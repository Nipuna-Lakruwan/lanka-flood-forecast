"""Train and evaluate river-level forecasts on two time blocks the model never sees.

Folds (with a 48-hour buffer around each test block so no training row overlaps it):
- "ditwah": 23 Nov - 31 Dec 2025, which contains Cyclone Ditwah's floods; trained on Jan-Nov 2026.
- "recent": 15 Sep - 17 Nov 2026, the most recent months; trained on everything before.

Training on later data to test an earlier block (the "ditwah" fold) is not a real-time setup, so that fold
is reported as an event study. The "recent" fold is the honest forward-in-time result.
"""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from data import PROCESSED, ROOT
from features import FEATURES, HORIZONS, RAIN_FEATURES, build_table

RAIN_SET = FEATURES + RAIN_FEATURES

RESULTS = ROOT / "results"
BUFFER = pd.Timedelta(hours=48)
FOLDS = {
    "ditwah": (pd.Timestamp("2025-11-23", tz="UTC"), pd.Timestamp("2026-01-01", tz="UTC")),
    "recent": (pd.Timestamp("2026-09-15", tz="UTC"), pd.Timestamp("2026-11-18", tz="UTC")),
}


def split(table: pd.DataFrame, fold: str):
    start, end = FOLDS[fold]
    test = table[(table.time >= start) & (table.time < end)]
    if fold == "recent":
        train = table[table.time < start - BUFFER]
    else:
        train = table[(table.time >= end + BUFFER)]
    return train, test


WARN_QUANTILES = (0.8, 0.9)


def fit_lgb(
    train: pd.DataFrame, h: int, quantile: float | None = None, features=FEATURES
) -> lgb.LGBMRegressor:
    rows = train[train[f"y{h}"].notna()]
    loss = {"objective": "quantile", "alpha": quantile} if quantile else {"objective": "huber", "alpha": 1.5}
    model = lgb.LGBMRegressor(
        n_estimators=600,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=50,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        verbose=-1,
        **loss,
    )
    model.fit(rows[features], rows[f"y{h}"].clip(-60, 60))
    return model


def predictions(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    out = test[["time", "station", "level", "scale", "minor_flood_level_m", "alert_level_m"]].copy()
    for h in HORIZONS:
        model = fit_lgb(train, h)
        out[f"true_{h}"] = test[f"level_future_{h}"]
        out[f"persist_{h}"] = test.level
        # Trend baseline: continue the last 3 hours' rate of change, damped by half.
        out[f"trend_{h}"] = test.level + 0.5 * test.d3 * test.scale * h / 3
        out[f"lgb_{h}"] = test.level + model.predict(test[FEATURES]) * test.scale
        rain_model = fit_lgb(train, h, features=RAIN_SET)
        out[f"lgbrain_{h}"] = test.level + rain_model.predict(test[RAIN_SET]) * test.scale
        if h == 12:
            # Upper-quantile forecasts, used only for warnings: "how high could it plausibly get?"
            for q in WARN_QUANTILES:
                for name, feats in (("lgbq", FEATURES), ("lgbrainq", RAIN_SET)):
                    mq = fit_lgb(train, h, quantile=q, features=feats)
                    out[f"{name}{int(q * 100)}_{h}"] = test.level + mq.predict(test[feats]) * test.scale
    return out


def scores(pred: pd.DataFrame) -> dict:
    res = {}
    for h in HORIZONS:
        p = pred[pred[f"true_{h}"].notna()]
        row = {}
        for m in ("persist", "trend", "lgb", "lgbrain"):
            err = (p[f"{m}_{h}"] - p[f"true_{h}"]).abs()
            row[f"{m}_mae_cm"] = round(100 * err.mean(), 2)
            # During fast change: hours where the level moved more than 2 typical 6-hour changes.
            big = (p[f"true_{h}"] - p.level).abs() > 2 * p.scale
            row[f"{m}_mae_big_moves_cm"] = round(100 * err[big].mean(), 2) if big.any() else None
        row["lgb_skill_vs_persistence"] = round(1 - row["lgb_mae_cm"] / row["persist_mae_cm"], 3)
        row["lgbrain_skill_vs_persistence"] = round(1 - row["lgbrain_mae_cm"] / row["persist_mae_cm"], 3)
        row["rows"] = len(p)
        row["big_move_rows"] = int(big.sum())
        res[f"{h}h"] = row
    return res


def flood_warnings(pred: pd.DataFrame, h: int, level_col: str = "minor_flood_level_m") -> dict:
    """Warn when the forecast for h hours ahead is at or above the flood level while the river is still below it.

    An event is a station going from below to at-or-above the level. A warning counts as early if it was issued
    at least 1 hour before the crossing (and at most h hours before).
    """
    out = {}
    models = ["persist", "trend", "lgb", "lgbrain"] + [
        f"{n}{int(q * 100)}" for n in ("lgbq", "lgbrainq") for q in WARN_QUANTILES
    ]
    for m in models:
        if f"{m}_{h}" not in pred:
            continue
        hits, misses, false_alarms, leads = 0, 0, 0, []
        for _, g in pred[pred[level_col].notna()].groupby("station", observed=True):
            g = g.sort_values("time").reset_index(drop=True)
            thr = g[level_col].iloc[0]
            below = g.level < thr
            crossings = g.index[(~below) & below.shift(1, fill_value=False)]
            warn = below & (g[f"{m}_{h}"] >= thr)
            used = set()
            for c in crossings:
                window = g.index[(g.index >= c - h) & (g.index < c)]
                w = [i for i in window if warn[i]]
                if w:
                    hits += 1
                    leads.append(c - w[0])
                    used.update(range(w[0], c))
                else:
                    misses += 1
            # A false alarm is a warning hour with no crossing in the next h hours.
            for i in g.index[warn]:
                if i in used:
                    continue
                if not any(0 < c - i <= h for c in crossings):
                    false_alarms += 1
        out[m] = {
            "flood_events": hits + misses,
            "warned_in_time": hits,
            "missed": misses,
            "false_alarm_hours": false_alarms,
            "median_lead_hours": float(np.median(leads)) if leads else None,
            "detection_rate": round(hits / max(1, hits + misses), 3),
        }
    return out


def main() -> None:
    levels = pd.read_parquet(PROCESSED / "levels_hourly.parquet")
    stations = pd.read_parquet(PROCESSED / "stations.parquet")
    table = build_table(levels, stations)
    RESULTS.mkdir(exist_ok=True)
    report = {}
    for fold in FOLDS:
        train, test = split(table, fold)
        pred = predictions(train, test)
        pred.to_parquet(RESULTS / f"pred_{fold}.parquet")
        report[fold] = {
            "train_rows": len(train),
            "test_rows": len(test),
            "accuracy": scores(pred),
            "minor_flood_warnings_12h": flood_warnings(pred, 12),
            "alert_level_warnings_12h": flood_warnings(pred, 12, "alert_level_m"),
        }
        print(f"\n=== {fold}: train {len(train):,} rows, test {len(test):,} rows")
        print(pd.DataFrame(report[fold]["accuracy"]).T.to_string())
        for name in ("minor_flood_warnings_12h", "alert_level_warnings_12h"):
            print(f"\n{name}")
            print(pd.DataFrame(report[fold][name]).T.to_string())
    Path(RESULTS / "metrics.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
