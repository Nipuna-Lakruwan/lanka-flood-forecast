"""Turn the hourly level table into a training table: one row per (station, hour).

Every feature uses only readings at or before the forecast time. Targets are the change in water level
`h` hours later, scaled by each station's typical variability so that one model can learn from all stations.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from data import RAW
from rain import WINDOWS as RAIN_WINDOWS
from rain import rain_features

HORIZONS = (6, 12, 24)
LAGS = (1, 2, 3, 6, 12, 24)


def upstream_map(stations: pd.DataFrame) -> dict[str, list[str]]:
    """Stations upstream on the same river, using the river's upstream-to-downstream location order."""
    rivers = json.loads((RAW / "rivers.json").read_text())
    known = set(stations.index)
    ups: dict[str, list[str]] = {s: [] for s in known}
    for river in rivers:
        order = [n for n in river["location_names"] if n in known]
        for i, s in enumerate(order):
            ups[s] = order[:i]
    return ups


def basin_peers(stations: pd.DataFrame) -> dict[str, list[str]]:
    peers = {}
    for s, basin in stations.basin_name.items():
        peers[s] = [o for o in stations.index[stations.basin_name == basin] if o != s]
    return peers


def build_table(levels: pd.DataFrame, stations: pd.DataFrame, with_rain: bool = True) -> pd.DataFrame:
    stations = stations.loc[stations.index.intersection(levels.columns)]
    ups, peers = upstream_map(stations), basin_peers(stations)
    rain = rain_features(levels.index, stations) if with_rain else {}
    # Per-station scale: median absolute 6-hour change, so a 0.3 m rise means the same "size" everywhere.
    scale = levels.diff(6).abs().median().clip(lower=0.01)
    mean = levels.mean()
    z = (levels - mean) / levels.std().clip(lower=0.01)
    dz6 = levels.diff(6) / scale
    dz3 = levels.diff(3) / scale

    frames = []
    for s in stations.index:
        x = levels[s]
        f = pd.DataFrame(index=levels.index)
        f["level_z"] = z[s]
        for lag in LAGS:
            f[f"d{lag}"] = (x - x.shift(lag)) / scale[s]
        f["max24_minus_now"] = (x.rolling(24, min_periods=12).max() - x) / scale[s]
        f["min24_minus_now"] = (x.rolling(24, min_periods=12).min() - x) / scale[s]
        minor = stations.loc[s, "minor_flood_level_m"]
        f["to_minor_flood"] = (minor - x) / scale[s] if pd.notna(minor) else np.nan
        up = ups.get(s, [])
        f["n_upstream"] = len(up)
        f["up_d3_max"] = dz3[up].max(axis=1) if up else np.nan
        f["up_d6_max"] = dz6[up].max(axis=1) if up else np.nan
        f["up_z_max"] = z[up].max(axis=1) if up else np.nan
        pr = peers.get(s, [])
        f["basin_d6_mean"] = dz6[pr].mean(axis=1) if pr else np.nan
        f["basin_z_mean"] = z[pr].mean(axis=1) if pr else np.nan
        # Country-wide signal: how many stations are rising fast right now (a proxy for heavy rain).
        f["national_rising_share"] = (dz3 > 2).mean(axis=1)
        local = levels.index.tz_convert("Asia/Colombo")
        f["hour_sin"] = np.sin(2 * np.pi * local.hour / 24)
        f["hour_cos"] = np.cos(2 * np.pi * local.hour / 24)
        f["doy_sin"] = np.sin(2 * np.pi * local.dayofyear / 365.25)
        f["doy_cos"] = np.cos(2 * np.pi * local.dayofyear / 365.25)
        if s in rain:
            f = f.join(rain[s])
        for h in HORIZONS:
            f[f"y{h}"] = (x.shift(-h) - x) / scale[s]
            f[f"level_future_{h}"] = x.shift(-h)
        f["level"] = x
        f["scale"] = scale[s]
        f["minor_flood_level_m"] = minor
        f["alert_level_m"] = stations.loc[s, "alert_level_m"]
        f["station"] = s
        frames.append(f.reset_index())
    table = pd.concat(frames, ignore_index=True)
    table["station"] = table["station"].astype("category")
    return table[table.level.notna()]


FEATURES = [
    "level_z",
    *[f"d{lag}" for lag in LAGS],
    "max24_minus_now",
    "min24_minus_now",
    "to_minor_flood",
    "n_upstream",
    "up_d3_max",
    "up_d6_max",
    "up_z_max",
    "basin_d6_mean",
    "basin_z_mean",
    "national_rising_share",
    "hour_sin",
    "hour_cos",
    # Day-of-year is left out on purpose: with one year of data it would let the model memorise dates.
]

RAIN_FEATURES = [*[f"rain_{w}h" for w in RAIN_WINDOWS], "rain_national_max24"]
