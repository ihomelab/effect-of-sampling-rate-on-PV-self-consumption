import numpy as np
import pandas as pd
from .util import grouper


def samples_per_hour(params):
    return 3600 / pd.Timedelta(params["frequency_string"]).total_seconds()


def energy_from_series(series, params):
    return series.sum() / samples_per_hour(params)


def narrow_active_area(active_mask, active_area):
    start, end = active_area
    active_mask = active_mask.loc[start:end]
    return (
        active_mask[active_mask == True].index[0],
        active_mask[active_mask == True].index[-1],
    )


def detect_active_areas_unfiltered(
    input_series, window_size=12, greater_than=0, narrow=True
):
    active_mask = input_series > greater_than
    temp = active_mask.rolling(window_size, center=True).sum()
    temp = (temp > 0) & (np.isfinite(input_series))
    temp = temp.astype("int8").diff()
    areas = grouper(temp.index[(temp == 1) | (temp == -1)], n=2)
    areas = [(start, end) for start, end in areas if active_mask.loc[start:end].any()]
    if not narrow:
        return list(areas)
    areas = (narrow_active_area(active_mask, a) for a in areas)
    areas = ((a, b) for a, b in areas if a != b)
    return list(areas)


def force_split_active_area(active_area_series, n):
    ind = active_area_series.index
    borders = np.linspace(0, len(ind), n + 1).astype(np.int32)
    borders[-1] = len(ind)
    if len(set(borders)) != n + 1:
        return None
    return list(zip(borders, borders[1:]))


def rate_active_area_split(active_area_series, split, params):
    split_series = [active_area_series.iloc[start:end] for start, end in split]

    if any(s.max() < params["min_power_threshold"] for s in split_series):
        return np.inf

    energy_values = np.array([energy_from_series(s, params) for s in split_series])

    undershoot = (params["energy_min"] - energy_values).clip(min=0).sum()
    overshoot = (energy_values - params["energy_max"]).clip(min=0).sum()

    energy_limit_loss = (overshoot + undershoot) / (
        params["energy_min"] + params["energy_max"]
    )

    balance_loss = energy_values.std() / energy_values.mean() if len(split) > 1 else 0

    return energy_limit_loss + 0.1 * balance_loss


def get_rated_active_area_splits(active_area_series, params):
    active_area_series = active_area_series.asfreq(params["frequency_string"])
    rated_splits = []
    max_n = min(
        10,
        int(energy_from_series(active_area_series, params) / params["energy_min"]) + 1,
    )
    for n in range(1, max_n):
        split = force_split_active_area(active_area_series, n)
        if split is None:
            continue
        rating = rate_active_area_split(active_area_series, split, params)
        rated_splits.append((split, rating))
    return rated_splits


def detect_active_areas(machine_series, params):
    if machine_series.index.freq is None:
        raise ValueError(
            "machine_series.index.freq must not be None (to guarantee that there are no holes)"
        )
    machine_series_i_indexed = machine_series.reset_index(drop=True)

    all_active_areas = []
    offset = 0
    for _, day_series in list(
        machine_series_i_indexed.groupby(machine_series.index.date)
    ):
        last_offset = offset
        offset += len(day_series)

        if day_series.max() < params["min_power_threshold"]:
            continue

        active_areas = detect_active_areas_unfiltered(
            day_series,
            window_size=int(params["typical_duration"] * samples_per_hour(params) // 2),
            greater_than=params["min_power_threshold"] / 10,
        )
        for a in active_areas:
            all_active_areas.append((a[0], a[1] + 1))

    filtered_active_areas = []
    for start, end in all_active_areas:
        active_area_series = machine_series[start:end]
        if energy_from_series(active_area_series, params) < params["energy_min"] / 2:
            continue

        rated_splits = get_rated_active_area_splits(active_area_series, params)
        if len(rated_splits) == 0:
            continue
        i_best = np.argmin([rating for _, rating in rated_splits])
        split, rating = rated_splits[i_best]
        if not np.isfinite(rating):
            continue

        for i_start, i_end in split:
            filtered_active_areas.append((start + i_start, start + i_end,))

    return filtered_active_areas
