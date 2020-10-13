import numpy as np
import pandas as pd
import scipy.optimize
import sklearn.linear_model
from pathlib import Path


def calc_peak_power_from_day_of_year(radiation_array, radiation_day_of_year):
    def f(day_of_year, mean, amplitude, phase):
        return mean + amplitude * np.cos(day_of_year * 2 * np.pi / 365 + phase)

    f_params, _ = scipy.optimize.curve_fit(
        f, radiation_day_of_year, radiation_array.max(axis=1), p0=[800, 200, 3],
    )
    phase = f_params[2]
    del f_params

    regressor = sklearn.linear_model.RANSACRegressor()
    regressor.fit(
        np.cos(radiation_day_of_year * 2 * np.pi / 365 + phase).reshape((-1, 1)),
        radiation_array.max(axis=1),
    )
    f_params = [
        regressor.estimator_.intercept_,
        regressor.estimator_.coef_.item(),
        phase,
    ]
    f_params

    def peak_power_from_day_of_year(day_of_year):
        return f(day_of_year, *f_params)

    return peak_power_from_day_of_year


def convert_radiation_data(raw_series):
    freq = pd.infer_freq(raw_series.index)
    if freq != "5T":
        raise ValueError(f"Unexpected sample rate: {freq}")

    radiation_array = []
    radiation_day_of_year = []
    for time, group_df in raw_series.dropna().groupby(pd.Grouper(freq="1D")):
        if len(group_df) != 24 * 12:
            continue
        radiation_day_of_year.append(time.dayofyear)
        radiation_array.append(group_df.values)
    radiation_array = np.stack(radiation_array)
    radiation_day_of_year = np.array(radiation_day_of_year)
    radiation_array.shape, radiation_day_of_year.shape

    peak_power_from_day_of_year = calc_peak_power_from_day_of_year(
        radiation_array, radiation_day_of_year,
    )
    radiation_array /= peak_power_from_day_of_year(radiation_day_of_year)[:, np.newaxis]

    return radiation_array, radiation_day_of_year


def load_all_radiation_data(base_path):
    input_files = [
        ("dfA_300s.hdf", "A_exp_power"),
        ("dfB_300s.hdf", "B_pv_prod_power"),
        ("dfC_300s.hdf", "C_pv_prod_power"),
        ("dfD_300s.hdf", "D_exp_power"),
        ("dfE_300s.hdf", "E_prod_power"),
    ]

    combined_radiation_array = []
    combined_radiation_day_of_year = []

    for filename, column in input_files:
        df = pd.read_hdf(Path(base_path) / filename)

        radiation_array, radiation_day_of_year = convert_radiation_data(df[column])
        combined_radiation_array.append(radiation_array)
        combined_radiation_day_of_year.append(radiation_day_of_year)

    combined_radiation_array = np.concatenate(combined_radiation_array).astype(
        np.float32
    )
    combined_radiation_day_of_year = np.concatenate(
        combined_radiation_day_of_year
    ).astype(np.float32)

    return combined_radiation_array, combined_radiation_day_of_year
