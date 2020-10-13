import pandas as pd
import numpy as np
import os
import re
import sys
from pathlib import Path
from .run_detection import detect_active_areas
from .util import grouper
import argparse


def calc_self_consumption(
    df, consumed_column="total_cons_power", produced_column="exp_power"
):
    return (
        100
        * df[[consumed_column, produced_column]].min(axis=1).sum()
        / df[produced_column].sum()
    )


def apply_shift(df, shift_df, total_column, appliance_column):
    appliance_values = df[appliance_column].to_numpy(copy=True)
    total_values = df[total_column].to_numpy(copy=True)

    def assert_index_in_range(i):
        if i < 0 or i >= len(appliance_values):
            raise RuntimeError(f"shifted index ({i}) out of range ")

    for row in shift_df.itertuples():
        if not row.shift_periods:
            continue

        i1 = row.start
        j1 = row.end

        shifted_values = appliance_values[i1:j1].copy()
        total_values[i1:j1] -= shifted_values
        appliance_values[i1:j1] = 0

        i2 = i1 + row.shift_periods
        assert_index_in_range(i2)
        j2 = j1 + row.shift_periods
        assert_index_in_range(j2)

        appliance_values[i2:j2] += shifted_values
        total_values[i2:j2] += shifted_values

    df = df.copy()
    df[appliance_column] = appliance_values
    df[total_column] = total_values
    return df


def map_columns(raw_input_df):
    column_mapping = {}
    for c in raw_input_df.columns:
        if c in column_mapping:
            continue
        m = re.match(r"^[A-Z]_(\w+)$", c)
        if m is None:
            raise RuntimeError(
                "Unsupported column name format {}. You will have to map the column manually.".format(
                    c
                )
            )
        if m[1] == "prod_power":
            column_mapping[c] = "exp_power"
        else:
            column_mapping[c] = m[1]
    input_df = raw_input_df.rename(columns=column_mapping)
    return input_df


def get_shiftable_appliances(input_df):
    return sorted(
        {"dishwasher_power", "washing_machine_power", "tumble_dryer_power"}
        & set(input_df.columns)
    )


def calc_self_consumption_bound_ges(
    df,
    appliance_columns,
    consumed_column="total_cons_power",
    produced_column="exp_power",
):
    df = df.copy()
    consumed_by_appliances = df[appliance_columns].sum().sum()
    df[consumed_column] -= df[appliance_columns].sum(axis=1)
    df[appliance_columns] = 0
    return (
        df[[consumed_column, produced_column]].min(axis=1).sum()
        + consumed_by_appliances
    ) / df[produced_column].sum()


def calc_self_consumption_bound(
    df,
    appliance_column,
    consumed_column="total_cons_power",
    produced_column="exp_power",
):
    """
    Assumes that all of the appliance power is compensable with energy out of PV
    """
    df = df.copy()
    consumed_by_appliance = df[appliance_column].sum()
    df[consumed_column] -= df[appliance_column]
    df[appliance_column] = 0
    return (
        df[[consumed_column, produced_column]].min(axis=1).sum() + consumed_by_appliance
    ) / df[produced_column].sum()


def calc_self_consumption_bound_per_day(
    df,
    appliance_column,
    consumed_column="total_cons_power",
    produced_column="exp_power",
):
    """
    Compensates only the energy of the appliances on a day where the PV has overproduction
    """
    input_mod = df.copy()
    input_mod[consumed_column] -= input_mod[appliance_column]
    unused_pv = input_mod[produced_column] - input_mod[consumed_column]
    unused_pv.loc[unused_pv < 0] = 0
    unused_pv = (
        unused_pv.resample("D").sum() - input_mod[appliance_column].resample("D").sum()
    )
    unused_pv.loc[unused_pv < 0] = 0
    return 100 * (1 - (unused_pv.sum() / input_mod[produced_column].sum()))


def get_naive_forward_shifts(input_df, washes_df):
    old_start = input_df.index[washes_df.start].to_series().reset_index(drop=True)
    new_start = np.maximum(
        old_start, old_start.dt.floor("D") + pd.Timedelta(12, "hours")
    )
    shift = (new_start - old_start) / input_df.index.freq
    shift = np.minimum(washes_df.trailing_gap, shift)
    return np.floor(shift).astype("int32")


def shift_for_max_self_consumption(unused_power, shiftable_power):
    if len(unused_power) < len(shiftable_power):
        raise ValueError("unused_power must be at least as long as shiftable_power")
    if len(unused_power) == len(shiftable_power):
        return 0
    return np.argmax(
        [
            np.minimum(
                unused_power[i : i + len(shiftable_power)], shiftable_power
            ).sum()
            for i in range(len(unused_power) - len(shiftable_power))
        ]
    )


def get_educated_forward_shifts(
    df,
    actions_df,
    appliance_column,
    total_column="total_cons_power",
    exp_column="exp_power",
):
    unused_power = np.maximum(
        0, df[exp_column].values - df[total_column].values + df[appliance_column].values
    )
    shiftable_power = df[appliance_column].values
    max_shift = int(np.floor(pd.Timedelta("24H") / df.index.freq))
    return pd.Series(
        [
            shift_for_max_self_consumption(
                unused_power[row.start : row.end + min(row.trailing_gap, max_shift)],
                shiftable_power[row.start : row.end],
            )
            for row in actions_df.itertuples()
        ]
    )


def prepare_result(result, house_appliance):
    result_df = pd.DataFrame(result).transpose()
    result_df.columns = pd.MultiIndex.from_tuples(house_appliance)
    result_df["row_names"] = [
        "before",
        "naive",
        "educated_forward",
        "upper_bound",
        "produced_energy",
        "days",
    ]
    result_df = result_df.set_index("row_names")
    result_df = result_df.T
    return result_df


def calc_load_shifting_potential(data_dir, frequency_string, param_df):
    house_appliance = []  # tuple (house-name,appliance)
    result = []  # array [before,improved,upper_bound]

    for file in os.listdir(data_dir):
        m = re.match(r"^house_(\d+)_" + frequency_string + ".hdf$", file)
        if m is None:
            continue
        house = int(m[1])

        print(file)

        # load data
        data_path = os.path.join(data_dir, file)
        raw_input_df = pd.read_hdf(data_path)
        input_df = map_columns(raw_input_df)

        input_df = input_df.asfreq(frequency_string)
        number_of_days = (~input_df.isna().any(axis=1)).resample("D").mean().sum()
        produced_energy = (
            input_df.exp_power.resample("H").mean().sum()
        )  # number of Wh produced during whole dataset

        shiftable_appliances = get_shiftable_appliances(input_df)
        if len(shiftable_appliances) == 0:
            print("no shiftabple appliances\n")
            continue

        for j, appliance_column in enumerate(shiftable_appliances):
            m = re.match(r"^(\w+)_power$", appliance_column)
            if m is None:
                raise ValueError(
                    f"unsupported appliance_column format ({appliance_column})"
                )
            machine = m[1]

            params = param_df[(param_df.house == house) & (param_df.machine == machine)]
            if len(params) != 1:
                raise ValueError(f"No parameters defined for house {house} {machine}")
            params = params.to_dict("records")[0]
            params["frequency_string"] = frequency_string

            # calculate active areas
            all_appliance_active_areas = detect_active_areas(
                input_df[appliance_column], params
            )
            print(
                "Appliance {} used {} times in {} days".format(
                    appliance_column,
                    len(all_appliance_active_areas),
                    (input_df.index[-1] - input_df.index[0]).days,
                )
            )

            # get start, end and gap
            temp = (x for a in all_appliance_active_areas for x in a)
            next(temp)
            washes_df = pd.DataFrame(
                data={
                    "start": (a for a, b in all_appliance_active_areas[:-1]),
                    "end": (b for a, b in all_appliance_active_areas[:-1]),
                    "trailing_gap": (
                        b - a for a, b in grouper(temp, n=2) if b is not None
                    ),
                }
            )

            # calculate shift
            shift_df_naive = washes_df.assign(
                shift_periods=get_naive_forward_shifts(input_df, washes_df)
            )
            shift_df_educated_forward = washes_df.assign(
                shift_periods=get_educated_forward_shifts(
                    input_df, washes_df, appliance_column
                )
            )
            print(
                "{} usages will be shifted (naive)".format(
                    (shift_df_naive.shift_periods > 0).sum()
                )
            )
            print(
                "{} usages will be shifted (educated_forward)".format(
                    (shift_df_educated_forward.shift_periods > 0).sum()
                )
            )

            # shift loads
            modified_df_naive = apply_shift(
                input_df, shift_df_naive, "total_cons_power", appliance_column
            )
            modified_df_educated_forward = apply_shift(
                input_df,
                shift_df_educated_forward,
                "total_cons_power",
                appliance_column,
            )

            house_appliance.append((file, appliance_column))
            result.append(
                [
                    calc_self_consumption(input_df),
                    calc_self_consumption(modified_df_naive),
                    calc_self_consumption(modified_df_educated_forward),
                    calc_self_consumption_bound_per_day(input_df, appliance_column),
                    produced_energy,
                    number_of_days,
                ]
            )

        print()

    return prepare_result(result, house_appliance)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simulate load shifting and calculate PV self-consumption"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        help="Path to the directory containing files like house_3_5T.hdf generated by refit_to_hdf.py",
        default=Path(__file__).parent / "../data/refit",
    )
    parser.add_argument(
        "--params-file",
        type=Path,
        help="Path to CSV file containing paramters for activation detection",
        default=Path(__file__).parent / "../refit_thresholds.csv",
    )
    parser.add_argument(
        "--sample-rate",
        type=str,
        help="Pandas frequency string which matches the desired input files",
        required=True,
    )
    parser.add_argument(
        "--output-filename-base",
        type=str,
        help="First part of the output filename (see --output-dir)",
        default="refit_bsrn",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Path to the directory where output file (<output-filename-base>_<sample-rate>.csv)",
        default=Path(__file__).parent / "../results",
    )
    args = parser.parse_args()

    param_df = pd.read_csv(str(args.params_file))
    param_df = param_df[~param_df.min_power_threshold.isna()]

    result_df_all = calc_load_shifting_potential(
        str(args.input_dir), args.sample_rate, param_df
    )
    result_df_all.to_csv(
        args.output_dir / f"{args.output_filename_base}_{args.sample_rate}.csv"
    )
