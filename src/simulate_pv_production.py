import pandas as pd
import numpy as np
import random
from tqdm import tqdm
import itertools
import os
from pathlib import Path
import pvlib
import glob
import re


def load_single_bsrn_df(path):
    with open(path, "r") as csv_file:
        comment_line_count = None
        for i, line in enumerate(csv_file):
            if i == 0:
                if not line.strip().startswith("/*"):
                    raise ValueError("expected comment starting on first line")
            elif line.strip().endswith("*/"):
                comment_line_count = i + 1
                break
        if comment_line_count is None:
            raise ValueError("end of comment not found")

    df = pd.read_csv(
        path, index_col=0, skiprows=comment_line_count, sep="\t", parse_dates=True,
    )
    df = df.tz_localize("UTC").tz_convert("Europe/London")
    df = df.rename(
        columns={"SWD [W/m**2]": "swd", "DIR [W/m**2]": "dir", "DIF [W/m**2]": "dif",}
    )[["swd", "dir", "dif"]]
    return df


def load_single_weather_df(path):
    with open(path, "r") as csv_file:
        skip_line_count = None
        last_line_index = None
        for i, line in enumerate(csv_file):
            if line.strip() == "data":
                if skip_line_count is not None:
                    raise ValueError("duplicate start of data")
                skip_line_count = i + 1
            if line.strip() == "end data":
                if last_line_index is not None:
                    raise ValueError("duplicate end of data")
                last_line_index = i - 1
        if skip_line_count is None:
            raise ValueError("start of data not found")
        if last_line_index is None:
            raise ValueError("end of data not found")

    df = pd.read_csv(
        path,
        index_col=0,
        usecols=["ob_time", "wind_speed_unit_id", "wind_speed", "air_temperature"],
        skiprows=skip_line_count,
        nrows=last_line_index - skip_line_count,
        parse_dates=True,
    )
    df = df.tz_localize("UTC")
    df = df.dropna()

    if (df.wind_speed_unit_id != 4).any():
        raise ValueError("unsupported wind speed unit")
    df = df.drop(columns=["wind_speed_unit_id"])
    df.wind_speed *= 0.514444  # convert to m/s

    return df


def load_concatenated_bsrn_df(data_dir):
    parts = []
    for filename in glob.iglob(str(data_dir / "bsrn/cam*.csv")):
        if Path(filename).name == "cam0815.csv":
            # There is no direct irradiance measurement for this month
            continue
        parts.append(load_single_bsrn_df(filename))
    return pd.concat(parts).sort_index()


def load_concatenated_weather_df(data_dir):
    glob_suffix = "uk_met_office/midas-open_uk-hourly-weather-obs_dv-201908_cornwall_01395_camborne_qcv-1_*.csv"
    parts = []
    for filename in glob.iglob(str(data_dir / glob_suffix)):
        parts.append(load_single_weather_df(filename))
    return pd.concat(parts).sort_index()


def simulate_pv_production(data_dir):
    output_dir = data_dir / "simulated_pv_production"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "camborne_3480kwp.hdf"

    print("=== Loading data ===")

    bsrn_df = load_concatenated_bsrn_df(data_dir)
    weather_df = load_concatenated_weather_df(data_dir)

    pvlib_input_df = bsrn_df.dropna().join(weather_df).resample("T").interpolate()
    pvlib_input_df = pvlib_input_df.rename(
        columns={"dir": "dni", "swd": "ghi", "dif": "dhi"}
    )

    system = pvlib.pvsystem.PVSystem(
        orientation_strategy="south_at_latitude_tilt",
        module_parameters=pvlib.pvsystem.retrieve_sam("SandiaMod")[
            "LG_LG290N1C_G3__2013_"
        ],
        inverter_parameters=pvlib.pvsystem.retrieve_sam("CECInverter")[
            "Fronius_International_GmbH__Fronius_Primo_3_8_1_208_240__240V_"
        ],
        temperature_model_parameters=pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS[
            "sapm"
        ]["close_mount_glass_glass"],
        racking_model="close_mount",
        modules_per_string=6,
        strings_per_inverter=2,
    )

    # Camborne (NOT Cambourne)
    location = pvlib.location.Location(
        latitude=50.216660, longitude=-5.316660, tz="Europe/London", altitude=87,
    )

    model_chain = pvlib.modelchain.ModelChain(system, location,)
    print(model_chain)

    print("=== Simulating ===")

    model_chain.run_model(pvlib_input_df)

    output_df = pd.DataFrame(
        {
            "exp_power": model_chain.ac.fillna(0),
            "poa_irradiance": model_chain.total_irrad["poa_global"],
        }
    )
    print(output_df.describe())

    print("=== Saving output ===")
    output_df.to_hdf(output_path, "/data", complib="blosc:snappy", complevel=9)
    print(output_path)


if __name__ == "__main__":
    data_dir = (Path(__file__).parent / "../data").resolve()
    simulate_pv_production(data_dir)
