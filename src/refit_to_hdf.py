import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import itertools
from .refit import load_house_df, metadata

parser = argparse.ArgumentParser(
    description="Convert REFIT electricity consumption data and PVGIS simulated PV production data to HDF files with iHomeLab-like format"
)
parser.add_argument(
    "--input-csv-dir",
    type=Path,
    default=Path(__file__).parent / "../data/clean_refit",
    help="Path to the directory containing the CSV files CLEAN_House1.csv, CLEAN_House2.csv, etc. This is the extracted version of CLEAN_REFIT_081116.7z.",
)
parser.add_argument(
    "--input-solar-csv",
    type=Path,
    help="Path to the CSV file exported from PVGIS containing solar power data.",
)
parser.add_argument(
    "--input-solar-hdf",
    type=Path,
    default=Path(__file__).parent
    / "../data/simulated_pv_production/camborne_3480kwp.hdf",
    help="Path to the HDF file exported from simulate_pv_production.py containing solar power data.",
)
parser.add_argument(
    "--output-dir", type=Path, default=Path(__file__).parent / "../data/refit",
)
parser.add_argument(
    "--output-col-prefix",
    default="X_",
    help="Prefix added to the name of every appliance column in the generated HDF files. Useful for staying compatible with the format of the iHomeLab dataset.",
)
parser.add_argument(
    "--output-sample-rate",
    default="5T",
    help="Data is always resampled to this sample rate.",
)
parser.add_argument(
    "--output-file-suffix",
    default="",
    help="Suffix added to every output filename, right before the extension.",
)
args = parser.parse_args()


def load_solar_df_from_csv():
    with open(args.input_solar_csv, "r") as csv_file:
        header_line_index = None
        empty_line_index = None
        for i, line in enumerate(csv_file):
            if header_line_index is None:
                if line.strip() == "time,P,G(i),H_sun,T2m,WS10m,Int":
                    header_line_index = i
            elif not line.strip():
                empty_line_index = i
                break
        if header_line_index is None or empty_line_index is None:
            raise ValueError("header or footer not found in solar CSV")

    solar_df = pd.read_csv(
        args.input_solar_csv,
        skiprows=header_line_index,
        nrows=empty_line_index - header_line_index - 1,
    )

    solar_df = pd.DataFrame(
        {"exp_power": solar_df["P"].values,},
        index=pd.DatetimeIndex(
            pd.to_datetime(solar_df["time"], format="%Y%m%d:%H%M", utc=True).rename(
                None
            )
        ).tz_convert("Europe/London"),
    )

    return solar_df


def load_solar_df_from_hdf():
    solar_df = pd.read_hdf(args.input_solar_hdf).tz_convert("Europe/London")
    solar_df = solar_df[["exp_power"]]
    return solar_df


def load_solar_df():
    if bool(args.input_solar_csv is None) == bool(args.input_solar_hdf is None):
        raise ValueError(
            "exactly one of --input-solar-csv or --input-solar-hdf must be set"
        )
    if args.input_solar_csv:
        return load_solar_df_from_csv()
    else:
        return load_solar_df_from_hdf()


solar_df = load_solar_df()

args.output_dir.mkdir(exist_ok=True)

for house_number, appliances in tqdm(metadata.items()):
    csv_path = args.input_csv_dir / f"CLEAN_House{house_number}.csv"
    house_df = load_house_df(csv_path, appliances)

    joined_df = house_df.join(solar_df, how="outer").add_prefix(args.output_col_prefix)
    joined_df = joined_df.resample(args.output_sample_rate).mean()
    joined_df = joined_df["2014-05-03":"2015-05-02"]
    joined_df = joined_df.dropna()
    if joined_df.empty:
        raise ValueError("joined_df is empty")

    filename = (
        f"house_{house_number}_{args.output_sample_rate}{args.output_file_suffix}.hdf"
    )
    joined_df.to_hdf(
        args.output_dir / filename,
        key="/data",
        complevel=9,
        complib="blosc:snappy",
        format="table",
    )
