import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import itertools
import glob
import re
from .refit import load_house_df, metadata

parser = argparse.ArgumentParser(
    description="Calculate total and per-appliance electricity consumption from REFIT CSVs"
)
parser.add_argument(
    "--input-csv-dir",
    type=Path,
    required=True,
    help="Path to the directory containing the CSV files CLEAN_House1.csv, CLEAN_House2.csv, etc. This is the extracted version of CLEAN_REFIT_081116.7z.",
)
parser.add_argument(
    "--output-csv",
    type=Path,
    default=Path(__file__).parent / "../data/refit_absolute_consumption.csv",
    help="Path where to write output CSV file",
)
args = parser.parse_args()


def house_number_from_path(csv_path):
    m = re.match(r"^CLEAN_House(\d+).csv$", Path(csv_path).name)
    if m is None:
        raise ValueError("unsupported filename format")
    return int(m[1])


output_df = {}
for csv_path in tqdm(glob.glob(str(args.input_csv_dir / "CLEAN_House*.csv"))):
    house_number = house_number_from_path(csv_path)
    metadata_entry = metadata.get(house_number, None)
    if metadata_entry is None:
        continue
    house_df = load_house_df(csv_path, metadata_entry)
    output_df[house_number] = house_df.resample("H").mean().resample("D").sum().mean()
output_df = pd.concat(output_df, names=["house"]).unstack()

output_df.to_csv(str(args.output_csv))
