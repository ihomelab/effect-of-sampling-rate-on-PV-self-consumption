import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import re

parser = argparse.ArgumentParser(
    description="Extract time series data for one appliance into a CSV file"
)
parser.add_argument(
    "--input-file",
    type=Path,
    default=Path(__file__).parent / "../data/published_data_vs01/data/dfA_300s.hdf",
)
parser.add_argument(
    "--appliance", default="dishwasher",
)
parser.add_argument(
    "--output-file",
    type=Path,
    default=Path(__file__).parent / "../data/appliance_csvs/A_dishwasher.csv",
)
parser.add_argument(
    "--write-index", action="store_true",
)
args = parser.parse_args()

df = pd.read_hdf(args.input_file)

column_mapping = {}
for c in df.columns:
    if c in column_mapping:
        continue
    m = re.match(r"^[A-Z]_(\w+)$", c)
    if m is None:
        raise RuntimeError(
            "Unsupported column name format {}. You will have to map the column manually.".format(
                c
            )
        )
    column_mapping[c] = m[1]
df.rename(columns=column_mapping, inplace=True)

input_array = df["{}_power".format(args.appliance)].to_numpy()
input_array = input_array[~np.isnan(input_array)]

args.output_file.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(input_array).to_csv(args.output_file, header=False, index=args.write_index)
print(args.output_file.resolve())
