import pandas as pd

# Metadata based on https://pure.strath.ac.uk/ws/portalfiles/portal/62090183/CLEAN_READ_ME_081116.txt
metadata = {
    1: {"tumble_dryer": 4, "washing_machine": 5, "dishwasher": 6},
    2: {"washing_machine": 2, "dishwasher": 3},
    # 3: {"tumble_dryer": 4, "washing_machine": 6, "dishwasher": 5}, # Aggregate measurement not usable, see below
    4: {"washing_machine": 4, "washing_machine_2": 5},
    5: {"tumble_dryer": 2, "washing_machine": 3, "dishwasher": 4},
    6: {"washing_machine": 2, "dishwasher": 3},
    7: {"tumble_dryer": 4, "washing_machine": 5, "dishwasher": 6},
    8: {"tumble_dryer": 3, "washing_machine": 4},
    9: {"tumble_dryer": 2, "washing_machine": 3, "dishwasher": 4},
    10: {"washing_machine": 5, "dishwasher": 6},
    # 11: {"washing_machine": 3, "dishwasher": 4},  # Dishwasher changed 2014-10-04 # Aggregate measurement not usable, see below
    13: {"washing_machine": 3, "dishwasher": 4},  # Washing machine changed 2015-03-25
    15: {"tumble_dryer": 2, "washing_machine": 3, "dishwasher": 4},
    16: {"washing_machine": 5, "dishwasher": 6},
    17: {"tumble_dryer": 3, "washing_machine": 4},
    18: {"tumble_dryer": 4, "washing_machine": 5, "dishwasher": 6},
    19: {"washing_machine": 2},
    20: {"tumble_dryer": 3, "washing_machine": 4, "dishwasher": 5},
    # 21: {"tumble_dryer": 2, "washing_machine": 3, "dishwasher": 4}, # Aggregate measurement not usable, see below
}

# WARNING Unusable aggregate measurement in houses 3, 11 and 21
# From the REFIT publication:
#   Six homes in the study had solar panels installed. In three cases rewiring
#   was done to remove the effect of solar panel generation (Houses 1, 6 & 7).
#   In the other three (Houses 3, 11 & 21) rewiring was not possible and the
#   aggregate of these houses was recorded as is with solar interference.


def load_house_df(csv_path, appliances):
    cols = ["Unix", "Aggregate"]
    for appliance_number in appliances.values():
        cols.append(f"Appliance{appliance_number}")
    house_df = pd.read_csv(csv_path, usecols=cols)

    house_df_data = {"total_cons_power": house_df["Aggregate"]}
    for appliance_name, appliance_number in appliances.items():
        house_df_data[f"{appliance_name}_power"] = house_df[
            f"Appliance{appliance_number}"
        ].values
    index = pd.DatetimeIndex(
        pd.to_datetime(house_df["Unix"], unit="s", utc=True).rename(None)
    ).tz_convert("Europe/London")
    house_df = pd.DataFrame(house_df_data).set_index(index)
    house_df.sort_index(inplace=True)

    return house_df
