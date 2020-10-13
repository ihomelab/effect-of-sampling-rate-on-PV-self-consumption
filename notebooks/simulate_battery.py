import pandas as pd
import numpy as np
import os
import re
from tqdm import tqdm

def test_simulation(df, battary_df):
    compens_power =  battary_df['energy_flow'].clip(upper=0)*(-1)
    unused_pv = (df['exp_power'] - df['total_cons_power']).clip(lower=0)
    print('{}% of the unused PV-Power got accesabel'.format(np.round(compens_power.sum()/unused_pv.sum()*100,1)))
    print('{}% of the total energy usage got compensated'.format(np.round(compens_power.sum()/df['total_cons_power'].sum()*100,1)))


def battery_usage(df,max_energy,max_charge,phi_charge,max_drain,phi_drain):
    max_energy = max_energy/(5/60*1) # 5min*1h/60min
    energy = (df['exp_power'] - df['total_cons_power'])
    energy[energy>0] *= phi_charge
    energy[energy<0] /= phi_drain
    energy = energy.clip(upper=max_charge,lower=max_drain)
    for i in tqdm(range(len(energy))):
        energy[i] = max(0, min(energy[i] + energy[i-1], max_energy)) if i is not 0 else 0
    energy_flows = energy.diff()
    energy_flows[0] = 0
    energy_flows[energy_flows>0] /= phi_charge
    energy_flows[energy_flows<0] *= phi_drain
    out_df = pd.concat([energy*5/60,energy_flows],axis=1)
    out_df.columns = ['energy','energy_flow']
    return out_df


def remap_columns(raw_input_df):
    column_mapping = {}
    for c in raw_input_df.columns:
        if c in column_mapping:
            continue
        m = re.match(r'^[A-Z]_(\w+)$', c)
        if m is None:
            raise RuntimeError('Unsupported column name format {}. You will have to map the column manually.'.format(c))
        if m[1] == 'prod_power':
            column_mapping[c] = 'exp_power'
        else:
            column_mapping[c] = m[1]
    return raw_input_df.rename(columns=column_mapping)
    

def get_battery_simulation(data_dir,max_energy=16000,max_charge=2000,phi_charge=0.98,max_drain=-2000,phi_drain=0.98):
    simulations = []
    file_names = []
    for file in os.listdir(data_dir):
        print(file)
        input_df = remap_columns(pd.read_hdf(os.path.join(data_dir, file)))
        if 'batt_state' in input_df.columns:
            print("contains already a battery")
            continue
        battery = battery_usage(input_df,max_energy,max_charge,phi_charge,max_drain,phi_drain)
        simulations.append(battery)
        test_simulation(input_df,battery)
        file_names.append(file)
        print()
    return simulations,file_names