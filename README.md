# Effect of Sampling Rate on Photovoltaic Self-Consumption in Load Shifting Simulations

This repository contains the code used in the experiments for our paper [Effect of Sampling Rate on Photovoltaic Self-Consumption in Load Shifting Simulations](https://doi.org/10.3390/en13205393). The paper gives a detailed overview of our methodology, so we recommend reading that first. This readme explains the practical steps to reproduce our results.

## Install development tools

We developed this code in a Linux environment, so it's easiest if you use that. You can use [WSL 2](https://docs.microsoft.com/en-us/windows/wsl/wsl2-index) to get a Linux environment on Windows. You can also run our code natively on Windows or macOS, but it may require a few minor adjustments.

You should install the following tools before continuing. The latest versions should work fine, but we've added the ones we used for reference.

- [Python](https://www.python.org/downloads/) and pip (we used Python 3.6.9)
- [Node.js](https://nodejs.org/en/download/) (we used Node.js 12.18.0 LTS)
- [Yarn](https://classic.yarnpkg.com/en/docs/install) (we used Yarn 1.21.1)

Unless you want to manage your Python virtual environments using your favorite tool, you should also install the Python 3 `venv` module (`sudo apt-get install python3-venv` on Ubuntu).

## Install dependencies

The dependencies for our Jupyter notebooks and Python scripts can be installed as follows:

- Create a virtual environment
  - Run `python3 -m venv .venv`
- Activate the virtual environment
  - Run `source .venv/bin/activate`
  - You should see `(.venv)` or similar in your command line prompt
- Install dependencies from `requirements.txt`
  - `pip install -r requirements.txt`

`pangaea-scraper` has separate dependencies which have to be installed using Yarn. In the `pangaea-scraper` folder, run `yarn`.

## Create accounts

You will need a couple of accounts to get access to various input data. To get the Met Office weather data, you'll need an account for the [CEDA Archive](http://archive.ceda.ac.uk/). To get irradiance data, you'll need a ["read account" for the BSRN](https://bsrn.awi.de/data/data-retrieval-via-pangaea/).

## Download data

All input and output data is be saved in the `data` folder (on the same level as this readme file). You will need to create that folder first.

### UK Met Office weather

To download the UK Met Office weather files, open [this page on the CEDA Archive](http://data.ceda.ac.uk/badc/ukmo-midas-open/data/uk-hourly-weather-obs/dataset-version-201908/cornwall/01395_camborne/qc-version-1). Make sure you are logged in. Download the three files named like `midas-open_uk-hourly-weather-obs_dv-201908_cornwall_01395_camborne_qcv-1_YEAR.csv`, where `YEAR` is `2013`, `2014` and `2015`. Save these files in `data/uk_met_office` (you will need to create this folder).

### BSRN irradiance

Irradiance data from the BSRN is distributed in monthly files. To avoid downloading all those files by hand, we've written a small script. Follow the instructions in [`pangaea-scraper/README.md`](./pangaea-scraper/README.md) to run it. Once that's done, you should have files named like `data/bsrn/camMMYY.csv`.

### REFIT electrical load

We use the _cleaned_ electrical load data from the REFIT project. You should download `CLEAN_REFIT_081116.7z` from [this page](https://pureportal.strath.ac.uk/en/datasets/refit-electrical-load-measurements-cleaned). Extract that archive so that the files end up at paths like `data/clean_refit/CLEAN_House1.csv`.

## Run PV simulation

This step uses the irradiance data (and other weather data) to simulate the AC power output of a hypothetical PV system. It can be run with `python -m src.simulate_pv_production`. It will create the file `data/simulated_pv_production/camborne_3480kwp.hdf`.

## Run experiments

This process consists of two steps. You have to run both of them at each sample rate which you want to investigate. Sample periods are specified using the [syntax accepted by pandas.Timedelta](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Timedelta.html). Well use the sample rate `3T` (3 minute sample period) in these examples.

First, combine the PV production data (from the previous step) and electrical load data (from REFIT). This is done by running `python -m src.refit_to_hdf --output-sample-rate 3T`, replacing `3T` with your chosen sample rate. This will create files like `data/refit/house_1_3T.hdf`.

Second, perform load shifting and collect statistics. Do this using `python -m src.shift --sample-rate 3T`. This will create (or overwrite) a file like `results/refit_bsrn_3T.csv`. That script will also print some basic information about the shifting performed.

## Evaluate results

The values saved in `results/refit_bsrn_<sample_rate>.csv` are relative self-consumption (relative to PV produced energy). The column `produced_energy` is the total produced energy in Wh. The `days` column contains the number of days of valid data that were used for calculations (total length of all valid samples, not calendar days). The `educated_forward` algorithm is the one that is referred to as "optimal" in the paper.

We have written some Jupyter notebooks for various evaluations. Most of these were used in our early experiments and are not relevant any more. The only interesting notebook is `notebooks/load_shifting_potential_evaluate.ipynb`. It generates the plots that we used in the paper, along with a few others. You can view the notebook by running `jupyter lab`.
