


import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime as dt

from powergenome.resource_clusters import ResourceGroup
from pathlib import Path
import sqlalchemy as sa
import typer

import pandas as pd
from powergenome.fuels import fuel_cost_table
from powergenome.generators import GeneratorClusters
from powergenome.util import (
    build_scenario_settings,
    init_pudl_connection,
    load_settings,
    check_settings,
)
from powergenome.eia_opendata import fetch_fuel_prices
import geopandas as gpd
from powergenome.generators import *
from powergenome.external_data import (
    make_demand_response_profiles,
    make_generator_variability,
)
from powergenome.GenX import add_misc_gen_values
os.getcwd()

from conversion_functions import (
    switch_fuel_cost_table,
    switch_fuels,
    create_dict_plantgen,
    create_dict_plantpudl,
    plant_dict,
    plant_gen_id,
    plant_pudl_id,
    gen_build_predetermined,
    gen_build_costs_table,
    generation_projects_info,
    hydro_timeseries,
    load_zones_table,
    fuel_market_tables,
    timeseries,
    timepoints_table,
    hydro_timepoints_table,
    graph_timestamp_map_table,
    loads_table,
    variable_capacity_factors_table,
    transmission_lines_table,
    balancing_areas,
)

from powergenome.load_profiles import (
    make_load_curves, 
    add_load_growth, 
    make_final_load_curves, 
    make_distributed_gen_profiles,
)

if not sys.warnoptions:
    import warnings

    warnings.simplefilter("ignore")

cwd = Path.cwd()

settings_path = (
    cwd / "settings_TD_east.yml" 
)
settings = load_settings(settings_path)

pudl_engine, pudl_out, pg_engine = init_pudl_connection(
        freq="AS",
        start_year=min(settings.get("data_years")),
        end_year=max(settings.get("data_years")),)
check_settings(settings, pg_engine)
input_folder = cwd / settings["input_folder"]
settings["input_folder"] = input_folder
scenario_definitions = pd.read_csv(
    input_folder / settings["scenario_definitions_fn"])
scenario_settings = build_scenario_settings(settings, scenario_definitions)

gc = GeneratorClusters(pudl_engine, pudl_out, pg_engine, scenario_settings[2020]["p1"])


all_gen = gc.create_all_generators()
all_gen["plant_id_eia"] = all_gen["plant_id_eia"].astype("Int64")
existing_gen = all_gen.loc[
    all_gen["plant_id_eia"].notna(), :
]  # gc.create_region_technology_clusters()

data_years = gc.settings.get("data_years", [])
if not isinstance(data_years, list):
    data_years = [data_years]
data_years = [str(y) for y in data_years]
s = f"""
    SELECT
        "plant_id_eia",
        "generator_id",
        "operational_status",
        "retirement_date",
        "planned_retirement_date",
        "current_planned_operating_date"
    FROM generators_eia860
    WHERE strftime('%Y',report_date) in ({','.join('?'*len(data_years))})
    """
# generators_eia860 = pd.read_sql_table("generators_eia860", pudl_engine)
generators_eia860 = pd.read_sql_query(
    s,
    pudl_engine,
    params=data_years,
    parse_dates=[
        "planned_retirement_date",
        "retirement_date",
        "current_planned_operating_date",
    ],
)

generators_entity_eia = pd.read_sql_table("generators_entity_eia", pudl_engine)
# create copies of PUDL tables and filter to relevant columns
pudl_gen = generators_eia860.copy()
pudl_gen = pudl_gen[
    [
        "plant_id_eia",
        "generator_id",
        "operational_status",
        "retirement_date",
        "planned_retirement_date",
        "current_planned_operating_date",
    ]
]  #'utility_id_eia',

pudl_gen_entity = generators_entity_eia.copy()
pudl_gen_entity = pudl_gen_entity[
    ["plant_id_eia", "generator_id", "operating_date"]
]

eia_Gen = gc.operating_860m
eia_Gen = eia_Gen[
    [
        "utility_id_eia",
        "utility_name",
        "plant_id_eia",
        "plant_name",
        "generator_id",
        "operating_year",
        "planned_retirement_year",
    ]
]
eia_Gen = eia_Gen.loc[eia_Gen["plant_id_eia"].notna(), :]

# create identifier to connect to powergenome data
eia_Gen["plant_gen_id"] = (
    eia_Gen["plant_id_eia"].astype(str) + "_" + eia_Gen["generator_id"]
)

eia_Gen_prop = gc.proposed_gens.reset_index()
eia_Gen_prop = eia_Gen_prop[
    [
        # "utility_id_eia",
        # "utility_name",
        "plant_id_eia",
        # "plant_name",
        "generator_id",
        "planned_operating_year",
    ]
]
eia_Gen_prop = eia_Gen_prop.loc[eia_Gen_prop["plant_id_eia"].notna(), :]
eia_Gen_prop["plant_gen_id"] = (
    eia_Gen_prop["plant_id_eia"].astype(str) + "_" + eia_Gen_prop["generator_id"]
)

# create copies of potential_build_yr (powergenome)
pg_build = gc.units_model.copy()
pg_build = pg_build[
    [
        "plant_id_eia",
        "generator_id",
        "unit_id_pudl",
        "planned_operating_year",
        "planned_retirement_date",
        "operating_date",
        "operating_year",
        "retirement_year",
    ]
]

retirement_ages = settings.get("retirement_ages")

# add in the plant+generator ids to pg_build and pudl tables (plant_id_eia + generator_id)
pudl_gen = plant_gen_id(pudl_gen)
pudl_gen_entity = plant_gen_id(pudl_gen_entity)
pg_build = plant_gen_id(pg_build)

# add in the plant+pudl id to the all_gen and pg_build tables (plant_id_eia + unit_pudl_id)
pg_build = plant_pudl_id(pg_build)
all_gen = plant_pudl_id(all_gen)

load_curves = make_final_load_curves(pg_engine, scenario_settings[2020]["p1"])
timeseries_df = timeseries(load_curves, max_weight=20.2778, avg_weight=283.8889, ts_duration_of_tp=4, 
                        ts_num_tps=6)
timeseries_dates = timeseries_df['timeseries'].to_list()
timestamp_interval = ['00', '04', '08', '12','16', '20'] # should align with ts_duration_of_tp and ts_num_tps
timepoints_df = timepoints_table(timeseries_dates, timestamp_interval)
# create lists and dictionary for later use
timepoints_timestamp = timepoints_df['timestamp'].to_list() # timestamp list
timepoints_tp_id = timepoints_df['timepoint_id'].to_list() # timepoint_id list
timepoints_dict = dict(zip(timepoints_timestamp, timepoints_tp_id)) # {timestamp: timepoint_id}

period_list = ['2020', '2030', '2040','2050']
loads, loads_with_year_hour = loads_table(load_curves, timepoints_timestamp, timepoints_dict, period_list)
# for fuel_cost and regional_fuel_market issue
dummy_df = pd.DataFrame({'TIMEPOINT':timepoints_tp_id})
dummy_df.insert(0,'LOAD_ZONE','loadzone')
dummy_df.insert(2,'zone_demand_mw',0)
loads = loads.append(dummy_df)

year_hour = loads_with_year_hour['year_hour'].to_list()
all_gen_variability = make_generator_variability(all_gen)

v_capacity_factors = all_gen_variability.copy().transpose()
v_capacity_factors["GENERATION_PROJECT"] = v_capacity_factors.index
v_c_f = v_capacity_factors.melt(
    id_vars="GENERATION_PROJECT",
    var_name="year_hour",
    value_name="gen_max_capacity_factor",
)
# reduce variability to just the hours of the year that have a timepoint
v_c_f = v_c_f[v_c_f["year_hour"].isin(year_hour)]

mod_vcf = v_c_f.copy()
# get the dates from hour of the year
start = pd.to_datetime("2021-01-01 0:00")  # 2020 is a leap year
mod_vcf["date"] = mod_vcf["year_hour"].apply(
    lambda x: start + pd.to_timedelta(x, unit="H")
)
mod_vcf["reformat"] = mod_vcf["date"].apply(lambda x: x.strftime("%Y%m%d%H"))
mod_vcf["reformat"] = mod_vcf["reformat"].astype(str)
date_list = mod_vcf["reformat"].to_list()
# change 2021 to correct period year/decade
# to get the timestamp
updated_dates20 = ["2020" + x[4:] for x in date_list]
updated_dates30 = ["2030" + x[4:] for x in date_list]
updated_dates40 = ["2040" + x[4:] for x in date_list]
updated_dates50 = ["2050" + x[4:] for x in date_list]
mod_vcf1 = mod_vcf.copy()
mod_vcf2 = mod_vcf.copy()
mod_vcf3 = mod_vcf.copy()
mod_vcf4 = mod_vcf.copy()
mod_vcf1["timestamp"] = updated_dates20
mod_vcf2["timestamp"] = updated_dates30
mod_vcf3["timestamp"] = updated_dates40
mod_vcf4["timestamp"] = updated_dates50
# go from timestamp to timepoint
mod_vcf1["timepoint"] = mod_vcf1["timestamp"].apply(lambda x: timepoints_dict[x])
mod_vcf2["timepoint"] = mod_vcf2["timestamp"].apply(lambda x: timepoints_dict[x])
mod_vcf3["timepoint"] = mod_vcf3["timestamp"].apply(lambda x: timepoints_dict[x])
mod_vcf4["timepoint"] = mod_vcf4["timestamp"].apply(lambda x: timepoints_dict[x])
# get final columns
mod_vcf1.drop(["year_hour", "date", "reformat", "timestamp"], axis=1, inplace=True)
mod_vcf2.drop(["year_hour", "date", "reformat", "timestamp"], axis=1, inplace=True)
mod_vcf3.drop(["year_hour", "date", "reformat", "timestamp"], axis=1, inplace=True)
mod_vcf4.drop(["year_hour", "date", "reformat", "timestamp"], axis=1, inplace=True)
# bring all decades together
var_cap_fac = pd.concat([mod_vcf1, mod_vcf2, mod_vcf3, mod_vcf4], ignore_index=True)

# only get all_gen plants that are wind or solar
technology = all_gen["technology"].to_list()

def Filter(list1, list2):
    return [n for n in list1 if any(m in n for m in list2)]

wind_solar = set(Filter(technology, ["Wind", "Solar"]))
all_gen.loc[all_gen["technology"].isin(wind_solar), "gen_is_variable"] = True
all_gen = all_gen[all_gen["gen_is_variable"] == True]

# get the correct GENERATION_PROJECT instead of region_resource_cluster from variability table
# all_gen = all_gen.copy()
# all_gen["region_resource_cluster"] = (
#     all_gen["region"]
#     + "_"
#     + all_gen["Resource"]
#     + "_"
#     + all_gen["cluster"].astype(str)
# )
# all_gen["gen_id"] = all_gen.index
# all_gen_convert = dict(
#     zip(all_gen["region_resource_cluster"].to_list(), all_gen["gen_id"].to_list())
# )

# reg_res_cl = all_gen["region_resource_cluster"].to_list()
reg_res_cl = all_gen["index"].to_list()
all(isinstance(n, float) for n in reg_res_cl)

import math
reg_res_cl_copy =[str(i) for i in reg_res_cl]
all(isinstance(n, str) for n in reg_res_cl_copy)

reg_res_cl =[i[0:-2] for i in reg_res_cl_copy]

var_cap_fac = var_cap_fac[var_cap_fac["GENERATION_PROJECT"].isin(reg_res_cl)]

# var_cap_fac["GENERATION_PROJECT"] = var_cap_fac["GENERATION_PROJECT"].apply(
#     lambda x: all_gen_convert[x]
# )
# filter to final columns
var_cap_fac = var_cap_fac[
    ["GENERATION_PROJECT", "timepoint", "gen_max_capacity_factor"]
]
var_cap_fac["GENERATION_PROJECT"] = (
    var_cap_fac["GENERATION_PROJECT"] + 1
) 

vcf = variable_capacity_factors_table(all_gen_variability, year_hour, timepoints_dict, all_gen)