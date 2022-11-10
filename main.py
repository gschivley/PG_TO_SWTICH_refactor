


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

from powergenome.conversion_functions import (
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



def fuel_files(
    fuel_prices: pd.DataFrame,
    planning_years: List[int],
    regions: List[str],
    fuel_region_map: Dict[str, List[str]],
    fuel_emission_factors: Dict[str, float],
    out_folder: Path,
):

    fuel_cost = switch_fuel_cost_table(
        fuel_region_map,
        fuel_prices,
        regions,
        scenario="reference",
        year_list=planning_years,
    )

    fuels_table = switch_fuels(fuel_prices, fuel_emission_factors)
    fuels_table.loc[len(fuels_table.index)] = [
        "Fuel",
        0,
        0,
    ]  # adding in a dummy fuel for regional_fuel_market

    fuel_cost.to_csv(out_folder / "fuel_cost.csv", index=False)
    fuels_table.to_csv(out_folder / "fuels.csv", index=False)


def gen_projects_info_file(
    fuel_prices: pd.DataFrame,
    # gc: GeneratorClusters,
    # pudl_engine: sa.engine,
    # settings_list: List[dict],
    # settings_file: str,
    complete_gens: pd.DataFrame, settings: dict, out_folder: Path
):

    if settings.get("cogen_tech"):
        cogen_tech = settings["cogen_tech"]
    else:
        cogen_tech = {
            "Onshore Wind Turbine": False,
            "Biomass": False,
            "Conventional Hydroelectric": False,
            "Conventional Steam Coal": False,
            "Natural Gas Fired Combined Cycle": False,
            "Natural Gas Fired Combustion Turbine": False,
            "Natural Gas Steam Turbine": False,
            "Nuclear": False,
            "Solar Photovoltaic": False,
            "Hydroelectric Pumped Storage": False,
            "Offshore Wind Turbine": False,
            "Small Hydroelectric": False,
            "NaturalGas_CCCCSAvgCF_Conservative": False,
            "NaturalGas_CCAvgCF_Moderate": False,
            "NaturalGas_CTAvgCF_Moderate": False,
            "Battery_*_Moderate": False,
            "NaturalGas_CCS100_Moderate": False,
            "heat_load_shifting": False,
        }
    if settings.get("baseload_tech"):
        baseload_tech = settings.get("baseload_tech")
    else:
        baseload_tech = {
            "Onshore Wind Turbine": False,
            "Biomass": False,
            "Conventional Hydroelectric": False,
            "Conventional Steam Coal": True,
            "Natural Gas Fired Combined Cycle": False,
            "Natural Gas Fired Combustion Turbine": False,
            "Natural Gas Steam Turbine": False,
            "Nuclear": True,
            "Solar Photovoltaic": False,
            "Hydroelectric Pumped Storage": False,
            "Offshore Wind Turbine": False,
            "Small Hydroelectric": False,
            "NaturalGas_CCCCSAvgCF_Conservative": False,
            "NaturalGas_CCAvgCF_Moderate": False,
            "NaturalGas_CTAvgCF_Moderate": False,
            "Battery_*_Moderate": False,
            "NaturalGas_CCS100_Moderate": False,
            "heat_load_shifting": False,
        }
    if settings.get("energy_tech"):
        energy_tech = settings["energy_tech"]
    else:
        energy_tech = {
            "Onshore Wind Turbine": "Wind",
            "Biomass": "Bio Solid",
            "Conventional Hydroelectric": "Water",
            "Conventional Steam Coal": "Coal",
            "Natural Gas Fired Combined Cycle": "Naturalgas",
            "Natural Gas Fired Combustion Turbine": "Naturalgas",
            "Natural Gas Steam Turbine": "Naturalgas",
            "Nuclear": "Uranium",
            "Solar Photovoltaic": "Solar",
            "Hydroelectric Pumped Storage": "Water",
            "Offshore Wind Turbine": "Wind",
            "Small Hydroelectric": "Water",
            "NaturalGas_CCCCSAvgCF_Conservative": "Naturalgas",
            "NaturalGas_CCAvgCF_Moderate": "Naturalgas",
            "NaturalGas_CTAvgCF_Moderate": "Naturalgas",
            "Battery_*_Moderate": "Electricity",
            "NaturalGas_CCS100_Moderate": "Naturalgas",
            "heat_load_shifting": False,
        }
    if settings.get("forced_outage_tech"):
        forced_outage_tech = settings["forced_outage_tech"]
    else:
        forced_outage_tech = {
            "Onshore Wind Turbine": 0.0,
            "Biomass": 0.04,
            "Conventional Hydroelectric": 0.05,
            "Conventional Steam Coal": 0.04,
            "Natural Gas Fired Combined Cycle": 0.4,
            "Natural Gas Fired Combustion Turbine": 0.4,
            "Natural Gas Steam Turbine": 0.4,
            "Nuclear": 0.04,
            "Solar Photovoltaic": 0.0,
            "Hydroelectric Pumped Storage": 0.05,
            "Offshore Wind Turbine": 0.05,
            "Small Hydroelectric": 0.05,
            "NaturalGas_CCCCSAvgCF_Conservative": 0.4,
            "NaturalGas_CCAvgCF_Moderate": 0.4,
            "NaturalGas_CTAvgCF_Moderate": 0.4,
            "Battery_*_Moderate": 0.02,
            "NaturalGas_CCS100_Moderate": 0.4,
            "heat_load_shifting": False,
        }
    if settings.get("sched_outage_tech"):
        sched_outage_tech = settings["sched_outage_tech"]
    else:
        sched_outage_tech = {
            "Onshore Wind Turbine": 0.0,
            "Biomass": 0.06,
            "Conventional Hydroelectric": 0.05,
            "Conventional Steam Coal": 0.06,
            "Natural Gas Fired Combined Cycle": 0.6,
            "Natural Gas Fired Combustion Turbine": 0.6,
            "Natural Gas Steam Turbine": 0.6,
            "Nuclear": 0.06,
            "Solar Photovoltaic": 0.0,
            "Hydroelectric Pumped Storage": 0.05,
            "Offshore Wind Turbine": 0.01,
            "Small Hydroelectric": 0.05,
            "NaturalGas_CCCCSAvgCF_Conservative": 0.6,
            "NaturalGas_CCAvgCF_Moderate": 0.6,
            "NaturalGas_CTAvgCF_Moderate": 0.6,
            "Battery_*_Moderate": 0.01,
            "NaturalGas_CCS100_Moderate": 0.6,
            "heat_load_shifting": False,
        }

    gen_project_info = generation_projects_info(
        complete_gens,
        settings.get("transmission_investment_cost")["spur"]["capex_mw_mile"],
        settings.get("retirement_ages"),
        cogen_tech,
        baseload_tech,
        energy_tech,
        sched_outage_tech,
        forced_outage_tech,
    )

    gen_type_tech = {'Onshore Wind Turbine':'Wind', 'Biomass': 'Biomass', 'Conventional Hydroelectric':'Hydro', 
                'Conventional Steam Coal': 'Coal', 'Natural Gas Fired Combined Cycle':'Gas', 
                 'Natural Gas Fired Combustion Turbine':'Gas', 'Natural Gas Steam Turbine':'Gas', 
                'Nuclear':'Nuclear', 'Solar Photovoltaic':'Solar', 'Hydroelectric Pumped Storage':'Hydro', 
                'Offshore Wind Turbine':'Wind', 'NaturalGas_CCCCSAvgCF_Conservative':'Naturalgas', 
                'NaturalGas_CCAvgCF_Moderate':'Naturalgas', 'NaturalGas_CTAvgCF_Moderate':'Naturalgas', 
                 'Battery_*_Moderate':'Storage', 'NaturalGas_CCS100_Moderate':'Naturalgas'}

    gen_tech = gen_project_info['gen_tech'].unique()
    graph_tech_types_table = pd.DataFrame(columns=['map_name', 'gen_type', 'gen_tech', 'energy_source'])
    graph_tech_types_table['gen_tech'] = gen_tech
    graph_tech_types_table['energy_source'] = graph_tech_types_table['gen_tech'].apply(lambda x: energy_tech[x])
    graph_tech_types_table['map_name'] = 'default'
    graph_tech_types_table['gen_type'] = graph_tech_types_table['gen_tech'].apply(lambda x: gen_type_tech[x])

    # settings = load_settings(path=settings_file)
    # pudl_engine, pudl_out, pg_engine = init_pudl_connection(
    #     freq="AS",
    #     start_year=min(settings.get("data_years")),
    #     end_year=max(settings.get("data_years")),
    # )

    # gc = GeneratorClusters(pudl_engine, pudl_out, pg_engine, settings_list[0])    
    # fuel_prices = gc.fuel_prices
    fuels = fuel_prices['fuel'].unique()
    fuels = [fuel.capitalize() for fuel in fuels]
    non_fuel_table = graph_tech_types_table[~graph_tech_types_table['energy_source'].isin(fuels)]
    non_fuel_energy = list(set(non_fuel_table['energy_source'].to_list()))
    # non_fuel_energy_table = pd.DataFrame(non_fuel_energy, columns=['energy_source'])


    gen_project_info['gen_full_load_heat_rate'] = gen_project_info.apply(
                lambda row: '.' if row.gen_energy_source in non_fuel_energy else row.gen_full_load_heat_rate,  axis=1)


    # Do I need to set full load heat rate to "." for non-fuel energy generators?
    gen_project_info.to_csv(out_folder / "generation_projects_info.csv", index=False)


def gen_prebuild_newbuild_info_files(
    gc: GeneratorClusters,
    pudl_engine: sa.engine,
    settings_list: List[dict],
    out_folder: Path,
    pg_engine: sa.engine   
):
    out_folder.mkdir(parents=True, exist_ok=True)
    settings = settings_list[0]
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

    gen_buildpre, gen_build_with_id = gen_build_predetermined(
        all_gen,
        pudl_gen,
        pudl_gen_entity,
        pg_build,
        {},  # manual_build_yr,
        eia_Gen,
        eia_Gen_prop,
        {},  # plant_gen_manual,
        {},  # plant_gen_manual_proposed,
        {},  # plant_gen_manual_retired,
        retirement_ages,
    )

    retired = gen_build_with_id.loc[
        gen_build_with_id["retirement_year"] < settings["model_year"], :
    ]
    retired_ids = retired["GENERATION_PROJECT"].to_list()

    # newbuild options
    df_list = []
    for settings in settings_list:
        gc.settings = settings
        new_gen = gc.create_new_generators()
        new_gen["build_year"] = settings["model_year"]
        df_list.append(new_gen)

    newgens = pd.concat(df_list, ignore_index=True)

    build_yr_list = gen_build_with_id["build_year"].to_list()
    # using gen_build_with_id because it has plants that were removed for the final gen_build_pred. (ie. build year=2020)
    gen_project = gen_build_with_id["GENERATION_PROJECT"].to_list()
    build_yr_plantid_dict = dict(zip(gen_project, build_yr_list))

    gen_build_costs = gen_build_costs_table(
        existing_gen, newgens, build_yr_plantid_dict, all_gen
    )

    gen_build_costs.drop(
        gen_build_costs[gen_build_costs["GENERATION_PROJECT"].isin(retired_ids)].index,
        inplace=True,
    )
    # drop retired plants
    gen_buildpre.drop(
        gen_buildpre[gen_buildpre["GENERATION_PROJECT"].isin(retired_ids)].index,
        inplace=True,
    )

    # Create a complete list of existing and new-build options
    complete_gens = pd.concat([existing_gen, newgens]).drop_duplicates(
        subset=["Resource"]
    )
    complete_gens = add_misc_gen_values(complete_gens, gc.settings)
    gen_projects_info_file(gc.fuel_prices,complete_gens, gc.settings, out_folder)

    ### edit by RR
    load_curves = make_final_load_curves(pg_engine, settings_list[0])
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
    vcf = variable_capacity_factors_table(all_gen_variability, year_hour, timepoints_dict, all_gen)
    
    loads.to_csv(out_folder / "loads.csv", index=False)
    vcf.to_csv(out_folder / "variable_capacity_factors.csv", index=False)
    ###

    gen_buildpre.to_csv(out_folder / "gen_build_predetermined.csv", index=False)
    gen_build_costs.to_csv(out_folder / "gen_build_costs.csv", index=False)


def main(settings_file: str, results_folder: str):
    """Create inputs for the Switch model using PowerGenome data

    Parameters
    ----------
    settings_file : str
        The path to a YAML file or folder of YAML files with settings parameters
    results_folder : str
        The folder where results will be saved
    """
    cwd = Path.cwd()
    out_folder = cwd / results_folder
    out_folder.mkdir(exist_ok=True)
    # Load settings, create db connections, and build dictionary of settings across
    # cases/years

    settings = load_settings(path=settings_file)
    pudl_engine, pudl_out, pg_engine = init_pudl_connection(
        freq="AS",
        start_year=min(settings.get("data_years")),
        end_year=max(settings.get("data_years")),
    )
    check_settings(settings, pg_engine)
    input_folder = cwd / settings["input_folder"]
    settings["input_folder"] = input_folder
    scenario_definitions = pd.read_csv(
        input_folder / settings["scenario_definitions_fn"]
    )
    scenario_settings = build_scenario_settings(settings, scenario_definitions)

    # Should switch the case_id/year layers in scenario settings dictionary.
    # Run through the different cases and save files in a new folder for each.
    for case_id in scenario_definitions["case_id"].unique():
        print(f"starting case {case_id}")
        case_folder = out_folder / case_id
        case_folder.mkdir(parents=True, exist_ok=True)

        settings_list = []
        case_years = []
        for year in scenario_definitions.query("case_id == @case_id")["year"]:
            case_years.append(year)
            settings_list.append(scenario_settings[year][case_id])

        gc = GeneratorClusters(pudl_engine, pudl_out, pg_engine, settings_list[0])
        gen_prebuild_newbuild_info_files(gc, pudl_engine, settings_list, case_folder, pg_engine)
        fuel_files(
            fuel_prices=gc.fuel_prices,
            planning_years=case_years,
            regions=settings["model_regions"],
            fuel_region_map=settings["aeo_fuel_region_map"],
            fuel_emission_factors=settings["fuel_emission_factors"],
            out_folder=case_folder,
        )




if __name__ == "__main__":
    typer.run(main)