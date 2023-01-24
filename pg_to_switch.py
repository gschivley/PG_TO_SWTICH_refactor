import os
import sys
import pandas as pd
import numpy as np
import math
from datetime import datetime as dt
import ast
import itertools
from statistics import mode


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
from powergenome.time_reduction import kmeans_time_clustering
from powergenome.eia_opendata import fetch_fuel_prices
import geopandas as gpd
from powergenome.generators import *
from powergenome.external_data import (
    make_demand_response_profiles,
    make_generator_variability,
)
from powergenome.GenX import add_misc_gen_values

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
    hydro_timepoints_table,
    graph_timestamp_map_table,
    loads_table,
    variable_capacity_factors_table,
    transmission_lines_table,
    balancing_areas,
    ts_tp_pg_kmeans,
    hydro_timepoints_pg_kmeans,
    hydro_timeseries_pg_kmeans,
    variable_cf_pg_kmeans,
    load_pg_kmeans,
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

    ### edit by RR
    IPM_regions = regions
    load_zones = load_zones_table(IPM_regions, zone_ccs_distance_km=0)
    # add in the dummy loadzone
    load_zones.loc[len(load_zones.index)] = [
        "loadzone",
        0,
        load_zones["zone_dbid"].max() + 1,
    ]
    load_zones.to_csv(out_folder / "load_zones.csv", index=False)

    regional_fuel_markets = pd.DataFrame(
        {"regional_fuel_market": "loadzone-Fuel", "fuel": "Fuel"}, index=[0]
    )
    regional_fuel_markets

    ### edited by RR. CHANGE COLUMN NAME from fuel to rfm.
    zone_regional_fm = pd.DataFrame(
        {"load_zone": "loadzone", "rfm": "loadzone-Fuel"}, index=[0]
    )
    zone_regional_fm
    # creating dummy values based on one load zone in REAM's input file
    # note:regional_fuel_market should align with the regional_fuel_market table.
    # TODO --RR
    fuel_supply_curves20 = pd.DataFrame(
        {
            "period": [2020, 2020, 2020, 2020, 2020, 2020],
            "tier": [1, 2, 3, 4, 5, 6],
            "unit_cost": [1.9, 4.0, 487.5, 563.7, 637.8, 816.7],
            "max_avail_at_cost": [651929, 3845638, 3871799, 3882177, 3889953, 3920836],
        }
    )
    fuel_supply_curves20.insert(0, "regional_fuel_market", "loadzone-Fuel")
    fuel_supply_curves30 = fuel_supply_curves20.copy()
    fuel_supply_curves30["period"] = 2030
    fuel_supply_curves40 = fuel_supply_curves20.copy()
    fuel_supply_curves40["period"] = 2040
    fuel_supply_curves50 = fuel_supply_curves20.copy()
    fuel_supply_curves50["period"] = 2050
    fuel_supply_curves = pd.concat(
        [
            fuel_supply_curves20,
            fuel_supply_curves30,
            fuel_supply_curves40,
            fuel_supply_curves50,
        ]
    )
    fuel_supply_curves

    regional_fuel_markets.to_csv(out_folder / "regional_fuel_markets.csv", index=False)
    zone_regional_fm.to_csv(
        out_folder / "zone_to_regional_fuel_market.csv", index=False
    )
    fuel_supply_curves.to_csv(out_folder / "fuel_supply_curves.csv", index=False)

    ###

    fuel_cost.to_csv(out_folder / "fuel_cost.csv", index=False)
    fuels_table.to_csv(out_folder / "fuels.csv", index=False)


def gen_projects_info_file(
    fuel_prices: pd.DataFrame,
    # gc: GeneratorClusters,
    # pudl_engine: sa.engine,
    # settings_list: List[dict],
    # settings_file: str,
    complete_gens: pd.DataFrame,
    settings: dict,
    out_folder: Path,
    gen_buildpre: pd.DataFrame,
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
            "OffShoreWind_Class1_Moderate_fixed_1": False,
            "Landbased Wind Turbine": False,
            "Small Hydroelectric": False,
            "NaturalGas_CCCCSAvgCF_Conservative": False,
            "NaturalGas_CCAvgCF_Moderate": False,
            "NaturalGas_CTAvgCF_Moderate": False,
            "Battery_*_Moderate": False,
            "NaturalGas_CCS100_Moderate": False,
            "heat_load_shifting": False,
            "UtilityPV_Class1_Moderate": False,
            "UtilityPV_Class1_Moderate_100": False,
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
            "OffShoreWind_Class1_Moderate_fixed_1": False,
            "Landbased Wind Turbine": False,
            "Small Hydroelectric": False,
            "NaturalGas_CCCCSAvgCF_Conservative": False,
            "NaturalGas_CCAvgCF_Moderate": False,
            "NaturalGas_CTAvgCF_Moderate": False,
            "Battery_*_Moderate": False,
            "NaturalGas_CCS100_Moderate": False,
            "heat_load_shifting": False,
            "UtilityPV_Class1_Moderate": False,
            "UtilityPV_Class1_Moderate_100": False,
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
            "OffShoreWind_Class1_Moderate_fixed_1": "Wind",
            "Landbased Wind Turbine": "Wind",  ## add by RR because run into an erro of KeyError: 'LandbasedWind_Class1_Moderate_'
            "LandbasedWind_Class1_Moderate": "Wind",  ## add by RR because run into an erro of KeyError: 'LandbasedWind_Class1_Moderate_'
            "Small Hydroelectric": "Water",
            "NaturalGas_CCCCSAvgCF_Conservative": "Naturalgas",
            "NaturalGas_CCAvgCF_Moderate": "Naturalgas",
            "NaturalGas_CTAvgCF_Moderate": "Naturalgas",
            "Battery_*_Moderate": "Electricity",
            "NaturalGas_CCS100_Moderate": "Naturalgas",
            "heat_load_shifting": False,
            "UtilityPV_Class1_Moderate": "Solar",
            "UtilityPV_Class1_Moderate_100": "Solar",
        }
    if settings.get("forced_outage_tech"):
        forced_outage_tech = settings["forced_outage_tech"]
    else:
        forced_outage_tech = {
            "Onshore Wind Turbine": 0.0,
            "Biomass": 0.04,
            "Conventional Hydroelectric": 0.05,
            "Conventional Steam Coal": 0.04,
            # "Natural Gas Fired Combined Cycle": 0.4,
            # "Natural Gas Fired Combustion Turbine": 0.4,
            # "Natural Gas Steam Turbine": 0.4,
            "Natural Gas Fired Combined Cycle": 0.04,
            "Natural Gas Fired Combustion Turbine": 0.04,
            "Natural Gas Steam Turbine": 0.04,
            "Nuclear": 0.04,
            "Solar Photovoltaic": 0.0,
            "Hydroelectric Pumped Storage": 0.05,
            "Offshore Wind Turbine": 0.05,
            "OffShoreWind_Class1_Moderate_fixed_1": 0.05,
            "Landbased Wind Turbine": 0.05,
            "Small Hydroelectric": 0.05,
            # "NaturalGas_CCCCSAvgCF_Conservative": 0.4,
            # "NaturalGas_CCAvgCF_Moderate": 0.4,
            # "NaturalGas_CTAvgCF_Moderate": 0.4,
            # "NaturalGas_CCS100_Moderate": 0.4,
            "NaturalGas_CCCCSAvgCF_Conservative": 0.04,
            "NaturalGas_CCAvgCF_Moderate": 0.04,
            "NaturalGas_CTAvgCF_Moderate": 0.04,
            "NaturalGas_CCS100_Moderate": 0.04,
            "Battery_*_Moderate": 0.02,
            "heat_load_shifting": False,
            "UtilityPV_Class1_Moderate": 0.0,
            "UtilityPV_Class1_Moderate_100": 0.0,
        }
    if settings.get("sched_outage_tech"):
        sched_outage_tech = settings["sched_outage_tech"]
    else:
        sched_outage_tech = {
            "Onshore Wind Turbine": 0.0,
            "Biomass": 0.06,
            "Conventional Hydroelectric": 0.05,
            "Conventional Steam Coal": 0.06,
            # "Natural Gas Fired Combined Cycle": 0.6,
            # "Natural Gas Fired Combustion Turbine": 0.6,
            # "Natural Gas Steam Turbine": 0.6,
            "Natural Gas Fired Combined Cycle": 0.06,
            "Natural Gas Fired Combustion Turbine": 0.06,
            "Natural Gas Steam Turbine": 0.06,
            "Nuclear": 0.06,
            "Solar Photovoltaic": 0.0,
            "Hydroelectric Pumped Storage": 0.05,
            "Offshore Wind Turbine": 0.01,
            "OffShoreWind_Class1_Moderate_fixed_1": 0.01,
            "Landbased Wind Turbine": 0.01,
            "Small Hydroelectric": 0.05,
            # "NaturalGas_CCCCSAvgCF_Conservative": 0.6,
            # "NaturalGas_CCAvgCF_Moderate": 0.6,
            # "NaturalGas_CTAvgCF_Moderate": 0.6,
            # "NaturalGas_CCS100_Moderate": 0.6,
            "NaturalGas_CCCCSAvgCF_Conservative": 0.06,
            "NaturalGas_CCAvgCF_Moderate": 0.06,
            "NaturalGas_CTAvgCF_Moderate": 0.06,
            "NaturalGas_CCS100_Moderate": 0.06,
            "Battery_*_Moderate": 0.01,
            "heat_load_shifting": False,
            "UtilityPV_Class1_Moderate": 0.0,
            "UtilityPV_Class1_Moderate_100": 0.0,
        }

    gen_project_info = generation_projects_info(
        complete_gens,
        settings.get("transmission_investment_cost")["spur"]["capex_mw_mile"],
        settings.get("retirement_ages"),
    )

    graph_tech_colors_data = {
        "gen_type": [
            "Biomass",
            "Coal",
            "Naturalgas",
            "Geothermal",
            "Hydro",
            "Nuclear",
            "Oil",
            "Solar",
            "Storage",
            "Waste",
            "Wave",
            "Wind",
            "Other",
        ],
        "color": [
            "green",
            "saddlebrown",
            "gray",
            "red",
            "royalblue",
            "blueviolet",
            "orange",
            "gold",
            "aquamarine",
            "black",
            "blue",
            "deepskyblue",
            "white",
        ],
    }
    graph_tech_colors_table = pd.DataFrame(graph_tech_colors_data)
    graph_tech_colors_table.insert(0, "map_name", "default")
    graph_tech_colors_table

    gen_type_tech = {
        "Onshore Wind Turbine": "Wind",
        "Biomass": "Biomass",
        "Conventional Hydroelectric": "Hydro",
        "Conventional Steam Coal": "Coal",
        "Natural Gas Fired Combined Cycle": "Naturalgas",
        "Natural Gas Fired Combustion Turbine": "Naturalgas",
        "Natural Gas Steam Turbine": "Naturalgas",
        "Nuclear": "Nuclear",
        "Solar Photovoltaic": "Solar",
        "Hydroelectric Pumped Storage": "Hydro",
        "Offshore Wind Turbine": "Wind",
        "OffShoreWind_Class1_Moderate_fixed_1": "Wind",
        "Landbased Wind Turbine": "Wind",  ## add by RR because run into an erro of KeyError: 'LandbasedWind_Class1_Moderate_'
        "LandbasedWind_Class1_Moderate": "Wind",
        "Small Hydroelectric": "Hydro",
        "NaturalGas_CCCCSAvgCF_Conservative": "Naturalgas",
        "NaturalGas_CCAvgCF_Moderate": "Naturalgas",
        "NaturalGas_CTAvgCF_Moderate": "Naturalgas",
        "Battery_*_Moderate": "Storage",
        "NaturalGas_CCS100_Moderate": "Naturalgas",
        "UtilityPV_Class1_Moderate": "Solar",
        "UtilityPV_Class1_Moderate_100": "Solar",
    }

    graph_tech_types_table = gen_project_info.drop_duplicates(subset="gen_tech")
    graph_tech_types_table["map_name"] = "default"
    graph_tech_types_table["energy_source"] = graph_tech_types_table[
        "gen_energy_source"
    ]

    cols = ["map_name", "gen_type", "gen_tech", "energy_source"]
    graph_tech_types_table = graph_tech_types_table[cols]

    # settings = load_settings(path=settings_file)
    # pudl_engine, pudl_out, pg_engine = init_pudl_connection(
    #     freq="AS",
    #     start_year=min(settings.get("data_years")),
    #     end_year=max(settings.get("data_years")),
    # )
    # gc = GeneratorClusters(pudl_engine, pudl_out, pg_engine, settings_list[0])
    # fuel_prices = gc.fuel_prices
    fuels = fuel_prices["fuel"].unique()
    fuels = [fuel.capitalize() for fuel in fuels]
    non_fuel_table = graph_tech_types_table[
        ~graph_tech_types_table["energy_source"].isin(fuels)
    ]
    non_fuel_energy = list(set(non_fuel_table["energy_source"].to_list()))
    non_fuel_energy_table = pd.DataFrame(non_fuel_energy, columns=["energy_source"])

    # non_fuel_energy_table = pd.DataFrame(non_fuel_energy, columns=['energy_source'])

    gen_project_info["gen_full_load_heat_rate"] = gen_project_info.apply(
        lambda row: "."
        if row.gen_energy_source in non_fuel_energy
        else row.gen_full_load_heat_rate,
        axis=1,
    )

    # Do I need to set full load heat rate to "." for non-fuel energy generators?
    graph_tech_colors_table.to_csv(out_folder / "graph_tech_colors.csv", index=False)
    graph_tech_types_table.to_csv(out_folder / "graph_tech_types.csv", index=False)
    non_fuel_energy_table.to_csv(
        out_folder / "non_fuel_energy_sources.csv", index=False
    )
    # change the gen_capacity_limit_mw for those from gen_build_predetermined
    gen_project_info_new = pd.merge(
        gen_project_info, gen_buildpre, how="left", on="GENERATION_PROJECT"
    )
    gen_project_info_new.loc[
        gen_project_info_new["gen_predetermined_cap"].notna(), "gen_capacity_limit_mw"
    ] = gen_project_info_new[["gen_capacity_limit_mw", "gen_predetermined_cap"]].max(
        axis=1
    )
    gen_project_info = gen_project_info_new.drop(
        ["build_year", "gen_predetermined_cap", "gen_predetermined_storage_energy_mwh"],
        axis=1,
    )
    # remove the duplicated GENERATION_PROJECT from generation_projects_info .csv, and aggregate the "gen_capacity_limit_mw"
    gen_project_info["total_capacity"] = gen_project_info.groupby(
        ["GENERATION_PROJECT"]
    )["gen_capacity_limit_mw"].transform("sum")
    gen_project_info = gen_project_info.drop(
        ["gen_capacity_limit_mw"],
        axis=1,
    )
    gen_project_info.rename(
        columns={"total_capacity": "gen_capacity_limit_mw"}, inplace=True
    )
    gen_project_info = gen_project_info.drop_duplicates(subset="GENERATION_PROJECT")
    gen_project_info.to_csv(out_folder / "generation_projects_info.csv", index=False)


def gen_prebuild_newbuild_info_files(
    gc: GeneratorClusters,
    pudl_engine: sa.engine,
    settings_list: List[dict],
    case_years: List,
    out_folder: Path,
    pg_engine: sa.engine,
    hydro_variability_new: pd.DataFrame,
):
    out_folder.mkdir(parents=True, exist_ok=True)
    settings = settings_list[0]
    all_gen = gc.create_all_generators()
    all_gen = add_misc_gen_values(all_gen, settings)
    all_gen["Resource"] = all_gen["Resource"].str.rstrip("_")
    all_gen["technology"] = all_gen["technology"].str.rstrip("_")
    # all_gen["plant_id_eia"] = all_gen["plant_id_eia"].astype("Int64")
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
            "unit_id_pg",
            "planned_operating_year",
            "planned_retirement_date",
            "operating_date",
            "operating_year",
            "retirement_year",
            settings.get("capacity_col", "capacity_mw"),
            "capacity_mwh",
            "technology",
        ]
    ]

    retirement_ages = settings.get("retirement_ages")

    row_list = []
    for row in all_gen.itertuples():
        if isinstance(row.plant_id_eia, list):
            for plant_id, unit_id in zip(row.plant_id_eia, row.unit_id_pg):
                new_row = row._replace(plant_id_eia=plant_id, unit_id_pg=unit_id)
                row_list.append(new_row)
        else:
            row_list.append(row)
    all_gen_units = pd.DataFrame(row_list)
    all_gen_units["plant_id_eia"] = all_gen_units["plant_id_eia"].astype("Int64")

    # add in the plant+generator ids to pg_build and pudl tables (plant_id_eia + generator_id)
    pudl_gen = plant_gen_id(pudl_gen)
    pudl_gen_entity = plant_gen_id(pudl_gen_entity)
    pg_build = plant_gen_id(pg_build)

    # add in the plant+pudl id to the all_gen and pg_build tables (plant_id_eia + unit_pudl_id)
    pg_build = plant_pudl_id(pg_build)
    all_gen_units = plant_pudl_id(all_gen_units)

    gen_buildpre, gen_build_with_id = gen_build_predetermined(
        all_gen_units,
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
        settings.get("capacity_col", "capacity_mw"),
    )

    retired = gen_build_with_id.loc[
        gen_build_with_id["retirement_year"] < settings["model_year"], :
    ]
    retired_ids = retired["GENERATION_PROJECT"].to_list()

    # newbuild options
    df_list = []
    planning_periods = []
    planning_period_start_yrs = []
    for settings in settings_list:
        gc.settings = settings
        new_gen = gc.create_new_generators()
        new_gen["Resource"] = new_gen["Resource"].str.rstrip("_")
        new_gen["technology"] = new_gen["technology"].str.rstrip("_")
        new_gen["build_year"] = settings["model_year"]
        new_gen["GENERATION_PROJECT"] = new_gen[
            "Resource"
        ]  # + f"_{settings['model_year']}"
        df_list.append(new_gen)
        planning_periods.append(settings["model_year"])
        planning_period_start_yrs.append(settings["model_first_planning_year"])

    newgens = pd.concat(df_list, ignore_index=True)

    build_yr_list = gen_build_with_id["build_year"].to_list()
    # using gen_build_with_id because it has plants that were removed for the final gen_build_pred. (ie. build year=2020)
    gen_project = gen_build_with_id["GENERATION_PROJECT"].to_list()
    build_yr_plantid_dict = dict(zip(gen_project, build_yr_list))

    gen_build_costs = gen_build_costs_table(gen_buildpre, newgens)

    # gen_build_costs.drop(
    #     gen_build_costs[gen_build_costs["GENERATION_PROJECT"].isin(retired_ids)].index,
    #     inplace=True,
    # )
    # drop retired plants
    # gen_buildpre.drop(
    #     gen_buildpre[gen_buildpre["GENERATION_PROJECT"].isin(retired_ids)].index,
    #     inplace=True,
    # )

    # Create a complete list of existing and new-build options
    complete_gens = pd.concat([existing_gen, newgens]).drop_duplicates(
        subset=["Resource"]
    )
    complete_gens = add_misc_gen_values(complete_gens, settings)
    gen_projects_info_file(
        gc.fuel_prices, complete_gens, gc.settings, out_folder, gen_buildpre
    )

    ### edit by RR
    load_curves = make_final_load_curves(pg_engine, settings_list[0])
    all_gen_variability = make_generator_variability(all_gen)
    all_gen_variability.columns = all_gen["Resource"]

    if settings.get("reduce_time_domain") is True:
        for p in ["time_domain_periods", "time_domain_days_per_period"]:
            assert p in settings.keys()

        # results is a dict with keys "resource_profiles" (gen_variability), "load_profiles",
        # "time_series_mapping" (maps clusters sequentially to potential periods in year),
        # "ClusterWeights", etc. See PG for full details.
        results, representative_point, weights = kmeans_time_clustering(
            resource_profiles=all_gen_variability,
            load_profiles=load_curves,
            days_in_group=settings["time_domain_days_per_period"],
            num_clusters=settings["time_domain_periods"],
            include_peak_day=settings.get("include_peak_day", True),
            load_weight=settings.get("demand_weight_factor", 1),
            variable_resources_only=settings.get("variable_resources_only", True),
        )

        load_curves = results["load_profiles"]
        all_gen_variability = results["resource_profiles"]

        timeseries_df, timepoints_df = ts_tp_pg_kmeans(
            representative_point["slot"],
            weights,
            settings["time_domain_days_per_period"],
            planning_periods,
            planning_period_start_yrs,
        )
        hydro_timepoints_df = hydro_timepoints_pg_kmeans(timepoints_df)
        hydro_timeseries_table = hydro_timeseries_pg_kmeans(
            existing_gen,
            all_gen_variability.loc[
                :, existing_gen.loc[existing_gen["HYDRO"] == 1, "Resource"]
            ],
            hydro_timepoints_df,
        )

        vcf = variable_cf_pg_kmeans(all_gen_variability, timepoints_df)

        loads = load_pg_kmeans(load_curves, timepoints_df)
    else:
        timeseries_df, timepoints_df, timestamp_interval = timeseries(
            load_curves,
            case_years,
            max_weight=20.2778,
            avg_weight=283.8889,
            ts_duration_of_tp=4,
            ts_num_tps=6,
            settings=settings,
        )

        # create lists and dictionary for later use
        timepoints_timestamp = timepoints_df["timestamp"].to_list()  # timestamp list
        timepoints_tp_id = timepoints_df["timepoint_id"].to_list()  # timepoint_id list
        timepoints_dict = dict(
            zip(timepoints_timestamp, timepoints_tp_id)
        )  # {timestamp: timepoint_id}
        hydro_timepoints_df = hydro_timepoints_table(timepoints_df)
        hydro_timepoints_df

        hydro_timeseries_table = hydro_timeseries(
            existing_gen, hydro_variability_new, planning_periods
        )

        loads, loads_with_year_hour = loads_table(
            load_curves, timepoints_timestamp, timepoints_dict, planning_periods
        )
        # for fuel_cost and regional_fuel_market issue
        dummy_df = pd.DataFrame({"TIMEPOINT": timepoints_tp_id})
        dummy_df.insert(0, "LOAD_ZONE", "loadzone")
        dummy_df.insert(2, "zone_demand_mw", 0)
        loads = loads.append(dummy_df)

        year_hour = loads_with_year_hour["year_hour"].to_list()

        vcf = variable_capacity_factors_table(
            all_gen_variability, year_hour, timepoints_dict, all_gen, case_years
        )

    graph_timestamp_map = graph_timestamp_map_table(timeseries_df, timestamp_interval)
    graph_timestamp_map
    timeseries_df.to_csv(out_folder / "timeseries.csv", index=False)
    timepoints_df.to_csv(out_folder / "timepoints.csv", index=False)
    hydro_timepoints_df.to_csv(out_folder / "hydro_timepoints.csv", index=False)
    graph_timestamp_map.to_csv(out_folder / "graph_timestamp_map.csv", index=False)

    balancing_tables(settings, pudl_engine, all_gen_units, out_folder)
    hydro_timeseries_table.to_csv(out_folder / "hydro_timeseries.csv", index=False)

    loads.to_csv(out_folder / "loads.csv", index=False)
    vcf.to_csv(out_folder / "variable_capacity_factors.csv", index=False)
    ###

    gen_buildpre.to_csv(out_folder / "gen_build_predetermined.csv", index=False)
    gen_build_costs.to_csv(out_folder / "gen_build_costs.csv", index=False)


### edit by RR


def other_tables(atb_data_year, out_folder):

    # Based on REAM
    carbon_policies_data = {
        "period": [2020, 2030, 2040, 2050],
        "carbon_cap_tco2_per_yr": [222591761.6, 149423302.5, 76328672.3, 0],
        "carbon_cap_tco2_per_yr_CA": [57699000, 36292500, 11400000, 0],
        "carbon_cost_dollar_per_tco2": [".", ".", ".", "."],
    }
    carbon_policies_table = pd.DataFrame(carbon_policies_data)
    carbon_policies_table

    # interest and discount based on REAM
    financials_data = {
        "base_financial_year": atb_data_year,
        "interest_rate": 0.05,
        "discount_rate": 0.05,
    }
    financials_table = pd.DataFrame(financials_data, index=[0])
    financials_table

    # based on REAM
    periods_data = {
        "INVESTMENT_PERIOD": [2020, 2030, 2040, 2050],
        "period_start": [2016, 2026, 2036, 2046],
        "period_end": [2025, 2035, 2045, 2055],
    }
    periods_table = pd.DataFrame(periods_data)
    periods_table

    carbon_policies_table.to_csv(out_folder / "carbon_policies.csv", index=False)
    financials_table.to_csv(out_folder / "financials.csv", index=False)
    periods_table.to_csv(out_folder / "periods.csv", index=False)


from powergenome.generators import load_ipm_shapefile
from powergenome.GenX import (
    network_line_loss,
    network_max_reinforcement,
    network_reinforcement_cost,
    add_cap_res_network,
)
from powergenome.transmission import (
    agg_transmission_constraints,
    transmission_line_distance,
)
from powergenome.util import init_pudl_connection, load_settings, check_settings
from statistics import mean


def transmission_tables(settings, out_folder, pg_engine):

    """
    pulling in information from PowerGenome transmission notebook
    Schivley Greg, PowerGenome, (2022), GitHub repository,
        https://github.com/PowerGenome/PowerGenome/blob/master/notebooks/Transmission.ipynb
    """
    IPM_regions = settings["model_regions"]

    transmission = agg_transmission_constraints(pg_engine=pg_engine, settings=settings)
    model_regions_gdf = load_ipm_shapefile(settings)

    transmission_line_distance(
        trans_constraints_df=transmission,
        ipm_shapefile=model_regions_gdf,
        settings=settings,
    )

    line_loss = network_line_loss(transmission=transmission, settings=settings)
    reinforcement_cost = network_reinforcement_cost(
        transmission=transmission, settings=settings
    )
    max_reinforcement = network_max_reinforcement(
        transmission=transmission, settings=settings
    )
    transmission = agg_transmission_constraints(pg_engine=pg_engine, settings=settings)
    add_cap = add_cap_res_network(transmission, settings)

    ## transmission lines
    # pulled from SWITCH load_zones file
    # need zone_dbid information to populate transmission_line column
    def load_zones_table(IPM_regions, zone_ccs_distance_km):
        load_zones = pd.DataFrame(
            columns=["LOAD_ZONE", "zone_ccs_distance_km", "zone_dbid"]
        )
        load_zones["LOAD_ZONE"] = IPM_regions
        load_zones["zone_ccs_distance_km"] = 0  # set to default 0
        load_zones["zone_dbid"] = range(1, len(IPM_regions) + 1)
        return load_zones

    IPM_regions = settings.get("model_regions")
    load_zones = load_zones_table(IPM_regions, zone_ccs_distance_km=0)
    zone_dict = dict(
        zip(load_zones["LOAD_ZONE"].to_list(), load_zones["zone_dbid"].to_list())
    )

    tx_capex_mw_mile_dict = settings.get("transmission_investment_cost")["tx"][
        "capex_mw_mile"
    ]

    def region_avg(tx_capex_mw_mile_dict, region1, region2):
        r1_value = tx_capex_mw_mile_dict[region1]
        r2_value = tx_capex_mw_mile_dict[region2]
        r_avg = mean([r1_value, r2_value])
        return r_avg

    def create_transm_line_col(lz1, lz2, zone_dict):
        t_line = zone_dict[lz1] + "-" + zone_dict[lz2]
        return t_line

    transmission_lines = transmission_lines_table(
        line_loss, add_cap, tx_capex_mw_mile_dict, zone_dict, settings
    )
    transmission_lines

    trans_capital_cost_per_mw_km = (
        min(
            settings.get("transmission_investment_cost")["tx"]["capex_mw_mile"].values()
        )
        * 1.60934
    )
    trans_params_table = pd.DataFrame(
        {
            "trans_capital_cost_per_mw_km": trans_capital_cost_per_mw_km,
            "trans_lifetime_yrs": 20,
            "trans_fixed_om_fraction": 0.03,
        },
        index=[0],
    )
    trans_params_table

    transmission_lines.to_csv(out_folder / "transmission_lines.csv", index=False)
    trans_params_table.to_csv(out_folder / "trans_params.csv", index=False)


import ast
import itertools
from statistics import mode


def balancing_tables(settings, pudl_engine, all_gen, out_folder):

    IPM_regions = settings.get("model_regions")
    bal_areas, zone_bal_areas = balancing_areas(
        pudl_engine,
        IPM_regions,
        all_gen,
        quickstart_res_load_frac=0.03,
        quickstart_res_wind_frac=0.05,
        quickstart_res_solar_frac=0.05,
        spinning_res_load_frac=".",
        spinning_res_wind_frac=".",
        spinning_res_solar_frac=".",
    )

    bal_areas

    # adding in the dummy loadzone for the fuel_cost / regional_fuel_market issue
    zone_bal_areas.loc[len(zone_bal_areas.index)] = ["loadzone", "BANC"]
    zone_bal_areas

    bal_areas.to_csv(out_folder / "balancing_areas.csv", index=False)
    zone_bal_areas.to_csv(out_folder / "zone_balancing_areas.csv", index=False)


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

    # load hydro_variability_new, and need to add varibality for region 'MIS_D_MS'
    # by copying values from ' MIS_AR'
    hydro_variability_new = pd.read_csv(input_folder / settings["hydro_variability_fn"])
    hydro_variability_new["MIS_D_MS"] = hydro_variability_new["MIS_AR"]

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
        gen_prebuild_newbuild_info_files(
            gc,
            pudl_engine,
            settings_list,
            case_years,
            case_folder,
            pg_engine,
            hydro_variability_new,
        )
        fuel_files(
            fuel_prices=gc.fuel_prices,
            planning_years=case_years,
            regions=settings["model_regions"],
            fuel_region_map=settings["aeo_fuel_region_map"],
            fuel_emission_factors=settings["fuel_emission_factors"],
            out_folder=case_folder,
        )
        other_tables(atb_data_year=settings["atb_data_year"], out_folder=case_folder)

        transmission_tables(settings, case_folder, pg_engine)


if __name__ == "__main__":
    typer.run(main)
