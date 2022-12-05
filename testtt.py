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

settings_path = cwd / "settings_TD_east.yml"
settings = load_settings(settings_path)

pudl_engine, pudl_out, pg_engine = init_pudl_connection(
    freq="AS",
    start_year=min(settings.get("data_years")),
    end_year=max(settings.get("data_years")),
)
check_settings(settings, pg_engine)
input_folder = cwd / settings["input_folder"]

settings["input_folder"] = input_folder
scenario_definitions = pd.read_csv(input_folder / settings["scenario_definitions_fn"])
scenario_settings = build_scenario_settings(settings, scenario_definitions)

gc = GeneratorClusters(pudl_engine, pudl_out, pg_engine, scenario_settings[2020]["p1"])


all_gen = gc.create_all_generators()
all_gen["Resource"] = all_gen["Resource"].str.rstrip("_")
all_gen["technology"] = all_gen["technology"].str.rstrip("_")
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
pudl_gen_entity = pudl_gen_entity[["plant_id_eia", "generator_id", "operating_date"]]

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
# pg_build = plant_pudl_id(pg_build)
all_gen = plant_pudl_id(all_gen)
######################################################################################
######################################################################################

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
        "Landbased Wind Turbine": False,
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
        "Landbased Wind Turbine": False,
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
        "Landbased Wind Turbine": "Wind",  ## add by RR because run into an erro of KeyError: 'LandbasedWind_Class1_Moderate_'
        "LandbasedWind_Class1_Moderate": "Wind",  ## add by RR because run into an erro of KeyError: 'LandbasedWind_Class1_Moderate_'
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
        "Landbased Wind Turbine": 0.05,
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
        "Landbased Wind Turbine": 0.01,
        "Small Hydroelectric": 0.05,
        "NaturalGas_CCCCSAvgCF_Conservative": 0.6,
        "NaturalGas_CCAvgCF_Moderate": 0.6,
        "NaturalGas_CTAvgCF_Moderate": 0.6,
        "Battery_*_Moderate": 0.01,
        "NaturalGas_CCS100_Moderate": 0.6,
        "heat_load_shifting": False,
    }

# newbuild options

list_decade = [2020, 2030, 2040, 2050]
newgens = pd.DataFrame()
for y in list_decade:
    print(y)
    gc = GeneratorClusters(pudl_engine, pudl_out, pg_engine, scenario_settings[y]["p1"])
    new_gen_decade = gc.create_new_generators()
    new_gen_decade["build_year"] = y
    newgens = newgens.append(new_gen_decade)


def create_new_generators(gc):
    gc.offshore_spur_costs = fetch_atb_offshore_spur_costs(gc.pg_engine, gc.settings)
    gc.offshore_spur_costs.columns
    gc.offshore_spur_costs["technology"].unique()
    gc.atb_costs = fetch_atb_costs(gc.pg_engine, gc.settings, gc.offshore_spur_costs)
    gc.atb_costs["technology"].unique()

    gc.new_generators = atb_new_generators(
        gc.atb_costs, gc.atb_hr, gc.settings, gc.cluster_builder
    )

    logger.info("Creating new resources for each region.")
    new_gen_types = settings["atb_new_gen"]
    model_year = settings["model_year"]
    try:
        first_planning_year = settings["model_first_planning_year"]
        model_year_range = range(first_planning_year, model_year + 1)
    except KeyError:
        model_year_range = list(range(model_year + 1))

    regions = settings["model_regions"]
    atb_costs = gc.atb_costs
    atb_hr = gc.atb_hr
    atb_costs_hr = atb_costs.merge(
        atb_hr, on=["technology", "tech_detail", "cost_case", "basis_year"], how="left"
    )
    if new_gen_types:
        new_gen_df = pd.concat(
            [
                single_generator_row(atb_costs_hr, new_gen, model_year_range)
                for new_gen in new_gen_types
            ],
            ignore_index=True,
        )
    else:
        new_gen_df = pd.DataFrame(
            columns=["region", "technology", "tech_detail", "cost_case"]
        )

    if settings.get("additional_technologies_fn"):
        if isinstance(settings.get("additional_new_gen"), list):
            # user_costs, user_hr = load_user_defined_techs(settings)
            user_tech = load_user_defined_techs(settings)
            # new_gen_df = pd.concat([new_gen_df, user_costs], ignore_index=True, sort=False)
            new_gen_df = pd.concat(
                [new_gen_df, user_tech], ignore_index=True, sort=False
            )
        # atb_hr = pd.concat([atb_hr, user_hr], ignore_index=True, sort=False)
        else:
            logger.warning(
                "A filename for additional technologies was included but no technologies"
                " were specified in the settings file."
            )

    if settings.get("modified_atb_new_gen"):
        modified_gens = add_modified_atb_generators(
            settings, atb_costs_hr, model_year_range
        )
        new_gen_df = pd.concat(
            [new_gen_df, modified_gens], ignore_index=True, sort=False
        )

    new_gen_df = new_gen_df.rename(
        columns={
            "heat_rate": "Heat_Rate_MMBTU_per_MWh",
            "fixed_o_m_mw": "Fixed_OM_Cost_per_MWyr",
            "fixed_o_m_mwh": "Fixed_OM_Cost_per_MWhyr",
            "variable_o_m_mwh": "Var_OM_Cost_per_MWh",
        }
    )

    for tech, _tech_modifiers in (settings.get("atb_modifiers") or {}).items():
        tech_modifiers = copy.deepcopy(_tech_modifiers)
        assert isinstance(tech_modifiers, dict), (
            "The settings parameter 'atb_modifiers' must be a nested list.\n"
            "Each top-level key is a short name of the technology, with a nested"
            " dictionary of items below it."
        )
        assert (
            "technology" in tech_modifiers
        ), "Each nested dictionary in atb_modifiers must have a 'technology' key."
        assert (
            "tech_detail" in tech_modifiers
        ), "Each nested dictionary in atb_modifiers must have a 'tech_detail' key."

        technology = tech_modifiers.pop("technology")
        tech_detail = tech_modifiers.pop("tech_detail")

        allowed_operators = ["add", "mul", "truediv", "sub"]

        for key, op_list in tech_modifiers.items():
            if isinstance(op_list, float) | isinstance(op_list, int):
                new_gen_df.loc[
                    (new_gen_df.technology == technology)
                    & (new_gen_df.tech_detail == tech_detail),
                    key,
                ] = op_list
            else:
                assert len(op_list) == 2, (
                    "Two values, an operator and a numeric value, are needed in the parameter\n"
                    f"'{key}' for technology '{tech}' in 'atb_modifiers'."
                )
                op, op_value = op_list

                assert op in allowed_operators, (
                    f"The key {key} for technology {tech} needs a valid operator from the list\n"
                    f"{allowed_operators}\n"
                    "in the format [<operator>, <value>] to modify the properties of an existing generator.\n"
                )

                f = operator.attrgetter(op)
                new_gen_df.loc[
                    (new_gen_df.technology == technology)
                    & (new_gen_df.tech_detail == tech_detail),
                    key,
                ] = f(operator)(
                    new_gen_df.loc[
                        (new_gen_df.technology == technology)
                        & (new_gen_df.tech_detail == tech_detail),
                        key,
                    ],
                    op_value,
                )

    new_gen_df["technology"] = (
        new_gen_df[["technology", "tech_detail", "cost_case"]]
        .astype(str)
        .agg("_".join, axis=1)
    )

    new_gen_df["cap_recovery_years"] = settings["atb_cap_recovery_years"]

    if new_gen_df.empty:
        results = new_gen_df.copy()
    else:
        for tech, years in (settings.get("alt_atb_cap_recovery_years") or {}).items():
            new_gen_df.loc[
                new_gen_df["technology"].str.lower().str.contains(tech.lower()),
                "cap_recovery_years",
            ] = years

        new_gen_df["Inv_Cost_per_MWyr"] = investment_cost_calculator(
            capex=new_gen_df["capex_mw"],
            wacc=new_gen_df["wacc_real"],
            cap_rec_years=new_gen_df["cap_recovery_years"],
        )

        new_gen_df["Inv_Cost_per_MWhyr"] = investment_cost_calculator(
            capex=new_gen_df["capex_mwh"],
            wacc=new_gen_df["wacc_real"],
            cap_rec_years=new_gen_df["cap_recovery_years"],
        )

        # Set no capacity limit on new resources that aren't renewables.
        new_gen_df["Max_Cap_MW"] = -1
        new_gen_df["Max_Cap_MWh"] = -1
        regional_cost_multipliers = pd.read_csv(
            DATA_PATHS["cost_multipliers"]
            / settings.get(
                "cost_multiplier_fn", "AEO_2020_regional_cost_corrections.csv"
            ),
            index_col=0,
        )
        if settings.get("user_regional_cost_multiplier_fn"):
            user_cost_multipliers = pd.read_csv(
                Path(settings["input_folder"])
                / settings["user_regional_cost_multiplier_fn"],
                index_col=0,
            )
            regional_cost_multipliers = pd.concat(
                [regional_cost_multipliers, user_cost_multipliers], axis=1
            )
        rev_mult_region_map = reverse_dict_of_lists(
            settings["cost_multiplier_region_map"]
        )
        rev_mult_tech_map = reverse_dict_of_lists(
            settings["cost_multiplier_technology_map"]
        )

        df_list = []
        for region in regions:
            _df = new_gen_df.copy()
            _df["region"] = region
            _df = regional_capex_multiplier(
                _df,
                region,
                rev_mult_region_map,
                rev_mult_tech_map,
                regional_cost_multipliers,
            )
            df_list.append(_df)
            results = pd.concat(df_list, ignore_index=True, sort=False)
            results1 = add_renewables_clusters(
                results, region, settings, cluster_builder=None
            )
            _df = add_renewables_clusters(_df, region, settings, cluster_builder=None)

            if region in (settings.get("new_gen_not_available") or {}):
                techs = settings["new_gen_not_available"][region]
                for tech in techs:
                    _df = _df.loc[~_df["technology"].str.contains(tech), :]

            df_list.append(_df)

        results = pd.concat(df_list, ignore_index=True, sort=False)

        int_cols = [
            "Fixed_OM_Cost_per_MWyr",
            "Fixed_OM_Cost_per_MWhyr",
            "Inv_Cost_per_MWyr",
            "Inv_Cost_per_MWhyr",
            "cluster",
        ]
        int_cols = [c for c in int_cols if c in results.columns]
        results = results.fillna(0)
        results[int_cols] = results[int_cols].astype(int)
        results["Var_OM_Cost_per_MWh"] = results["Var_OM_Cost_per_MWh"].astype(float)

    if not gc.new_generators.empty:
        gc.new_generators = (
            gc.new_generators.pipe(startup_fuel, gc.settings)
            .pipe(add_fuel_labels, gc.fuel_prices, gc.settings)
            .pipe(startup_nonfuel_costs, gc.settings)
        )

        if gc.sort_gens:
            logger.info("Sorting new resources alphabetically.")
            gc.new_generators = gc.new_generators.sort_values(["region", "technology"])

        if gc.settings.get("capacity_limit_spur_fn"):
            gc.new_generators = gc.new_generators.pipe(
                add_resource_max_cap_spur, gc.settings
            )
        else:
            logger.warning("No settings parameter for max capacity/spur file")
        gc.new_generators = gc.new_generators.pipe(
            calculate_transmission_inv_cost, gc.settings, gc.offshore_spur_costs
        ).pipe(add_transmission_inv_cost, gc.settings)

    if gc.settings.get("demand_response_fn") or gc.settings.get(
        "electrification_stock_fn"
    ):
        dr_rows = gc.create_demand_response_gen_rows()
        gc.new_generators = pd.concat(
            [gc.new_generators, dr_rows], sort=False, ignore_index=True
        )
    gc.new_generators = add_genx_model_tags(gc.new_generators, gc.settings)
    if "cluster" not in gc.new_generators.columns:
        gc.new_generators["cluster"] = 1
    gc.new_generators["Resource"] = (
        gc.new_generators["region"]
        + "_"
        + snake_case_col(gc.new_generators["technology"])
        + "_"
        + gc.new_generators["cluster"].astype(str)
    )

    return gc.new_generators


complete_gens = pd.concat([existing_gen, newgens]).drop_duplicates(subset=["Resource"])
complete_gens = add_misc_gen_values(complete_gens, gc.settings)
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

complete_gens["gen_energy_source"] = complete_gens["technology"].apply(
    lambda x: energy_tech[x]
)

##################################################################################
##################################################################################

load_curves = make_final_load_curves(pg_engine, scenario_settings[2020]["p1"])
timeseries_df = timeseries(
    load_curves,
    max_weight=20.2778,
    avg_weight=283.8889,
    ts_duration_of_tp=4,
    ts_num_tps=6,
)
timeseries_dates = timeseries_df["timeseries"].to_list()
timestamp_interval = [
    "00",
    "04",
    "08",
    "12",
    "16",
    "20",
]  # should align with ts_duration_of_tp and ts_num_tps
timepoints_df = timepoints_table(timeseries_dates, timestamp_interval)
# create lists and dictionary for later use
timepoints_timestamp = timepoints_df["timestamp"].to_list()  # timestamp list
timepoints_tp_id = timepoints_df["timepoint_id"].to_list()  # timepoint_id list
timepoints_dict = dict(
    zip(timepoints_timestamp, timepoints_tp_id)
)  # {timestamp: timepoint_id}

period_list = ["2020", "2030", "2040", "2050"]
loads, loads_with_year_hour = loads_table(
    load_curves, timepoints_timestamp, timepoints_dict, period_list
)
# for fuel_cost and regional_fuel_market issue
dummy_df = pd.DataFrame({"TIMEPOINT": timepoints_tp_id})
dummy_df.insert(0, "LOAD_ZONE", "loadzone")
dummy_df.insert(2, "zone_demand_mw", 0)
loads = loads.append(dummy_df)

year_hour = loads_with_year_hour["year_hour"].to_list()
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


# def Filter(list1, list2):
#     return [n for n in list1 if any(m in n for m in list2)]


# wind_solar = set(Filter(technology, ["Wind", "Solar"]))
# all_gen.loc[all_gen["technology"].isin(wind_solar), "gen_is_variable"] = True
all_gen["temp_id"] = all_gen.index
all_gen.loc[all_gen["technology"].str.contains("Wind"), "gen_is_variable"] = True
all_gen.loc[all_gen["technology"].str.contains("Solar"), "gen_is_variable"] = True
all_gen.loc[all_gen["technology"].str.contains("PV"), "gen_is_variable"] = True

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
reg_res_cl = all_gen["temp_id"].to_list()
all(isinstance(n, float) for n in reg_res_cl)

import math

reg_res_cl_copy = [str(i) for i in reg_res_cl]
all(isinstance(n, str) for n in reg_res_cl_copy)

# reg_res_cl = [i[0:-2] for i in reg_res_cl_copy]

# var_cap_fac["GENERATION_PROJECT"] = var_cap_fac["GENERATION_PROJECT"] + 1

var_cap_fac["GENERATION_PROJECT"] = [int(i) for i in var_cap_fac["GENERATION_PROJECT"]]

all(isinstance(n, str) for n in var_cap_fac["GENERATION_PROJECT"])

var_cap_fac = var_cap_fac[var_cap_fac["GENERATION_PROJECT"].isin(reg_res_cl_copy)]

# var_cap_fac["GENERATION_PROJECT"] = var_cap_fac["GENERATION_PROJECT"].apply(
#     lambda x: all_gen_convert[x]
# )
# filter to final columns
var_cap_fac = var_cap_fac[
    ["GENERATION_PROJECT", "timepoint", "gen_max_capacity_factor"]
]
var_cap_fac["GENERATION_PROJECT"] = var_cap_fac["GENERATION_PROJECT"] + 1

vcf = variable_capacity_factors_table(
    all_gen_variability, year_hour, timepoints_dict, all_gen
)
