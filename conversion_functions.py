"""
Functions to convert data from PowerGenome for use with Switch
"""

from statistics import mean, mode

import numpy as np
import pandas as pd
import math


def switch_fuel_cost_table(
    aeo_fuel_region_map, fuel_prices, IPM_regions, scenario, year_list
):
    """
    Create the fuel_cost input file based on REAM Scenario 178.
    Inputs:
        * aeo_fuel_region_map: has aeo_fuel_regions and the ipm regions within each aeo_fuel_region
        * fuel_prices: output from PowerGenome gc.fuel_prices
        * IPM_regions: from settings('model_regions')
        * scenario: filtering the fuel_prices table. Suggest using 'reference' for now.
        * year_list: the periods - 2020, 2030, 2040, 2050.  To filter the fuel_prices year column
    Output:
        the fuel_cost_table
            * load_zone: IPM region
            * fuel: based on PowerGenome fuel_prices table
            * period: based on year_list
            * fuel_cost: based on fuel_prices.price
    """

    ref_df = fuel_prices.copy()
    ref_df = ref_df.loc[
        fuel_prices["scenario"] == scenario
    ]  # use reference scenario for now
    ref_df = ref_df[ref_df["year"].isin(year_list)]
    ref_df = ref_df.drop(["full_fuel_name", "scenario"], axis=1)

    # loop through aeo_fuel_regions.
    # for each of the ipm regions in the aeo_fuel, duplicate the fuel_prices table while adding ipm column
    fuel_cost = pd.DataFrame(columns=["year", "price", "fuel", "region", "load_zone"])
    data = list()
    for region in aeo_fuel_region_map.keys():
        df = ref_df.copy()
        df = df[df["region"] == region]
        for ipm in aeo_fuel_region_map[region]:
            ipm_region = ipm
            df["load_zone"] = ipm_region
            fuel_cost = fuel_cost.append(df)
    #     fuel_cost = fuel_cost.append(data)
    fuel_cost.rename(columns={"year": "period", "price": "fuel_cost"}, inplace=True)
    fuel_cost = fuel_cost[["load_zone", "fuel", "period", "fuel_cost"]]
    fuel_cost["period"] = fuel_cost["period"].astype(int)
    fuel_cost = fuel_cost[fuel_cost["load_zone"].isin(IPM_regions)]
    fuel_cost["fuel"] = fuel_cost[
        "fuel"
    ].str.capitalize()  # align with energy_source in gen_pro_info? switch error.
    return fuel_cost


def switch_fuels(fuel_prices, REAM_co2_intensity):
    """
    Create fuels table using fuel_prices (from gc.fuel_prices) and basing other columns on REAM scenario 178
    Output columns
        * fuel: based on the fuels contained in the PowerGenome fuel_prices table
        * co2_intensity: based on REAM scenario 178
        * upstream_co2_intensity: based on REAM scenario 178
    """
    fuels = pd.DataFrame(fuel_prices["fuel"].unique(), columns=["fuel"])
    fuels["co2_intensity"] = fuels["fuel"].apply(lambda x: REAM_co2_intensity[x])
    fuels["upstream_co2_intensity"] = 0  # based on REAM scenario 178
    # switch error - capitalize to align with gen pro info energy_source?
    fuels["fuel"] = fuels["fuel"].str.capitalize()
    return fuels


def create_dict_plantgen(df, column):
    """
    Create dictionary from two columns, removing na's beforehand
    {plant_gen_id: year}
    """
    df = df[df[column].notna()]
    ids = df["plant_gen_id"].to_list()
    dates = df[column].to_list()
    dictionary = dict(zip(ids, dates))
    return dictionary


def create_dict_plantpudl(df: pd.DataFrame, column: str):
    """
    Create dictionary from two columns, removing na's beforehand
    {plant_pudl_id: year}
    """
    df = df.dropna(subset=["build_final"])
    ids = df["plant_pudl_id"].to_list()
    dates = df[column].to_list()
    dictionary = dict(zip(ids, dates))
    return dictionary


def plant_dict(plantideia, dictionary):
    """
    Take key from pandas column, return value from dictionary. Passing if not in dictionary.
    """
    if plantideia in dictionary:
        return dictionary[plantideia]
    else:
        pass


def plant_gen_id(df):
    """
    Create unique id for generator by combining plant_id_eia and generator_id
    """
    plant_id_eia = df["plant_id_eia"]
    df["plant_gen_id"] = plant_id_eia.astype(str) + "_" + df["generator_id"].astype(str)
    return df


def plant_pudl_id(df):
    """
    Create unique id for generator by combining plant_id_eia and unit_pudl_id
    """
    has_plant_id = df.loc[df["plant_id_eia"].notna(), :]
    no_plant_id = df.loc[df["plant_id_eia"].isna(), :]
    plant_id_eia = has_plant_id["plant_id_eia"]
    unit_id_pg = has_plant_id["unit_id_pg"].astype(str)
    has_plant_id.loc[~unit_id_pg.str.contains("_"), "plant_pudl_id"] = (
        plant_id_eia.astype(str) + "_" + unit_id_pg
    )
    has_plant_id.loc[
        has_plant_id["plant_pudl_id"].isna(), "plant_pudl_id"
    ] = has_plant_id.loc[has_plant_id["plant_pudl_id"].isna(), "unit_id_pg"]

    return pd.concat([has_plant_id, no_plant_id], ignore_index=True)


def gen_build_predetermined(
    all_gen: pd.DataFrame,
    pudl_gen: pd.DataFrame,
    pudl_gen_entity: pd.DataFrame,
    pg_build: pd.DataFrame,
    manual_build_yr: dict,
    eia_Gen: pd.DataFrame,
    eia_Gen_prop: pd.DataFrame,
    plant_gen_manual: dict,
    plant_gen_manual_proposed: dict,
    plant_gen_manual_retired: dict,
    retirement_ages: dict,
    capacity_col: str,
):
    """
    Create the gen_build_predetermined table
    Inputs
        1) all_gen: from PowerGenome gc.create_all_generators()
        2) pudl_gen: from PUDL generators_eia860
            - retirement_date
            - planned_retirement)date
            - current_planned_operating_date
        3) pudl_gen_entity: from PUDL generators_entity_eia
            - operating_date
        4) pg_build: from PowerGenome gc.units_model
            - planned_retirement_date
            - operating_date
            - operating_year
            - retirement_year
        5) manual_build_yr: dictionary of build years that were found manually (outside of PUDL and PG)
        6) eia_Gen: eia operable plants
        7) eia_Gen_prop: eia proposed plants
        8) plant_gen_manual, plant_gen_manual_proposed, plant_gen_manual_retired: manually found build_years
        9) retirement_ages: how many years until plant retires
    Output columns
        * GENERATION_PROJECT: index from all_gen
        * build_year: using pudl_gen, pudl_gen_entity, eia excel file, and pg_build to get years
        * gen_predetermined_cap: based on Existing_Cap_MW from all_gen
        * gen_predetermined_storage_energy_mwh: based on Existing_Cap_MWh from all_gen
    Outputs
        gen_buildpre: is the 'offical' table
        gen_build_with_id: is gen_buildpre before 2020 was taken out and with plant_id in it

    """

    """
    Use dictionaries to get the build year from the various sources of information
    """

    # create dictionaries {plant_gen_id: date} from pudl_gen
    plant_op_date_dict = create_dict_plantgen(
        pudl_gen, "current_planned_operating_date"
    )
    plant_plan_ret_date_dict = create_dict_plantgen(pudl_gen, "planned_retirement_date")
    plant_ret_date_dict = create_dict_plantgen(pudl_gen, "retirement_date")

    # create dictionaries {plant_gen_id: date} from pudl_gen_entity
    entity_op_date_dict = create_dict_plantgen(pudl_gen_entity, "operating_date")

    # create dictionaries {plant_gen_id: date} from pg_build
    PG_pl_retire_date_dict = create_dict_plantgen(pg_build, "planned_retirement_date")
    PG_retire_yr_dict = create_dict_plantgen(pg_build, "retirement_year")
    PG_op_date_dict = create_dict_plantgen(pg_build, "operating_date")
    PG_op_yr_dict = create_dict_plantgen(pg_build, "operating_year")

    #  create dictionaries {plant_gen_id: date} from eia excel file
    eia_Gen_dict = create_dict_plantgen(eia_Gen, "operating_year")
    eia_Gen_prop_dict = create_dict_plantgen(eia_Gen_prop, "planned_operating_year")

    """
    Bring in dates based on dictionaries and the plant_gen_id column
    """
    # based on pudl_gen
    pg_build["op_date"] = pg_build["plant_gen_id"].apply(
        lambda x: plant_dict(x, plant_op_date_dict)
    )
    pg_build["plan_retire_date"] = pg_build["plant_gen_id"].apply(
        lambda x: plant_dict(x, plant_plan_ret_date_dict)
    )
    pg_build["retirement_date"] = pg_build["plant_gen_id"].apply(
        lambda x: plant_dict(x, plant_ret_date_dict)
    )

    # based on pudl_gen_entity
    pg_build["entity_op_date"] = pg_build["plant_gen_id"].apply(
        lambda x: plant_dict(x, entity_op_date_dict)
    )

    # based on pg_build
    pg_build["PG_pl_retire"] = pg_build["plant_gen_id"].apply(
        lambda x: plant_dict(x, PG_pl_retire_date_dict)
    )
    pg_build["PG_retire_yr"] = pg_build["plant_gen_id"].apply(
        lambda x: plant_dict(x, PG_retire_yr_dict)
    )
    pg_build["PG_op_date"] = pg_build["plant_gen_id"].apply(
        lambda x: plant_dict(x, PG_op_date_dict)
    )
    pg_build["PG_op_yr"] = pg_build["plant_gen_id"].apply(
        lambda x: plant_dict(x, PG_op_yr_dict)
    )

    # based on manual_build
    pg_build["manual_yr"] = pg_build["plant_id_eia"].apply(
        lambda x: plant_dict(x, manual_build_yr)
    )

    # based on eia excel
    pg_build["eia_gen_op_yr"] = pg_build["plant_gen_id"].apply(
        lambda x: plant_dict(x, eia_Gen_dict)
    )
    pg_build["proposed_year"] = pg_build["plant_gen_id"].apply(
        lambda x: plant_dict(x, eia_Gen_prop_dict)
    )

    # based on eia excel manual dictionary
    pg_build["eia_gen_manual_yr"] = pg_build["plant_gen_id"].apply(
        lambda x: plant_dict(x, plant_gen_manual)
    )
    pg_build["proposed_manual_year"] = pg_build["plant_gen_id"].apply(
        lambda x: plant_dict(x, plant_gen_manual_proposed)
    )
    pg_build["eia_gen_retired_yr"] = pg_build["plant_gen_id"].apply(
        lambda x: plant_dict(x, plant_gen_manual_retired)
    )

    """
    Manipulating the build and retirement year data
        - change to year instead of date, 
        - bring all years into one column
        - remove nans
    """

    # the columns that have the dates as datetime
    columns = [
        "operating_date",
        "op_date",
        "plan_retire_date",
        "retirement_date",
        "entity_op_date",
        "planned_retirement_date",
        "PG_pl_retire",
        "PG_op_date",
    ]
    # change those columns to just year (instead of longer date)
    for c in columns:
        try:
            pg_build[c] = pd.DatetimeIndex(pg_build[c]).year
        except:
            pass

    op_columns = [
        "operating_date",
        "op_date",
        "entity_op_date",
        "PG_op_date",
        "operating_year",
        "planned_operating_year",
        "manual_yr",
        "PG_op_yr",
        "eia_gen_op_yr",
        "eia_gen_manual_yr",
        "proposed_year",
        "proposed_manual_year",
    ]
    pg_build["build_year"] = pg_build[op_columns].max(axis=1)
    # get all build years into one column (includes manual dates and proposed dates)

    # plant_unit_tech = all_gen.dropna(subset=["plant_pudl_id"])[
    #     ["plant_pudl_id", "technology"]
    # ]
    # plant_unit_tech = plant_unit_tech.drop_duplicates(subset=["plant_pudl_id"])
    # plant_unit_tech = plant_unit_tech.set_index("plant_pudl_id")["technology"]
    # pg_build["technology"] = pg_build["plant_pudl_id"].map(plant_unit_tech)
    pg_build["retirement_age"] = pg_build["technology"].map(retirement_ages)
    # pg_build["retirement_age"] = [
    #     val
    #     for key, val in retirement_ages.items()
    #     if pg_build["technology"].str.contains(key, case=False)
    # ]

    pg_build["calc_retirement_year"] = (
        pg_build["build_year"] + pg_build["retirement_age"]
    )
    if not pg_build.query("retirement_age.isna()").empty:
        missing_techs = pg_build.query("retirement_age.isna()")["technology"].unique()
        print(f"The technologies {missing_techs} do not have retirement ages.")
    ret_columns = [
        "planned_retirement_date",
        "retirement_year",
        "plan_retire_date",
        "retirement_date",
        "PG_pl_retire",
        "PG_retire_yr",
        "eia_gen_retired_yr",
        "calc_retirement_year",
    ]
    pg_build["retirement_year"] = pg_build[ret_columns].min(axis=1)

    """
    Start creating the gen_build_predetermined table
    """
    # base it off of PowerGenome all_gen
    gen_buildpre = all_gen.copy()
    # Use the unique "Resource" column for the generation project ID
    gen_buildpre["GENERATION_PROJECT"] = gen_buildpre["Resource"]
    gen_buildpre = gen_buildpre.loc[
        :,
        [
            "index",
            "GENERATION_PROJECT",
            "plant_id_eia",
            "Existing_Cap_MW",  # it was "Cap_Size",
            "capex_mwh",  # Should it be "capex_mwh" or 'Existing_Cap_MWh'?
            "region",
            "plant_pudl_id",
            "technology",
        ],
    ]

    # this ignores new builds
    new_builds = gen_buildpre[gen_buildpre["Existing_Cap_MW"].isna()]
    gen_buildpre = gen_buildpre[gen_buildpre["Existing_Cap_MW"].notna()]

    # create dictionary to go from pg_build to gen_buildpre (build_year)
    gen_buildpre = pd.merge(
        gen_buildpre,
        pg_build[
            [
                "plant_pudl_id",
                "build_year",
                "retirement_year",
                capacity_col,
                "capacity_mwh",
            ]
        ],
        how="left",
    )
    # pg_build_buildyr = create_dict_plantpudl(pg_build, "build_final")
    # gen_buildpre["build_year"] = gen_buildpre["plant_pudl_id"].apply(
    #     lambda x: plant_dict(x, pg_build_buildyr)
    # )

    # # create dictionary to go from pg_build to gen_buildpre (retirement_year)
    # pg_build_retireyr = create_dict_plantpudl(pg_build, "retire_year_final")
    # gen_buildpre["retirement_year"] = gen_buildpre["plant_pudl_id"].apply(
    #     lambda x: plant_dict(x, pg_build_retireyr)
    # )

    # for plants that still don't have a build year but have a retirement year.
    # Base build year off of retirement year: retirement year - retirement age (based on technology)
    # check to see if it is na or None if you get blank build years
    mask = gen_buildpre["build_year"] == "None"
    nans = gen_buildpre[mask]

    if not nans.empty:
        gen_buildpre.loc[mask, "build_year"] = nans.apply(
            lambda row: float(row.retirement_year) - retirement_ages[row.technology],
            axis=1,
        )

    # don't include new builds in gen_build_predetermined
    #     new_builds['GENERATION_PROJECT'] = range(gen_buildpre.shape[0]+1, gen_buildpre.shape[0]+1+new_builds.shape[0])
    #     new_builds = new_builds[['GENERATION_PROJECT', 'Existing_Cap_MW', 'Existing_Cap_MWh']]
    #     new_builds2020 = new_builds.copy()
    #     new_builds2030 = new_builds.copy()
    #     new_builds2040 = new_builds.copy()
    #     new_builds2050 = new_builds.copy()
    #     new_builds2020['build_year'] = 2020
    #     new_builds2030['build_year'] = 2030
    #     new_builds2040['build_year'] = 2040
    #     new_builds2050['build_year'] = 2050

    # filter to final columns
    # gen_build_with_id is an unmodified version of gen_build_pre (still has 2020 plant years)
    gen_build_with_id = gen_buildpre.copy()
    gen_build_with_id = gen_buildpre[
        [
            "GENERATION_PROJECT",
            "build_year",
            "plant_id_eia",
            "retirement_year",
            "plant_pudl_id",
            "technology",
        ]
    ]  # this table is for comparison/testing only
    gen_buildpre = gen_buildpre[
        ["GENERATION_PROJECT", "build_year", capacity_col, "capacity_mwh"]
    ]

    # don't include new builds
    #     gen_buildpre_combined = pd.concat([gen_buildpre, new_builds2020, new_builds2030, new_builds2040, new_builds2050],
    #                                      ignore_index=True)
    #     gen_buildpre = gen_buildpre.append([new_builds2020, new_builds2030, new_builds2040, new_builds2050],
    #                                        ignore_index=True)

    # TODO: #1 why is "capex_mwh" being renamed to "gen_predetermined_storage_energy_mwh"?
    # TODO: #2 should use Existing_Cap_MW instead of Cap_Size for existing capacity
    gen_buildpre.rename(
        columns={
            capacity_col: "gen_predetermined_cap",
            "capacity_mwh": "gen_predetermined_storage_energy_mwh",
        },
        inplace=True,
    )

    gen_buildpre["build_year"] = gen_buildpre["build_year"].astype("Int64")
    gen_buildpre = gen_buildpre.groupby(
        ["GENERATION_PROJECT", "build_year"],
        as_index=False,
        dropna=False,
        sort=False,
    ).sum()
    # based on REAM
    gen_buildpre["gen_predetermined_storage_energy_mwh"] = gen_buildpre[
        "gen_predetermined_storage_energy_mwh"
    ].fillna(".")
    gen_buildpre["gen_predetermined_storage_energy_mwh"] = gen_buildpre[
        "gen_predetermined_storage_energy_mwh"
    ].replace(0, ".")
    gen_buildpre = gen_buildpre.dropna(subset=["build_year"])

    #     gen_buildpre['GENERATION_PROJECT'] = gen_buildpre['GENERATION_PROJECT'].astype(str)

    # SWITCH doesn't like having build years that are in the period
    gen_buildpre.drop(
        gen_buildpre[gen_buildpre["build_year"] == 2020].index, inplace=True
    )

    return gen_buildpre, gen_build_with_id


def gen_build_costs_table(existing_gen, newgens):
    """
    Create gen_build_costs table based off of REAM Scenarior 178.
    Inputs
        pandas dataframes
            existing_gen - from PowerGenome gc.create_region_technology_clusters()
            new_gen_2020 - created by the gen_build_costs notebook
            new_gen_2030 - created by the gen_build_costs notebook
            new_gen_2040 - created by the gen_build_costs notebook
            new_gen_2050 - created by the gen_build_costs notebook
            all_gen - created by PowerGenome
        build_yr_plantid_dict - maps {generation_project: build_year}

    Output columns
        * GENERATION_PROJECT: based on index
        * build_year: based off of the build years from gen_build_predetermined
        * gen_overnight_cost: is 0 for existing, and uses PG capex_mw values for new generators
        * gen_fixed_om: is 0 for existing, and uses PG Fixed_OM_Cost_per_MWyr *1000 (SWITCH is per KW) for new gen
        * gen_storage_energy_overnight_cost: is 0 for existing and uses PG capex_mwh for new generators
    """

    existing = existing_gen.copy()
    #     existing = existing[['index','plant_id_eia']]
    # existing["GENERATION_PROJECT"] = existing["Resource"]
    # #     existing['GENERATION_PROJECT'] = existing['GENERATION_PROJECT'].astype(str)
    # existing["build_year"] = existing["GENERATION_PROJECT"].apply(
    #     lambda x: build_yr_plantid_dict[x]
    # )
    existing["gen_overnight_cost"] = 0
    existing["gen_fixed_om"] = 0
    existing["gen_storage_energy_overnight_cost"] = 0
    existing = existing[
        [
            "GENERATION_PROJECT",
            "build_year",
            "gen_overnight_cost",
            "gen_fixed_om",
            "gen_storage_energy_overnight_cost",
        ]
    ]

    # df_list = []
    # for year, df in newgens.groupby("build_year"):
    #     # start the new GENERATION_PROJECT ids from the end of existing_gen (should tie out to same as gen_proj_info)
    #     df["GENERATION_PROJECT"] = df["GENERATION_PROJECT"] + f"_{year}"
    #     df_list.append(df)
    # combined_new_gens = pd.concat(df_list)

    # combined_new_gens["gen_fixed_om"] = combined_new_gens[
    #     "Fixed_OM_Cost_per_MWyr"
    # ].apply(lambda x: x * 1000)
    newgens["gen_fixed_om"] = newgens["Fixed_OM_Cost_per_MWyr"]
    newgens.drop("Fixed_OM_Cost_per_MWyr", axis=1, inplace=True)
    newgens.rename(
        columns={
            "capex_mw": "gen_overnight_cost",
            "capex_mwh": "gen_storage_energy_overnight_cost",
        },
        inplace=True,
    )

    newgens = newgens[
        [
            "GENERATION_PROJECT",
            "build_year",
            "gen_overnight_cost",
            "gen_fixed_om",
            "gen_storage_energy_overnight_cost",
        ]
    ]

    gen_build_costs = existing.append(newgens, ignore_index=True)

    gen_build_costs["build_year"] = gen_build_costs["build_year"].astype("Int64")
    gen_build_costs = gen_build_costs.groupby(
        ["GENERATION_PROJECT", "build_year"], as_index=False
    ).mean()
    #     gen_build_costs.drop('index', axis=1, inplace=True)

    # gen_storage_energy_overnight_cost should only be for batteries
    gen_build_costs.loc[
        ~gen_build_costs["GENERATION_PROJECT"].str.contains(
            "batter|storage", case=False
        ),
        "gen_storage_energy_overnight_cost",
    ] = "."

    return gen_build_costs


def generation_projects_info(
    all_gen,
    spur_capex_mw_mile,
    retirement_age,
):
    """
    Create the generation_projects_info table based on REAM scenario 178.
    Inputs:
        * all_gen: from PowerGenome gc.create_all_generators()
        * spur_capex_mw_mile: based on the settings file ('transmission_investment_cost')['spur']['capex_mw_mile']
        * retirement age: pulled from settings
        * cogen_tech, baseload_tech, energy_tech, sched_outage_tech, forced_outage_tech
            - these are user defined dictionaries.  Will map values based on the technology
    Output columns:
        * GENERATION_PROJECT: basing on index
        * gen_tech: based on technology
        * gen_energy_source: based on energy_tech input
        * gen_load_zone: IPM region
        * gen_max_age: based on retirement_age
        * gen_is_variable: only solar and wind are true
        * gen_is_baseload: based on baseload_tech
        * gen_full_load_heat_rate: based on Heat_Rate_MMBTU_per_MWh from all_gen
            - if the energy_source is in the non_fuel_energy_sources, this should be '.'
        * gen_variable_om: based on var_om_cost_per_MWh from all_gen
        * gen_connect_cost_per_mw: based on spur_capex_mw_mile * spur_miles; ## plus substation cost
        * gen_dbid: same as generation_project
        * gen_scheduled_outage_rate: based on sched_outage_tech
        * gen_forced_outage_rate: based on forced_outage_tech
        * gen_capacity_limit_mw: based on Existing_Cap_MW from all_gen; ## should be . to new thermo plants, should be upper limits on new renewables(millions MW total across all).
        * gen_min_build_capacity: based on REAM using 0 for now
        * gen_is_cogen: based on cogen_tech input
        * gen_storage_efficiency: based on REAM scenario 178.  batteries use 0.75
        * gen_store_to_release_ratio: based on REAM scenario 178. batteries use 1
        * gen_can_provide_cap_reserves: based on REAM, all 1s
        * gen_self_discharge_rate, gen_discharge_efficiency, gen_land_use_rate, gen_storage_energy_to_power_ratio:
            blanks based on REAM
    """

    gen_project_info = all_gen.copy().reset_index(drop=True)
    gen_project_info["technology"] = gen_project_info["technology"].str.rstrip("_")
    gen_project_info["GENERATION_PROJECT"] = gen_project_info["Resource"]

    # get columns for GENERATION_PROJECT, gen_tech, gen_load_zone, gen_full_load_heat_rate, gen_variable_om,
    # gen_connect_cost_per_mw and gen_capacity_limit_mw
    gen_project_info = gen_project_info[
        [
            "index",
            "GENERATION_PROJECT",
            "technology",
            "region",
            "Heat_Rate_MMBTU_per_MWh",
            "Var_OM_Cost_per_MWh",
            "spur_miles",
            "Existing_Cap_MW",
            "spur_capex",
            "interconnect_capex_mw",
            "Eff_Up",
            "Eff_Down",
            "VRE",
            "Max_Cap_MW",
            "gen_energy_source",
            "gen_is_cogen",
            "gen_is_baseload",
            "gen_scheduled_outage_rate",
            "gen_forced_outage_rate",
            "gen_type",
        ]
    ]

    gen_project_info["gen_connect_cost_per_mw"] = gen_project_info[
        ["spur_capex", "interconnect_capex_mw"]
    ].sum()

    # create gen_connect_cost_per_mw from spur_miles and spur_capex_mw_mile
    gen_project_info["spur_capex_mw_mi"] = gen_project_info["region"].apply(
        lambda x: spur_capex_mw_mile[x]
    )
    gen_project_info["spur_miles"] = gen_project_info["spur_miles"].fillna(0)
    gen_project_info.loc[
        gen_project_info["gen_connect_cost_per_mw"] == 0, "gen_connect_cost_per_mw"
    ] = (gen_project_info["spur_capex_mw_mi"] * gen_project_info["spur_miles"])
    gen_project_info = gen_project_info.drop(["spur_miles", "spur_capex_mw_mi"], axis=1)

    # Heat_Rate_MMBTU_per_MWh needs to be converted to Btu/kWh for gen_full_load_heat_rate
    # mmbtu * 1000000 = btu and 1 mwh * 1000 = kwh
    # 1000000 / 1000 = * 1000
    gen_project_info["Heat_Rate_MMBTU_per_MWh"] = gen_project_info[
        "Heat_Rate_MMBTU_per_MWh"
    ].apply(lambda x: x * 1000)

    # for gen_is_variable - only solar and wind technologies are true
    technology = all_gen["technology"].to_list()

    def Filter(list1, list2):
        return [n for n in list1 if any(m in n for m in list2)]

    gen_project_info.loc[
        gen_project_info["gen_energy_source"].str.contains("Wind"), "gen_is_variable"
    ] = True
    gen_project_info.loc[
        gen_project_info["gen_energy_source"].str.contains("Solar"), "gen_is_variable"
    ] = True
    # gen_project_info.loc[
    #     gen_project_info["technology"].str.contains("PV"), "gen_is_variable"
    # ] = True

    gen_project_info["gen_is_variable"] = gen_project_info["gen_is_variable"].fillna(
        False
    )

    # gen_storage_efficiency and gen_store_to_release_ratio: battery info based on REAM
    battery = set(Filter(technology, ["Battery", "Batteries", "Storage"]))
    gen_project_info.loc[
        gen_project_info["technology"].isin(battery), "gen_storage_efficiency"
    ] = (gen_project_info[["Eff_Up", "Eff_Down"]].mean(axis=1) ** 2)
    gen_project_info["gen_storage_efficiency"] = gen_project_info[
        "gen_storage_efficiency"
    ].fillna(".")
    gen_project_info.loc[
        gen_project_info["technology"].isin(battery), "gen_store_to_release_ratio"
    ] = 1
    gen_project_info["gen_store_to_release_ratio"] = gen_project_info[
        "gen_store_to_release_ratio"
    ].fillna(".")

    # additional columns based on REAM
    gen_project_info["gen_min_build_capacity"] = 0  # REAM is just 0 or .
    gen_project_info[
        "gen_can_provide_cap_reserves"
    ] = 1  # all ones in scenario 178. either 1 or 0

    # these are blanks in scenario 178
    gen_project_info["gen_self_discharge_rate"] = "."
    gen_project_info["gen_discharge_efficiency"] = "."
    gen_project_info["gen_land_use_rate"] = "."
    gen_project_info["gen_storage_energy_to_power_ratio"] = "."

    # retirement ages based on settings file still need to be updated
    gen_project_info["gen_max_age"] = gen_project_info["technology"].map(retirement_age)

    # Tell user about missing retirement ages
    if not gen_project_info.query("gen_max_age.isna()").empty:
        missing_ret_tech = gen_project_info.query("gen_max_age.isna()")[
            "technology"
        ].unique()
        print(
            f"The technologies {missing_ret_tech} do not have a valid retirement age in "
            "your settings file."
        )

    # GENERATION_PROJECT - the all_gen.index column has NaNs for the new generators.  Use actual index for all_gen
    # gen_project_info["GENERATION_PROJECT"] = gen_project_info.index + 1
    gen_project_info["gen_dbid"] = gen_project_info["GENERATION_PROJECT"]

    # gen_capacity_limit_mw - edit by RR,
    # it was from 'Existing_Cap_MW' only, now takes the max of "Existing_Cap_MW" and "Max_Cap_MW" for new renewables.
    gen_project_info["gen_capacity_limit_mw"] = gen_project_info["Existing_Cap_MW"]
    gen_project_info.loc[
        gen_project_info["VRE"] == 1, "gen_capacity_limit_mw"
    ] = gen_project_info[["Existing_Cap_MW", "Max_Cap_MW"]].max(axis=1)

    # rename columns
    gen_project_info.rename(
        columns={
            "technology": "gen_tech",
            "region": "gen_load_zone",
            "Heat_Rate_MMBTU_per_MWh": "gen_full_load_heat_rate",
            "Var_OM_Cost_per_MWh": "gen_variable_om",
        },
        inplace=True,
    )  #'index':'GENERATION_PROJECT',
    # drop heat_load_shifting (not in SWITCH)
    gen_project_info.drop(
        gen_project_info[gen_project_info["gen_tech"] == "heat_load_shifting"].index,
        inplace=True,
    )

    cols = [
        "GENERATION_PROJECT",
        "gen_tech",
        "gen_energy_source",
        "gen_load_zone",
        "gen_max_age",
        "gen_is_variable",
        "gen_is_baseload",
        "gen_full_load_heat_rate",
        "gen_variable_om",
        "gen_connect_cost_per_mw",
        "gen_dbid",
        "gen_scheduled_outage_rate",
        "gen_forced_outage_rate",
        "gen_capacity_limit_mw",
        "gen_min_build_capacity",
        "gen_is_cogen",
        "gen_storage_efficiency",
        "gen_store_to_release_ratio",
        "gen_can_provide_cap_reserves",
        "gen_self_discharge_rate",
        "gen_discharge_efficiency",
        "gen_land_use_rate",
        "gen_storage_energy_to_power_ratio",
        "gen_type",
    ]  # index

    # remove NaN
    gen_project_info["gen_variable_om"] = gen_project_info["gen_variable_om"].fillna(0)
    # gen_project_info['gen_connect_cost_per_mw'] = gen_project_info['gen_variable_om'].fillna(0)
    # gen_project_info['gen_capacity_limit_mw'] = gen_project_info['gen_variable_om'].fillna('.')

    gen_project_info["gen_connect_cost_per_mw"] = gen_project_info[
        "gen_connect_cost_per_mw"
    ].fillna(0)
    gen_project_info["gen_capacity_limit_mw"] = gen_project_info[
        "gen_capacity_limit_mw"
    ].fillna(".")

    gen_project_info["gen_full_load_heat_rate"] = gen_project_info[
        "gen_full_load_heat_rate"
    ].replace(0, ".")

    gen_project_info = gen_project_info[cols]
    return gen_project_info


hydro_forced_outage_tech = {
    "conventional_hydroelectric": 0.05,
    "hydroelectric_pumped_storage": 0.05,
    "small_hydroelectric": 0.05,
}


def match_hydro_forced_outage_tech(x):
    for key in hydro_forced_outage_tech:
        if key in x:
            return hydro_forced_outage_tech[key]


def fuel_market_tables(fuel_prices, aeo_fuel_region_map, scenario):
    """
    Create regional_fuel_markets and zone_to_regional_fuel_market
    SWITCH does not seem to like this overlapping with fuel_cost. So all of this might be incorrect.
    """

    # create initial regional fuel market.  Format: region - fuel
    reg_fuel_mar_1 = fuel_prices.copy()
    reg_fuel_mar_1 = reg_fuel_mar_1.loc[
        reg_fuel_mar_1["scenario"] == scenario
    ]  # use reference for now
    reg_fuel_mar_1 = reg_fuel_mar_1.drop(
        ["year", "price", "full_fuel_name", "scenario"], axis=1
    )
    reg_fuel_mar_1 = reg_fuel_mar_1.rename(columns={"region": "regional_fuel_market"})
    reg_fuel_mar_1 = reg_fuel_mar_1[["regional_fuel_market", "fuel"]]

    fuel_markets = reg_fuel_mar_1["regional_fuel_market"].unique()

    # from region to fuel
    group = reg_fuel_mar_1.groupby("regional_fuel_market")
    fuel_market_dict = {}
    for region in fuel_markets:
        df = group.get_group(region)
        fuel = df["fuel"].to_list()
        fuel = list(set(fuel))
        fuel_market_dict[region] = fuel

    # create zone_regional_fuel_market
    data = list()
    for region in aeo_fuel_region_map.keys():
        for i in range(len(aeo_fuel_region_map[region])):
            ipm = aeo_fuel_region_map[region][i]
            for fuel in fuel_market_dict[region]:
                data.append([ipm, ipm + "-" + fuel])

    zone_regional_fm = pd.DataFrame(data, columns=["load_zone", "regional_fuel_market"])

    # use that to finish regional_fuel_markets
    regional_fuel_markets = zone_regional_fm.copy()
    regional_fuel_markets["fuel_list"] = regional_fuel_markets[
        "regional_fuel_market"
    ].str.split("-")
    regional_fuel_markets["fuel"] = regional_fuel_markets["fuel_list"].apply(
        lambda x: x[-1]
    )
    regional_fuel_markets = regional_fuel_markets[["regional_fuel_market", "fuel"]]

    return regional_fuel_markets, zone_regional_fm


def timeseries(
    load_curves,
    case_years,
    settings,
):  # 20.2778, 283.8889
    """
    Create the timeseries table based on REAM Scenario 178
    Input:
        1) load_curves: created using PowerGenome make(final_load_curves(pg_engine, scenario_settings[][]))
        2) max_weight: the weight to apply to the days with highest values
        3) avg_weight: the weight to apply to the days with average value
        3) ts_duration_of_tp: how many hours should the timpoint last
        4) ts_num_tps: number of timpoints in the selected day
    Output columns:
        - TIMESERIES: format: yyyy_yyyy-mm-dd
        - ts_period: the period decade
        - ts_duration_of_tp: based on input value
        - ts_num_tps: based on input value. Should multiply to 24 with ts_duration_of_tp
        - ts_scale_to_period: use the max&avg_weights for the average and max days in a month
    """
    if settings.get("sample_dates_fn") and settings.get("input_folder"):
        sample_dates = pd.read_csv(
            settings.get("input_folder") / settings["sample_dates_fn"]
        )
    else:
        sample_year = min(case_years)
        sample_year_start = str(sample_year) + "0101"
        sample_year_end = str(sample_year) + "1231"
        sample_dates = [
            d.strftime("%Y%m%d")
            for d in pd.date_range(sample_year_start, sample_year_end)
        ]

    leap_yr = str(sample_year) + "0229"
    if leap_yr in sample_dates:
        sample_dates.remove(leap_yr)  ### why remove Feb 29th? --RR

    hr_load_sum = pd.DataFrame(load_curves.sum(axis=1), columns=["sum_across_regions"])
    load_hrs = len(load_curves.index)  # number of hours PG outputs data for in a year
    baseyear_hours = len(sample_dates) * 24
    hr_interval = round(load_hrs / baseyear_hours)
    # hr_int_list = list(range(1, int(24 / hr_interval) + 1))
    hr_interval_load_sum = hr_load_sum.groupby(hr_load_sum.index // hr_interval).sum()
    # create initial date list for 2020
    timestamp = list()
    for d in range(len(sample_dates)):
        for i in range(1, 25):
            date_hr = sample_dates[d]
            timestamp.append(date_hr)

    timeseries = [x[:4] + "_" + x[:4] + "-" + x[4:6] + "-" + x[6:8] for x in timestamp]
    ts_period = [x[:4] for x in timestamp]
    timepoint_id = list(range(len(timestamp)))

    column_list = ["timeseries", "ts_period"]
    data = np.array([timeseries, ts_period]).T
    initial_df = pd.DataFrame(
        data, columns=column_list, index=hr_interval_load_sum.index
    )
    initial_df = initial_df.join(hr_interval_load_sum)

    if settings.get("chunk_days"):
        chunk_days = settings.get("chunk_days")
        # split dataframe into chunks of representative_days
        chunk_hr = chunk_days * 24
        n_chunks = len(sample_dates) // chunk_days
        chunk_df = []
        for i in range(n_chunks):
            ck_df = (
                (initial_df.iloc[i * chunk_hr : (i + 1) * chunk_hr, :])
                .groupby("timeseries")
                .sum()
            )
            chunk_df.append(ck_df)
    else:
        month_hrs = [744, 672, 744, 720, 744, 720, 744, 744, 720, 744, 720, 744]
        year_cumul = [
            744,
            1416,
            2160,
            2880,
            3624,
            4344,
            5088,
            5832,
            6552,
            7296,
            8016,
            8760,
        ]  # cumulative hours by month

        # split dataframe into months (grouped by day)
        chunk_df = []
        chunk_df.append(
            (initial_df.iloc[0 : year_cumul[0], :]).groupby("timeseries").sum()
        )
        for i in range(len(year_cumul) - 1):
            M_df = (
                (initial_df.iloc[year_cumul[i] : year_cumul[i + 1], :])
                .groupby("timeseries")
                .sum()
            )
            chunk_df.append(M_df)

    # find mean and max for each month, add date to a dataframe
    timeseries_df = pd.DataFrame(
        columns=["sum_across_regions", "timeseries", "close_to_mean"]
    )
    for df in chunk_df:
        df["timeseries"] = df.index
        mean = df["sum_across_regions"].mean()
        df["close_to_mean"] = abs(df["sum_across_regions"] - mean)
        df_mean = df.loc[df["close_to_mean"] == df["close_to_mean"].min()]
        df_max = df.loc[df["sum_across_regions"] == df["sum_across_regions"].max()]
        timeseries_df = timeseries_df.append(df_max)
        timeseries_df = timeseries_df.append(df_mean)
        timeseries_df["timeseries"] = timeseries_df.index

    # add in other columns
    timeseries_df["ts_period"] = str(sample_year)
    ts_duration_of_tp = settings.get("ts_duration_of_tp")
    ts_num_tps = settings.get("ts_num_tps")
    timeseries_df["ts_duration_of_tp"] = ts_duration_of_tp  # assuming 4 for now
    timeseries_df["ts_num_tps"] = ts_num_tps  # assuming 6 for now
    timeseries_df = timeseries_df.reset_index(drop=True)
    timeseries_df = timeseries_df.drop(["sum_across_regions"], axis=1)

    timeseries_df["ts_scale_to_period"] = None

    planning_years = settings.get("planning_years")
    max_days = settings.get("max_days")
    sample_to_year_ratio = round(8760 / (len(sample_dates) * 24), 3)
    max_weight = planning_years * max_days * sample_to_year_ratio
    avg_weight = planning_years * (chunk_days - max_days) * sample_to_year_ratio

    for i in range(len(timeseries_df)):
        if i % 2 == 0:
            timeseries_df.loc[i, "ts_scale_to_period"] = max_weight
    timeseries_df["ts_scale_to_period"].replace(
        to_replace=[None], value=avg_weight, inplace=True
    )

    # add in addtional years (just replace 2020 with new year)
    addtl_yrs = case_years
    addtl_yrs.remove(sample_year)
    addtl_df = pd.DataFrame(columns=timeseries_df.columns)
    for y in addtl_yrs:
        df = timeseries_df.copy()
        df["ts_period"] = str(y)
        col1 = df["timeseries"].to_list()
        col1 = [str(y) + "_" + str(y) + x[9:] for x in col1]
        df["timeseries"] = col1
        addtl_df = addtl_df.append(df)
    timeseries_df = timeseries_df.append(addtl_df)
    timeseries_df = timeseries_df[
        [
            "timeseries",
            "ts_period",
            "ts_duration_of_tp",
            "ts_num_tps",
            "ts_scale_to_period",
        ]
    ]

    timeseries_dates = timeseries_df["timeseries"].to_list()
    timestamp_interval = list()
    for i in range(ts_num_tps):
        s_interval = ts_duration_of_tp * i
        stamp_interval = str(f"{s_interval:02d}")
        timestamp_interval.append(stamp_interval)

    timepoint_id = list(range(1, len(timeseries_dates) + 1))
    timestamp = [x[:4] + x[10:12] + x[13:] for x in timeseries_dates]

    column_list = ["timepoint_id", "timestamp", "timeseries"]
    timepoints_df = pd.DataFrame(columns=column_list)
    for i in timestamp_interval:
        timestamp_interval = [x + i for x in timestamp]
        df_data = np.array([timepoint_id, timestamp_interval, timeseries_dates]).T
        df = pd.DataFrame(df_data, columns=column_list)
        timepoints_df = timepoints_df.append(df)

    timepoints_df["timepoint_id"] = range(
        1, len(timepoints_df["timeseries"].to_list()) + 1
    )

    return timeseries_df, timepoints_df, timestamp_interval


def hydro_time_tables(existing_gen, hydro_variability, period_list, timepoints_df):
    """
    Create the hydro_timepoints table based on REAM Scenario 178
    Inputs:
        1) timepoints_df: the SWITCH timepoints table
    Output Columns
        * timepoint_id: from the timepoints table
        * tp_to_hts: format: yyyy_M#. Based on the timestamp date from the timepoints table
    """

    hydro_timepoints = timepoints_df
    hydro_timepoints = hydro_timepoints.rename(columns={"timeseries": "tp_to_hts"})

    hydro_list = [
        "Conventional Hydroelectric",
        # "Hydroelectric Pumped Storage",
        "Small Hydroelectric",
    ]

    # filter existing gen to just hydro technologies
    # hydro_df = existing_gen.copy()
    # hydro_df['index'] = hydro_df.index
    # hydro_df = hydro_df[hydro_df['technology'].isin(hydro_list)]
    # hydro_indx = hydro_df['index'].to_list()

    # # get existing variability for the hydro technologies
    # hydro_variability = existing_variability.copy()

    # hydro_variability = hydro_variability.iloc[:, hydro_indx] # hydro hourly for 1 year

    #### edit by RR
    # filter existing gen to just hydro technologies
    hydro_df = existing_gen.copy()
    # hydro_df["index"] = hydro_df.index
    hydro_df = hydro_df[hydro_df["technology"].isin(hydro_list)]
    hydro_indx = hydro_df["Resource"].to_list()
    hydro_region = hydro_df["region"].to_list()

    # get existing variability for the hydro technologies
    # hydro_variability = existing_variability.copy()
    # hydro_variability = hydro_variability_new.copy()
    # slice the hours to 8760
    hydro_variability = hydro_variability.iloc[:8760]
    # hydro_variability = hydro_variability.iloc[:, hydro_indx] # hydro hourly for 1 year
    hydro_variability = hydro_variability.loc[:, hydro_region]
    hydro_variability.columns = hydro_indx
    ####

    # get cap size for each hydro tech
    hydro_Cap_Size = hydro_df["Cap_Size"].to_list()  # cap size for each hydro
    # multiply cap size by hourly
    for i in range(len(hydro_Cap_Size)):
        hydro_variability.iloc[:, i] = hydro_variability.iloc[:, i].apply(
            lambda x: x * hydro_Cap_Size[i]
        )

    hydro_transpose = hydro_variability.transpose()

    month_hrs = [744, 672, 744, 720, 744, 720, 744, 744, 720, 744, 720, 744]
    year_cumul = [
        744,
        1416,
        2160,
        2880,
        3624,
        4344,
        5088,
        5832,
        6552,
        7296,
        8016,
        8760,
    ]  # cumulative hours by month

    # split dataframe into months
    month_df = []
    month_df.append((hydro_transpose.iloc[:, 0 : year_cumul[0]]))
    for i in range(len(year_cumul) - 1):
        M_df = hydro_transpose.iloc[:, year_cumul[i] : year_cumul[i + 1]]
        month_df.append(M_df)

    month_names = [
        "M1",
        "M2",
        "M3",
        "M4",
        "M5",
        "M6",
        "M7",
        "M8",
        "M9",
        "M10",
        "M11",
        "M12",
    ]
    df_list = list()
    for i in range(len(month_df)):
        df = pd.DataFrame(hydro_transpose.index, columns=["hydro_project"])
        df["timeseries"] = month_names[i]
        df["outage_rate"] = list(
            map(match_hydro_forced_outage_tech, hydro_df["Resource"])
        )
        df["hydro_min_flow_mw"] = month_df[i].min(axis=1).to_list()
        df["hydro_avg_flow_mw"] = month_df[i].mean(axis=1).to_list()
        # df["hydro_min_flow_mw_raw"] = month_df[i].min(axis=1).to_list()
        # df["hydro_min_flow_mw"] = df["hydro_min_flow_mw_raw"] * (1 - df["outage_rate"])
        # df["hydro_avg_flow_mw_raw"] = month_df[i].mean(axis=1).to_list()
        # df["hydro_avg_flow_mw"] = df["hydro_avg_flow_mw_raw"] * (1 - df["outage_rate"])
        df_list.append(df)
    hydro_final = pd.concat(df_list, axis=0)
    # # get the index of existing gen for the hydro_project columnn (tie to GENERATION_PROJECTS)
    # hydro_df['region_resource_cluster'] = hydro_df["region"]+ "_"+ hydro_df["Resource"]+ "_"+ hydro_df["cluster"].astype(str)
    # hydro_index_dict = dict(zip(hydro_df['region_resource_cluster'].to_list(), hydro_df['index'].to_list()))
    # hydro_final['hydro_project'] = hydro_final['hydro_project'].apply(lambda x: hydro_index_dict[x])
    # # generation_project starts wtih 1 not 0
    # hydro_final["hydro_project"] = hydro_final["hydro_project"].apply(lambda x: x + 1)
    timeseries_list = list()
    for decade in period_list:
        df2 = hydro_final.copy()
        df2["timeseries"] = decade + "_" + df2["timeseries"]
        timeseries_list.append(df2)

    hydro_timeseries = pd.concat(timeseries_list, axis=0)

    # hydro_final_df = hydro_final_df.drop(
    #     columns=["outage_rate", "hydro_min_flow_mw_raw", "hydro_avg_flow_mw_raw"]
    # )
    return hydro_timepoints, hydro_timeseries


def graph_timestamp_map_table(timeseries_df, timestamp_interval):
    """
    Create the graph_timestamp_map table based on REAM Scenario 178
    Input:
        1) timeseries_df: the SWITCH timeseries table
        2) timestamp_interval:based on ts_duration_of_tp and ts_num_tps from the timeseries table.
                Should be between 0 and 24.
    Output columns:
        * timestamp: dates based on the timeseries table
        * time_row: the period decade year based on the timestamp
        * time_column: format: yyyymmdd. Using 2012 because that is the year data is based on.
    """

    timeseries_df_copy = timeseries_df.copy()
    timeseries_df_copy = timeseries_df_copy[["timeseries", "ts_period"]]
    # reformat timeseries for timestamp
    timeseries_df_copy["timestamp"] = timeseries_df_copy["timeseries"].apply(
        lambda x: x[5:9] + x[10:12] + x[13:]
    )

    # add in intervals to the timestamp
    graph_timeseries_map = pd.DataFrame(
        columns=["timeseries", "ts_period", "timestamp"]
    )
    for x in timestamp_interval:
        df = timeseries_df_copy[["timeseries", "ts_period"]]
        df["timestamp"] = timeseries_df_copy["timestamp"] + x
        graph_timeseries_map = graph_timeseries_map.append(df)

    # using 2012 for financial year
    graph_timeseries_map["time_column"] = graph_timeseries_map["timeseries"].apply(
        lambda x: str(2012) + x[10:12] + x[13:15]
    )

    graph_timeseries_map = graph_timeseries_map.drop(["timeseries"], axis=1)
    graph_timeseries_map = graph_timeseries_map.rename(
        columns={"ts_period": "time_row"}
    )
    graph_timeseries_map = graph_timeseries_map[
        ["timestamp", "time_row", "time_column"]
    ]

    return graph_timeseries_map


def loads_table(load_curves, timepoints_timestamp, timepoints_dict, period_list):
    """
    Inputs:
        load_curves: from powergenome
        timepoints_timestamp: the timestamps in timepoints
        timepoints_dict: to go from timestamp to timepoint
        period_list: the decade list
    Output columns
        * load_zone: the IPM regions
        * timepoint: from timepoints
        * zone_demand_mw: based on load_curves
    Output df
        loads: the 'final' table
        loads_with_hour_year: include hour year so it is easier to do variable_capacity_factors
    """

    loads_initial = pd.DataFrame(columns=["year_hour", "LOAD_ZONE", "zone_demand_mw"])
    hours = load_curves.index.to_list()
    cols = load_curves.columns.to_list()

    # add load zone for each hour of the year, adding in the load_curve values for each hour
    for c in cols:
        df = pd.DataFrame()
        df["year_hour"] = hours
        df["LOAD_ZONE"] = c
        df["zone_demand_mw"] = load_curves[c].to_list()
        loads_initial = loads_initial.append(df)

    # convert timepoints to date of the year
    start = pd.to_datetime("2021-01-01 0:00")  # use 2021 due to 2020 being a leap year
    loads_initial["date"] = loads_initial["year_hour"].apply(
        lambda x: start + pd.to_timedelta(x, unit="H")
    )
    # reformat to timestamp format
    loads_initial["reformat"] = loads_initial["date"].apply(
        lambda x: x.strftime("%Y%m%d%H")
    )
    loads_initial["reformat"] = loads_initial["reformat"].astype(str)
    # create timestamp
    date_list = loads_initial["reformat"].to_list()

    # loop through for each period
    df_list = list()
    for p in period_list:
        df = loads_initial.copy()
        updated_dates = [p + x[4:] for x in date_list]
        df["timestamp"] = updated_dates
        df_list.append(df)
    loads = pd.concat(df_list)

    # filter to correct timestamps for timepoints
    loads = loads[loads["timestamp"].isin(timepoints_timestamp)]
    loads["TIMEPOINT"] = loads["timestamp"].apply(lambda x: timepoints_dict[x])
    loads_with_year_hour = loads[["timestamp", "TIMEPOINT", "year_hour"]]
    loads = loads[["LOAD_ZONE", "TIMEPOINT", "zone_demand_mw"]]

    return loads, loads_with_year_hour


def variable_capacity_factors_table(
    all_gen_variability, year_hour, timepoints_dict, all_gen, case_years
):
    """
    Inputs
        all_gen_variability: from powergenome
        year_hour: the hour of the year that has a timepoint (based on loads)
        timepoints_dict: convert timestamp to timepoint
        all_gen: from powergenome
    Output:
        GENERATION_PROJECT: based on all_gen index
            the plants here should only be the ones with gen_is_variable =True
        timepoint: based on timepoints
        gen_max_capacity_factor: based on all_gen_variability
    """

    v_capacity_factors = all_gen_variability.copy().transpose()
    v_capacity_factors["GENERATION_PROJECT"] = all_gen["Resource"].values
    v_c_f = v_capacity_factors.melt(
        id_vars="GENERATION_PROJECT",
        var_name="year_hour",
        value_name="gen_max_capacity_factor",
    )
    # reduce variability to just the hours of the year that have a timepoint
    v_c_f = v_c_f.loc[v_c_f["year_hour"].isin(year_hour), :]

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
    var_cap_fac = pd.DataFrame()
    for i in case_years:
        updated_dates = [str(i) + x[4:] for x in date_list]
        mod_vcf_copy = mod_vcf.copy()
        mod_vcf_copy["timestamp"] = updated_dates
        mod_vcf_copy["timepoint"] = mod_vcf_copy["timestamp"].apply(
            lambda x: timepoints_dict[x]
        )
        mod_vcf_copy.drop(
            ["year_hour", "date", "reformat", "timestamp"], axis=1, inplace=True
        )
        var_cap_fac = pd.concat([var_cap_fac, mod_vcf_copy], ignore_index=True)

    # only get all_gen plants that are wind or solar
    technology = all_gen["technology"].to_list()

    def Filter(list1, list2):
        return [n for n in list1 if any(m in n for m in list2)]

    # all_gen["temp_id"] = all_gen.index
    all_gen.loc[
        all_gen["gen_energy_source"].str.contains("Wind"), "gen_is_variable"
    ] = True
    all_gen.loc[
        all_gen["gen_energy_source"].str.contains("Solar"), "gen_is_variable"
    ] = True
    # all_gen.loc[all_gen["technology"].str.contains("PV"), "gen_is_variable"] = True
    all_gen = all_gen[all_gen["gen_is_variable"] == True]

    var_cap_fac = var_cap_fac[
        var_cap_fac["GENERATION_PROJECT"].isin(all_gen["Resource"])
    ]

    # filter to final columns
    var_cap_fac = var_cap_fac[
        ["GENERATION_PROJECT", "timepoint", "gen_max_capacity_factor"]
    ]

    return var_cap_fac


def load_zones_table(IPM_regions, zone_ccs_distance_km):
    load_zones = pd.DataFrame(
        columns=["LOAD_ZONE", "zone_ccs_distance_km", "zone_dbid"]
    )
    load_zones["LOAD_ZONE"] = IPM_regions
    load_zones["zone_ccs_distance_km"] = 0  # set to default 0
    load_zones["zone_dbid"] = range(1, len(IPM_regions) + 1)
    return load_zones


def region_avg(tx_capex_mw_mile_dict, region1, region2):
    r1_value = tx_capex_mw_mile_dict[region1]
    r2_value = tx_capex_mw_mile_dict[region2]
    r_avg = mean([r1_value, r2_value])
    return r_avg


def create_transm_line_col(lz1, lz2, zone_dict):
    t_line = zone_dict[lz1] + "-" + zone_dict[lz2]
    return t_line


def transmission_lines_table(
    line_loss, add_cap, tx_capex_mw_mile_dict, zone_dict, settings
):
    """
    Create transmission_lines table based on REAM Scenario 178
    Output Columns:
        TRANSMISSION_LINE: zone_dbid-zone_dbid for trans_lz1 and lz2
        trans_lz1: split PG transmission_path_name
        trans_lz2: split PG transmission_path_name
        trans_length_km: PG distance_mile * need to convert to km (*1.60934)
        trans_efficiency: PG line_loss_percentage (1 - line_loss_percentage)
        existing_trans_cap: PG line_max_cap_flow. Take absolute value and take max of the two values
        trans_dbid: id number
        trans_derating_factor: assuming PG DerateCapRes_1 (0.95)
        trans_terrain_multiplier:
            trans_capital_cost_per_mw_km * trans_terrain_multiplier = the average of the two regions
            ('transmission_investment_cost')['tx']['capex_mw_mile'])
        trans_new_build_allowed: how to determine what is allowed. Assume all 1s to start
    """
    transmission_df = line_loss[
        [
            "Network_Lines",
            "transmission_path_name",
            "distance_mile",
            "Line_Loss_Percentage",
        ]
    ]

    # split to get trans_lz1 and trans_lz2
    split_path_name = transmission_df["transmission_path_name"].str.split(
        "_to_", expand=True
    )
    transmission_df = transmission_df.join(split_path_name)

    # convert miles to km for trans_length_km
    transmission_df["trans_length_km"] = transmission_df["distance_mile"].apply(
        lambda x: x * 1.609
    )

    # for trans_efficiency do 1 - line_loss_percentage
    transmission_df["trans_efficiency"] = transmission_df["Line_Loss_Percentage"].apply(
        lambda x: 1 - x
    )

    transmission_df = transmission_df.join(
        add_cap[["Line_Max_Flow_MW", "Line_Min_Flow_MW", "DerateCapRes_1"]]
    )

    # want the max value so take abosolute of line_min_flow_mw (has negatives) and then take max
    transmission_df["line_min_abs"] = transmission_df["Line_Min_Flow_MW"].abs()
    transmission_df["existing_trans_cap"] = transmission_df[
        ["Line_Max_Flow_MW", "line_min_abs"]
    ].max(axis=1)

    # get rid of columns
    transm_final = transmission_df.drop(
        [
            "transmission_path_name",
            "distance_mile",
            "Line_Loss_Percentage",
            "Line_Max_Flow_MW",
            "Line_Min_Flow_MW",
            "line_min_abs",
        ],
        axis=1,
    )

    transm_final = transm_final.rename(
        columns={
            "Network_Lines": "trans_dbid",
            0: "trans_lz1",
            1: "trans_lz2",
            "DerateCapRes_1": "trans_derating_factor",
        }
    )

    transm_final["tz1_dbid"] = transm_final["trans_lz1"].apply(lambda x: zone_dict[x])
    transm_final["tz2_dbid"] = transm_final["trans_lz2"].apply(lambda x: zone_dict[x])
    transm_final["TRANSMISSION_LINE"] = (
        transm_final["tz1_dbid"].astype(str)
        + "-"
        + transm_final["tz2_dbid"].astype(str)
    )
    # trans_capital_cost_per_mw_km * trans_terrain_multiplier = average of trans_lz1 and trans_lz2
    trans_capital_cost_per_mw_km = (
        min(
            settings.get("transmission_investment_cost")["tx"]["capex_mw_mile"].values()
        )
        * 1.60934
    )
    transm_final["region_avgs"] = transm_final.apply(
        lambda row: region_avg(tx_capex_mw_mile_dict, row.trans_lz1, row.trans_lz2),
        axis=1,
    )
    transm_final["trans_terrain_multiplier"] = transm_final["region_avgs"].apply(
        lambda x: x / trans_capital_cost_per_mw_km
    )

    # set as 1 for now
    transm_final["trans_new_build_allowed"] = 1
    # sort columns
    transm_final = transm_final[
        [
            "TRANSMISSION_LINE",
            "trans_lz1",
            "trans_lz2",
            "trans_length_km",
            "trans_efficiency",
            "existing_trans_cap",
            "trans_dbid",
            "trans_derating_factor",
            "trans_terrain_multiplier",
            "trans_new_build_allowed",
        ]
    ]
    return transm_final


def balancing_areas(
    pudl_engine,
    IPM_regions,
    all_gen,
    quickstart_res_load_frac,
    quickstart_res_wind_frac,
    quickstart_res_solar_frac,
    spinning_res_load_frac,
    spinning_res_wind_frac,
    spinning_res_solar_frac,
):
    """
    Function to create balancing_areas and zone_balancing_area tables
    Input:
        1) pudl_engine from init_pudl_connection
        2) IPM regions from settings.get('model_regions')
        3) all_gen pandas dataframe from gc.create_all_generators()
        4) quickstart_res_load_frac, quickstart_res_wind_frac, quickstart_res_solar_frac,
            spinning_res_load_frac, spinning_res_wind_frac, and spinning_res_solar_frac:
            --> set these equal to values based on REAM
    Output:
        BALANCING_AREAS
            * BALANCING_AREAS: based on balancing authority from pudl and connecting that to all_gen using plant_id_eia
            * other columns based on REAM Scenario 178
        ZONE_BALANCING_AREAS
            * Load_zone: IPM region
            * balancing_area
    """
    # get table from PUDL that has  balancing_authority_code_eia
    plants_entity_eia = pd.read_sql_table("plants_entity_eia", pudl_engine)
    # dataframe with only balancing_authority_code_eia and plant_id_eia
    plants_entity_eia = plants_entity_eia[
        ["balancing_authority_code_eia", "plant_id_eia"]
    ]
    # create a dictionary that has plant_id_eia as key and the balancing authority as value
    plants_entity_eia_dict = plants_entity_eia.set_index("plant_id_eia").T.to_dict(
        "list"
    )

    plant_region_df = all_gen.copy()
    plant_region_df = plant_region_df[["plant_id_eia", "region"]]

    # get rid of NAs
    plant_region_df = plant_region_df[plant_region_df["plant_id_eia"].notna()]

    """
    BALANCING_AREAS:
    take the plant_id_eia column from all_gen input, and return the balancing authority using 
        the PUDL plants_entity_eia dictionary
    
    """
    # define function to get balancing_authority_code_eia from plant_id_eia
    def id_eia_to_bal_auth(plant_id_eia, plants_entity_eia_dict):
        if plant_id_eia in plants_entity_eia_dict.keys():
            return plants_entity_eia_dict[plant_id_eia][
                0
            ]  # get balancing_area from [balancing_area]
        else:
            return "-"

    # return balancing_authority_code_eia from PUDL table based on plant_id_eia
    plant_region_df["balancing_authority_code_eia"] = plant_region_df[
        "plant_id_eia"
    ].apply(lambda x: id_eia_to_bal_auth(x, plants_entity_eia_dict))

    # create output table
    balancing_areas = plant_region_df["balancing_authority_code_eia"].unique()
    BALANCING_AREAS = pd.DataFrame(balancing_areas, columns=["BALANCING_AREAS"])
    BALANCING_AREAS["quickstart_res_load_frac"] = quickstart_res_load_frac
    BALANCING_AREAS["quickstart_res_wind_frac"] = quickstart_res_wind_frac
    BALANCING_AREAS["quickstart_res_solar_frac"] = quickstart_res_solar_frac
    BALANCING_AREAS["spinning_res_load_frac"] = spinning_res_load_frac
    BALANCING_AREAS["spinning_res_wind_frac"] = spinning_res_wind_frac
    BALANCING_AREAS["spinning_res_solar_frac"] = spinning_res_solar_frac

    """
    ZONE_BALANCING_AREAS table:
        for each of the IPM regions, find the most common balancing_authority to create table
    """

    zone_b_a_list = list()
    for ipm in IPM_regions:
        region_df = plant_region_df.loc[plant_region_df["region"] == ipm]
        # take the most common balancing authority (assumption)
        bal_aut = mode(region_df["balancing_authority_code_eia"].to_list())
        zone_b_a_list.append([ipm, bal_aut])
    zone_b_a_list.append(["_ALL_ZONES", "."])  # Last line in the REAM inputs
    ZONE_BALANCING_AREAS = pd.DataFrame(
        zone_b_a_list, columns=["LOAD_ZONE", "balancing_area"]
    )

    return BALANCING_AREAS, ZONE_BALANCING_AREAS
