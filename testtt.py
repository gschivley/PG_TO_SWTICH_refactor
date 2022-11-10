

def gen_build_predetermined(
    all_gen,
    pudl_gen,
    pudl_gen_entity,
    pg_build,
    manual_build_yr,
    eia_Gen,
    eia_Gen_prop,
    plant_gen_manual,
    plant_gen_manual_proposed,
    plant_gen_manual_retired,
    retirement_ages,
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
            - Operating Year
            - retirement_year
        5) manual_build_yr: dictionary of build years that were found manually (outside of PUDL and PG)
        6) eia_Gen: eia operable plants
        7) eia_Gen_prop: eia proposed plants
        8) plant_gen_manual, plant_gen_manual_proposed, plant_gen_manual_retired: manually found build_years
        9) retirement_ages: how many years until plant retires
    Output columns
        * GENERATION_PROJECT: index from all_gen
        * build_year: using pudl_gen, pudl_gen_entity, eia excel file, and pg_build to get years
        * gen_predetermined_cap: based on Cap_Size from all_gen
        * gen_predetermined_storage_energy_mwh: based on capex_mwh from all_gen
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
    PG_op_yr_dict = create_dict_plantgen(pg_build, "Operating Year")

    #  create dictionaries {plant_gen_id: date} from eia excel file
    eia_Gen_dict = create_dict_plantgen(eia_Gen, "Operating Year")
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
        "Operating Year",
        "planned_operating_year",
        "manual_yr",
        "PG_op_yr",
        "eia_gen_op_yr",
        "eia_gen_manual_yr",
        "proposed_year",
        "proposed_manual_year",
    ]
    pg_build["build_final"] = pg_build[op_columns].max(axis=1)
    # get all build years into one column (includes manual dates and proposed dates)

    plant_unit_tech = all_gen.dropna(subset=["plant_pudl_id"])[
        ["plant_pudl_id", "technology"]
    ]
    plant_unit_tech = plant_unit_tech.drop_duplicates(subset=["plant_pudl_id"])
    plant_unit_tech = plant_unit_tech.set_index("plant_pudl_id")["technology"]
    pg_build["technology"] = pg_build["plant_pudl_id"].map(plant_unit_tech)
    pg_build["retirement_age"] = pg_build["technology"].map(retirement_ages)
    pg_build["calc_retirement_year"] = (
        pg_build["build_final"] + pg_build["retirement_age"]
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
    pg_build["retire_year_final"] = pg_build[ret_columns].min(axis=1)

    """
    Start creating the gen_build_predetermined table
    """
    # base it off of PowerGenome all_gen
    gen_buildpre = all_gen.copy()
    gen_buildpre = gen_buildpre[
        [
            "index",
            "plant_id_eia",
            "Cap_Size",
            "capex_mwh",
            "region",
            "plant_pudl_id",
            "technology",
        ]
    ]

    # based GENERATION_PROJECT off of the index of all_gen
    gen_buildpre["GENERATION_PROJECT"] = gen_buildpre.index + 1

    # this ignores new builds
    new_builds = gen_buildpre[gen_buildpre["index"].isna()]
    gen_buildpre = gen_buildpre[gen_buildpre["index"].notna()]

    # create dictionary to go from pg_build to gen_buildpre (build_year)
    pg_build_buildyr = create_dict_plantpudl(pg_build, "build_final")
    gen_buildpre["build_year"] = gen_buildpre["plant_pudl_id"].apply(
        lambda x: plant_dict(x, pg_build_buildyr)
    )

    # create dictionary to go from pg_build to gen_buildpre (retirement_year)
    pg_retireyr = pg_build["retire_year_final"].to_list()
    pg_build_retireyr = create_dict_plantpudl(pg_build, "retire_year_final")
    gen_buildpre["retirement_year"] = gen_buildpre["plant_pudl_id"].apply(
        lambda x: plant_dict(x, pg_build_retireyr)
    )

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
    #     new_builds = new_builds[['GENERATION_PROJECT', 'Cap_Size', 'capex_mwh']]
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
        ["GENERATION_PROJECT", "build_year", "Cap_Size", "capex_mwh"]
    ]

    # don't include new builds
    #     gen_buildpre_combined = pd.concat([gen_buildpre, new_builds2020, new_builds2030, new_builds2040, new_builds2050],
    #                                      ignore_index=True)
    #     gen_buildpre = gen_buildpre.append([new_builds2020, new_builds2030, new_builds2040, new_builds2050],
    #                                        ignore_index=True)

    gen_buildpre.rename(
        columns={
            "Cap_Size": "gen_predetermined_cap",
            "capex_mwh": "gen_predetermined_storage_energy_mwh",
        },
        inplace=True,
    )
    # based on REAM
    gen_buildpre["gen_predetermined_storage_energy_mwh"] = gen_buildpre[
        "gen_predetermined_storage_energy_mwh"
    ].fillna(".")

    gen_buildpre["build_year"] = gen_buildpre["build_year"].astype(float).astype(int)
    #     gen_buildpre['GENERATION_PROJECT'] = gen_buildpre['GENERATION_PROJECT'].astype(str)

    # SWITCH doesn't like having build years that are in the period
    gen_buildpre.drop(
        gen_buildpre[gen_buildpre["build_year"] == 2020].index, inplace=True
    )

    return gen_buildpre, gen_build_with_id