
a = [1,2,3]
b = [5,6,7]
range(a : b)
a + 1


def regional_capex_multiplier(
    df: pd.DataFrame,
    region: str,
    region_map: Dict[str, str],
    tech_map: Dict[str, str],
    regional_multipliers: pd.DataFrame,
) -> pd.DataFrame:

    cost_region = region_map[region]
    tech_multiplier = regional_multipliers.loc[cost_region, :].squeeze()
    avg_multiplier = tech_multiplier.mean()

    tech_multiplier = tech_multiplier.fillna(avg_multiplier)

    tech_multiplier_map = {}
    for atb_tech, eia_tech in tech_map.items():
        if df["technology"].str.contains(atb_tech).sum() > 0:
            full_atb_tech = df.loc[
                df["technology"].str.contains(atb_tech).idxmax(), "technology"
            ]
            tech_multiplier_map[full_atb_tech] = tech_multiplier.at[eia_tech]
        if df["technology"].str.contains(atb_tech).sum() > 1:
            s = f"""
    ***************************
    There is an issue with assigning regional cost multipliers. In your settings file
    under the parameter 'cost_multiplier_technology_map`, the EIA technology '{eia_tech}'
    has an ATB technology '{atb_tech}'. This ATB name matches more than one new ATB tech
    listed in the settings parameter 'atb_new_gen'. Only the first matching tech in
    'atb_new_gen' will get a valid regional cost multiplier; the rest will have values of
    0, which will lead to annual investment costs of $0.
        """
            logger.warning(s)
    df["Inv_Cost_per_MWyr"] *= df["technology"].map(tech_multiplier_map)
    df["Inv_Cost_per_MWhyr"] *= df["technology"].map(tech_multiplier_map)
    df["regional_cost_multiplier"] = df["technology"].map(tech_multiplier_map)

    return df




def add_renewables_clusters(
    df: pd.DataFrame,
    region: str,
    settings: dict,
    cluster_builder: ClusterBuilder = None,
) -> pd.DataFrame:
    """
    Add renewables clusters

    Parameters
    ----------
    df
        New generation technologies.
            - `technology`: NREL ATB technology in the format
                <technology>_<tech_detail>_<cost_case>. Must be unique.
            - `region`: Model region.
    region
        Model region.
    settings
        Dictionary with the following keys:
            - `renewables_clusters`: Determines the clusters built for the region.
            - `region_aggregations`: Maps the model region to IPM regions.
    cluster_builder
        ClusterBuilder object. Reuse to save time. None by default.

    Returns
    -------
    pd.DataFrame
        Copy of the input dataframe joined to rows for renewables clusters
        on matching NREL ATB technology and model region.

    Raises
    ------
    ValueError
        NREL ATB technologies are not unique.
    ValueError
        Renewables clusters do not match NREL ATB technologies.
    ValueError
        Renewables clusters match multiple NREL ATB technologies.
    """
    if not df["technology"].is_unique:
        raise ValueError(
            f"NREL ATB technologies are not unique: {df['technology'].to_list()}"
        )
    atb_map = {
        x: map_nrel_atb_technology(x.split("_")[0], x.split("_")[1])
        for x in df["technology"]
    }
    mask = df["technology"].isin([tech for tech, match in atb_map.items() if match]) & (
        df["region"] == region
    )
    cdfs = []
    if region in (settings.get("region_aggregations", {}) or {}):
        ipm_regions = settings.get("region_aggregations", {})[region]
        ipm_regions.append(region)  # Add model region, sometimes listed in RG file
    else:
        ipm_regions = [region]
    for scenario in settings.get("renewables_clusters", []) or []:
        if scenario["region"] != region:
            continue
        # Match cluster technology to NREL ATB technologies
        technologies = [
            k
            for k, v in atb_map.items()
            if v and all([scenario.get(ki) == vi for ki, vi in v.items()])
        ]
        if not technologies:
            raise ValueError(
                f"Renewables clusters do not match NREL ATB technologies: {scenario}"
            )
        if len(technologies) > 1:
            raise ValueError(
                f"Renewables clusters match multiple NREL ATB technologies: {scenario}"
            )
        technology = technologies[0]
        # region not an argument to ClusterBuilder.get_clusters()
        scenario = scenario.copy()
        scenario.pop("region")
        if not cluster_builder:
            cluster_builder = build_resource_clusters(settings.get("RESOURCE_GROUPS"))
        clusters = (
            cluster_builder.get_clusters(
                **scenario,
                ipm_regions=ipm_regions,
                existing=False,
                utc_offset=settings.get("utc_offset", 0),
            )
            .rename(columns={"mw": "Max_Cap_MW"})
            .assign(technology=technology, region=region)
        )
        clusters["cluster"] = range(1, 1 + len(clusters))
        if scenario.get("min_capacity"):
            # Warn if total capacity less than expected
            capacity = clusters["Max_Cap_MW"].sum()
            if capacity < scenario["min_capacity"]:
                logger.warning(
                    f"Selected technology {scenario['technology']} capacity"
                    + f" in region {region}"
                    + f" less than minimum ({capacity} < {scenario['min_capacity']} MW)"
                )
        row = df[df["technology"] == technology].to_dict("records")[0]
        new_tech_name = "_".join(
            [
                str(v)
                for k, v in scenario.items()
                if k not in ["region", "technology", "max_clusters", "min_capacity"]
            ]
        )
        clusters["technology"] = clusters["technology"] + "_" + new_tech_name
        kwargs = {k: v for k, v in row.items() if k not in clusters}
        cdfs.append(clusters.assign(**kwargs))
    return pd.concat([df[~mask]] + cdfs, sort=False)




