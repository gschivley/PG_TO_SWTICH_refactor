from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from powergenome.generators import GeneratorClusters
from powergenome.GenX import reduce_time_domain
from powergenome.load_profiles import make_final_load_curves
from powergenome.params import DATA_PATHS
from powergenome.util import (
    build_scenario_settings,
    init_pudl_connection,
    load_settings,
    check_settings,
)
from powergenome.external_data import (
    make_demand_response_profiles,
    make_generator_variability,
)

pd.options.display.max_columns = 200

cwd = Path.cwd()

settings_path = cwd / "settings_TD_east.yml"
settings = load_settings(settings_path)
settings["input_folder"] = settings_path.parent / settings["input_folder"]
scenario_definitions = pd.read_csv(
    settings["input_folder"] / settings["scenario_definitions_fn"]
)
scenario_settings = build_scenario_settings(settings, scenario_definitions)

pudl_engine, pudl_out, pg_engine = init_pudl_connection(
    freq="AS",
    start_year=min(settings.get("data_years")),
    end_year=max(settings.get("data_years")),
)

s = """
        SELECT time_index, region_id_epaipm, load_mw, year
        FROM load_curves_ferc
    """
load_curves_ferc = pd.read_sql_query(s, pg_engine)
load_wecc2012 = load_curves_ferc.loc[
    load_curves_ferc["region_id_epaipm"].str.contains("WEC")
]
load_wecc2012
load2012 = load_curves_ferc.groupby("time_index").agg(
    {"load_mw": "sum", "year": "first"}
)
s = """
        SELECT time_index, region, sector, subsector, load_mw, year
        FROM load_curves_nrel_efs
    """

#     WHERE region in ('WEC_BANC', 'WEC_CALN', 'WEC_LADW', 'WEC_SDGE', 'WECC_AZ',
#    'WECC_CO', 'WECC_ID', 'WECC_IID', 'WECC_MT', 'WECC_NM', 'WECC_NNV',
#    'WECC_PNW', 'WECC_SCE', 'WECC_SNV', 'WECC_UT', 'WECC_WY')
load_curves_efs = pd.read_sql_query(s, pg_engine)
load_wecc2019 = load_curves_efs.loc[load_curves_efs["region"].str.contains("WEC")]
load2019 = load_wecc2019.groupby("time_index").agg({"load_mw": "sum", "year": "first"})


############# load 2050
import pandas as pd
import os

# import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from pytz import timezone
import pytz

loads = pd.read_csv(
    "/Users/rangrang/Dropbox/PowerGenomeSWITCH_documentation r1/Jupyter Notebooks/test_full/p1/loads.csv"
)
timepoints = pd.read_csv(
    "/Users/rangrang/Dropbox/PowerGenomeSWITCH_documentation r1/Jupyter Notebooks/test_full/p1/timepoints.csv"
)
timeseries = pd.read_csv(
    "/Users/rangrang/Dropbox/PowerGenomeSWITCH_documentation r1/Jupyter Notebooks/test_full/p1/timeseries.csv"
)

timepoints_weighted = pd.merge(
    left=timepoints,
    right=timeseries,
    on=["timeseries"],
    validate="many_to_one",
    how="left",
)

timepoints_weighted = timepoints_weighted.rename({"timepoint_id": "TIMEPOINT"}, axis=1)

loads
timepoints
loads_new = pd.merge(
    left=loads,
    right=timepoints_weighted,
    on=["TIMEPOINT"],
    validate="many_to_one",
    how="left",
)

loads_new["demand_mw"] = (
    loads_new["zone_demand_mw"] * loads_new["ts_scale_to_period"] / 10
)
loads_new["period"] = loads_new["timestamp"].astype(str).str[:4]
loads_new = loads_new.loc[loads_new["period"] == "2050"]
loads_new["hour"] = loads_new["TIMEPOINT"] - 17520

toplot = loads_new.groupby("hour", as_index=False).agg(
    {"demand_mw": "sum", "hour": "first"}
)
import pylab as plt

X = range(1, 8761)
Y1 = load2012["load_mw"].iloc[
    0:8760,
]
# Y2 = load2019["load_mw"]
Y3 = toplot["demand_mw"]
plt.xlim([1, 8760])
plt.scatter(X, Y1 / 1000, color="yellow", s=2)
# plt.scatter(X, Y2 / 1000, color="g", s=2)
plt.scatter(X, Y3 / 1000, color="pink", s=2)
plt.title("Annual demand -- wecc")
# Label configuration
plt.xlabel("Hours in a year", fontsize=9)
plt.ylabel("GW", fontsize=9)
plt.yticks(fontsize=9)
plt.xticks(fontsize=9)
# plt.legend(["2012", "2019", "2050"])
plt.legend(["2012", "2050"])

plt.show()
