from statistics import mean, mode
from typing import List

import numpy as np
import pandas as pd
import math

from powergenome.time_reduction import kmeans_time_clustering



def timeseries_full(
    load_curves,
    planning_year,
    planning_start_year,
    settings,
):  # 20.2778, 283.8889
    """Create timeseries and timepoints tables when using yearly data with 8760 hours
    Apply this function reduce_time_domain: False & full_time_domain: True in settings
    Parameters
    ----------
    planning_periods : List[int]
        A list of the planning years
    planning_period_start_years : List[int]
        A list of the start year for each planning period, used to calculate the number
        of years in each period

    Returns
    -------
    pd.DataFrame, pd.DataFrame
        A tuple of the timeseries and timepoints dataframes
    """

    if settings.get("sample_dates_fn") and settings.get("input_folder"):
        sample_dates = pd.read_csv(
            settings.get("input_folder") / settings["sample_dates_fn"]
        )
    else:
        sample_year = planning_year
        sample_year_start = str(sample_year) + "0101"
        sample_year_end = str(sample_year) + "1231"
        sample_dates = [
            d.strftime("%Y%m%d")
            for d in pd.date_range(sample_year_start, sample_year_end)
        ]

    leap_yr = str(sample_year) + "0229"
    if leap_yr in sample_dates:
        sample_dates.remove(leap_yr)  ### why remove Feb 29th? --RR
    num_days = len(sample_dates)
    sample_to_year_ratio = 8760 / (num_days * 24)
    planning_years = settings.get("planning_years")

    timeseries_df = pd.DataFrame()
    timeseries_df["timeseries"] = [
        x[:4] + "_" + x[:4] + "-" + x[4:6] + "-" + x[6:8] for x in sample_dates
    ]
    timeseries_df["ts_period"] = [x[:4] for x in sample_dates]
    timeseries_df["ts_duration_of_tp"] = 1  # each hour as one timepoint
    timeseries_df["ts_num_tps"] = 24  # 24 hours
    timeseries_df["ts_scale_to_period"] = planning_years * sample_to_year_ratio

    timeseries_dates = timeseries_df["timeseries"].to_list()
    timestamp_interval = list()
    for i in range(24):
        s_interval = i
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
