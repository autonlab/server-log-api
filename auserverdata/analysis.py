import matplotlib.pyplot as plt
from matplotlib.dates import HourLocator, DateFormatter
import pandas as pd
import numpy as np

def plot_time_series_data_on_single_plot(
    features:pd.DataFrame
    ):
    fig, ax = plt.subplots(figsize=(15,6))
    if isinstance(features, pd.DataFrame):
        for feature in features.columns:
            feature_values = features[feature].dropna()
            min_value = feature_values.min()
            max_value = feature_values.max()
            feature_values = (feature_values - min_value) / (max_value - min_value)

            plt.plot(feature_values.index, feature_values, label = feature);
    else:
        feature_values = features.dropna()
        min_value = feature_values.min()
        max_value = feature_values.max()
        feature_values = (feature_values - min_value) / (max_value - min_value)
        plt.plot(feature_values.index, feature_values);
        plt.legend(loc = 'center left', bbox_to_anchor =  (1, 0.5))
        plt.ylabel(features.name + '[Normalized]')
        plt.xlabel('Timestep')
    plt.show()

def plot_stacked_single_channel_data_on_single_plot(
    stacked_data_list:list[pd.Series],
    channel_name:str,
    normalize:bool
    ):
    fig, ax = plt.subplots(figsize=(18,6))
    timesteps_for_vlines = []
    end_timestep = 0
    for i, data in enumerate(stacked_data_list):
        if i != len(stacked_data_list) - 1:
            end_timestep = end_timestep + len(data)
            timesteps_for_vlines.append(end_timestep)
    data = pd.concat(stacked_data_list)
    data.index = [x for x in range(len(data))]
    data = data.dropna()
    if normalize:
        min_value = data.min()
        max_value = data.max()
        data = (data - min_value) / (max_value - min_value)
        plt.ylabel(f'{channel_name} [Normalized]')
    else:
        plt.ylabel(f'{channel_name}')
    min_value = data.min()
    max_value = data.max()
    plt.plot(data.index, data);
    plt.legend(loc = 'center left', bbox_to_anchor =  (1, 0.5))

    plt.xlabel('Timestep [10s Steps]')

    for timestep in timesteps_for_vlines:
        plt.vlines(timestep, ymin=min_value-((min_value+1e-2) * 0.06), ymax=max_value+((max_value+1e-2) * 0.06), colors='black', linestyles='dotted', linewidth = 3)
    plt.ylim(min_value-((min_value+1) * 0.05), max_value+((max_value+1e-2) * 0.05))
    plt.show()

def plot_time_series_data_in_vertical_stack(
    feature_df:pd.DataFrame
    ):
    fig, ax = plt.subplots(ncols=1, nrows=len(feature_df.columns), figsize=(12,2*len(feature_df.columns)))
    for i, feature in enumerate(feature_df.columns):
        feature_values = feature_df[feature].dropna()
        min_value = feature_values.min()
        max_value = feature_values.max()
        feature_values = (feature_values - min_value) / (max_value - min_value)
        timestamps = [pd.to_datetime(x) for x in list(feature_values.index)]

        ax[i].plot(timestamps, feature_values, label = feature)
        ax[i].legend(loc = 'center right', bbox_to_anchor = (1.28, 0.7))

        # Set the x-axis locator to show hours
        ax[i].xaxis.set_major_locator(HourLocator(interval=1))  # This sets the tick interval to 1 hour

        # Format the x-axis tick labels as hour:minute
        ax[i].xaxis.set_major_formatter(DateFormatter('%H:%M'))

    plt.show()