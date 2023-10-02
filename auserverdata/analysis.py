import matplotlib.pyplot as plt
from matplotlib.dates import HourLocator, DateFormatter
import pandas as pd

def plot_time_series_data_on_single_plot(
    feature_df:pd.DataFrame
    ):
    fig, ax = plt.subplots(figsize=(15,6))
    for feature in feature_df.columns:
        feature_values = feature_df[feature].dropna()
        min_value = feature_values.min()
        max_value = feature_values.max()
        feature_values = (feature_values - min_value) / (max_value - min_value)
        timestamps = [pd.to_datetime(x) for x in list(feature_values.index)]

        plt.plot(timestamps, feature_values, label = feature);

    # Set the x-axis locator to show hours
    ax.xaxis.set_major_locator(HourLocator(interval=1))  # This sets the tick interval to 1 hour

    # Format the x-axis tick labels as hour:minute
    ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
    plt.legend(loc = 'center left', bbox_to_anchor =  (1, 0.5))
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