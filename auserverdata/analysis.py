import matplotlib.pyplot as plt
from matplotlib.dates import HourLocator, DateFormatter
import pandas as pd

def plot_time_series_data(
    feature_df:pd.DataFrame
    ):
    fig, ax = plt.subplots(figsize=(15,6))
    for feature in feature_df.columns:
        feature_values = feature_df[feature].dropna()
        min_value = feature_values.min()
        max_value = feature_values.max()
        feature_values = (feature_values - min_value) / (max_value - min_value)
        timestamps = [pd.to_datetime(x) for x in list(feature_values.index)]

        plt.plot(timestamps, feature_values);

    # Set the x-axis locator to show hours
    ax.xaxis.set_major_locator(HourLocator(interval=1))  # This sets the tick interval to 1 hour

    # Format the x-axis tick labels as hour:minute
    ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))

    plt.show()