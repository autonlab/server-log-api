"""
Lorem
"""

import regex as re
import pandas as pd

def featurize_by_time_interval_log_format_matching(
    filtered_df:pd.DataFrame,
    t_interval:str='60S',
    src:str='regex_list.txt',
    )->pd.DataFrame:
    """Featurizes the log messages for a single host by splitting the data into
       separate time intervals (determined by t_interval) and counting the number of occurrences
       of different formats of log messages within those time intervals.

    Args:
        filtered_df (pd.DataFrame): A dataframe that contains data from a single host.
        t_interval (str, optional): The time interval that the data should be split into. Defaults to '60S'.
        src (str, optional): The file path where the regular expressions are saved,
                             should be a .txt file. Defaults to 'regex_list.txt'.

    Returns:
        pd.DataFrame: A dataframe with the interval start times as rows, the regular expressions
                      as columns, and the number of occurrences of each regular expression within
                      the time interval as values.
    """

    filtered_df.sort_values('datetime')
    filtered_df.set_index('datetime', inplace=True)
    resampled_df = filtered_df['content'].resample(t_interval)
    regex_list = load_regex_list(src)
    counts_per_time_interval = {}
    # Loop through the log messages in each time interval
    for interval_start, group_df in resampled_df:
        counts = []
        # Loop through each regular expression
        for regex in regex_list:
            # Count the number of occurrences of the regular expression in the group of
            # log messages
            regex_counts = group_df.str.contains(regex).sum()
            counts.append(regex_counts)
        counts_per_time_interval[interval_start] = counts

    result_df = pd.DataFrame.from_dict(counts_per_time_interval, orient='index')
    result_df.columns = regex_list

    return result_df

def build_and_save_regex_list(
    dst:str='regex_list.txt'
    ):
    """Build a list of regular expressions matching common log message
       formats and save it to a .txt file.

    Args:
        dst (str, optional): The file path where the regular expressions should be saved,
                             should be a .txt file. Defaults to 'regex_list.txt'.
    """

    regex_list = ['^CROND\[\d+\]:\s\(root\)\s.*$',
                '^kernel: perf: *',
                '^kernel: md: *',
                '^kernel: nvidia *',
                '^kernel: raid6: *',
                '^kernel: XFS *',
                '^kernel: Process accounting resumed',
                '^kernel: md*',
                '^kernel: ata1: *',
                '^kernel: ata1.00: failed command: *',
                '^kernel: ata1.00: error: *',
                '^kernel: ata1.00: exception*',
                '^kernel: ata1.00: cmd*',
                '^kernel: ata1.00: status*',
                '^kernel: ata1.00: sd*',
                '^kernel: blk_update_request:*',
                '^kernel: sd*',
                '^abrt-server:*',
                '^python:*'
                ]

    # Saving the regular expressions to a text file
    with open("regex_list.txt", "w") as file:
        for regex in regex_list:
            file.write(regex + "\n")

    return

def load_regex_list(
    src:str='regex_list.txt'
    )->list[str]:
    """Load a list of regular expressions matching common log message
       formats.

    Args:
        src (str, optional): The file path where the regular expressions are saved, should be a .txt file.
                             Defaults to 'regex_list.txt'.

    Returns:
        list[str]: A list of regular expressions matching common log message
                   formats.
    """
    # Reading the regular expressions from the text file
    with open(src, "r") as file:
        loaded_regex_list = [line.strip() for line in file]

    return loaded_regex_list