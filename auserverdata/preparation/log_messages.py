"""
Lorem
"""

import datetime as dt
import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm

def determine_available_data(
    src: Path,
    dst: Path = 'available_log_messages.csv'
    ):
    """Writes to local directory a csv enumerating possible log message data from which to draw from.
       The csv contains the following columns: host, date, log_type, file_path.
       Writes it to `dst`, ensure it is a csv path.

    Args:
        src (_type_, optional): _description_. Defaults to Path(__file__).parent/'data'.
        dst (Path, optional): _description_. Defaults to 'available_log_messages.csv'.
    """
    hosts, dates, log_types, paths = [], [], [], []

    for file_path in src.rglob('*'):
        if (not os.path.isfile(file_path)):
            continue
        file_path = Path(file_path)
        hostname = file_path.parent.name
        if (file_path.name.endswith('.gz')):
            base = file_path.name.split('.')[0]
        else:
            base = file_path.name
        log_type, year, month, day = base.split('-')
        year, month, day = int(year), int(month), int(day)
        date = dt.datetime(year=year, month=month, day=day)
        hosts.append(hostname)
        dates.append(date)
        log_types.append(log_type)
        paths.append(str(file_path))
    res = pd.DataFrame({
        'host': hosts,
        'date': dates,
        'log_type': log_types,
        'file_path': paths
    })
    res.to_csv(dst, index=False)

def parse_all(
    src_df : pd.DataFrame,
    excluded_types: set(),
    dst: str = 'all_log_messages_extracted.csv'
    )->pd.DataFrame:
    """Returns a dataframe with the following columns: datetime, host, log_type, log_message

    Args:
        src_df (pd.DataFrame): A dataframe containing the following columns: host, date, log_type, file_path
        excluded_types (set): A set of log types to exclude. Defaults to an empty set.
        dst (str): The file path that the output shoould be saved to. Defaults to 'all_log_messages_extracted.csv'.

    Returns:
        pd.DataFrame: A dataframe containing all messages in the form `datetime,host,log_type,log_message`.
    """
    result_dict = {
        'datetime': [],
        'host': [],
        'log_type': [],
        'log_message': []
    }
    for _, row in tqdm(src_df.iterrows(), total=len(src_df)):
        host, date, log_type, file_path = row['host'], row['date'], row['log_type'], row['file_path']

        if (log_type in excluded_types):
            continue

        # Check if the file needs to be gzip'd
        if (file_path.endswith('.gz')):
            file_path = file_path[:-3]
            os.system(f'gzip -d {file_path}')

        # Open the log message file
        log_message_file = open(file_path, 'r')
        try:
            log_messages = log_message_file.read()
        except (OSError, UnicodeDecodeError):
            log_message_file.close()
            del log_message_file
            continue
        log_message_file.close();
        del log_message_file

        # Iterate through each line of the log_messages
        for line in log_messages.splitlines():
            linesplit = line.split()
            datetime = ' '.join(linesplit[:3])
            content = ' '.join(linesplit[3:])
            try:
                parsed_dt = dt.datetime.strptime(datetime, '%b %d %H:%M:%S')
            except ValueError:
                continue
            parsed_dt = parsed_dt.replace(year=date.year)

            result_dict['datetime'].append(parsed_dt)
            result_dict['host'].append(host)
            result_dict['log_type'].append(log_type)
            # Find the ending index of the host name in the content
            start_index = content.index(' ') + 1
            # Don't include the hostname in the content
            result_dict['log_message'].append(content[start_index:])

    result = pd.DataFrame(result_dict)
    result.to_csv(dst, index=False)
    return result

def filter_log_messages_by_host(
    csv_file:str,
    host:str,
    chunk_size:int=10000
    )->pd.DataFrame:
    """This function iteratively lazy loads the full server log csv by chunks
       and filters based on a given host.

    Args:
        csv_file (str): The filepath to the full csv file containing all log data.
        host (str): The host name that you want to filter by.
        chunk_size (int): The size of the chunks to load iteratively. Defaults to 10000.

    Returns:
        pd.DataFrame: A dataframe only containing logs for the given host.
    """
    chunk_size = 10000  # Number of rows to process at a time
    dataframes = []
    # Define a generator to lazily read the CSV file in chunks
    reader = pd.read_csv(csv_file, chunksize=chunk_size)

    # Iterate over the chunks and filter rows by the host column
    for chunk in reader:
        filtered_chunk = chunk[chunk['host'] == host]
        if not filtered_chunk.empty:
            dataframes.append(filtered_chunk)

    # Build the full dataframe
    filtered_dataframe = pd.concat(dataframes)

    return filtered_dataframe

def filter_log_messages_by_timeframe(
    src_df:pd.DataFrame,
    start_time:dt.datetime,
    end_time:dt.datetime
    )->pd.DataFrame:
    """This function filters the given dataframe to only include
       data between [start_time, end_time].

    Args:
        src_df (pd.DataFrame): A dataframe containing the server log data.
        start_time (dt.datetime): The start time that you want to filter by.
        end_time (dt.datetime): The end time that you want to filter by.

    Returns:
        pd.DataFrame: A filtered dataframe of server log data where each log is between
                      the start_time and end_time.
    """
    # Convert the datetime column to Python's datetime format
    src_df['datetime'] = pd.to_datetime(src_df['datetime'])

    # Filter the dataframe based on the time range
    filtered_dataframe = src_df[(src_df['datetime'] >= start_time) & (src_df['datetime'] <= end_time)]

    return filtered_dataframe