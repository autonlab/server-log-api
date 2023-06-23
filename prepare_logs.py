import datetime as dt
import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm

def determineAvailableData(
    src = Path(__file__).parent / 'data', 
    dst: Path = 'available_data.csv'
    ):
    """Writes to local directory a csv enumerating possible log data from which to draw from. 
       Writes it to `dst`, ensure it is a csv path.

    Args:
        src (_type_, optional): _description_. Defaults to Path(__file__).parent/'data'.
        dst (Path, optional): _description_. Defaults to 'available_data.csv'.
    """
    hosts, dates, logTypes, paths = [], [], [],[]
    i = int()
    for f in src.rglob('*'):
        if (not os.path.isfile(f)):
            continue
        f = Path(f)
        hostname = f.parent.name
        if (f.name.endswith('.gz')):
            base = f.name.split('.')[0]
        else:
            base = f.name
        logType, year, month, day = base.split('-')
        year, month, day = int(year), int(month), int(day)
        date = dt.datetime(year=year, month=month, day=day) 
        hosts.append(hostname)
        dates.append(date)
        logTypes.append(logType)
        paths.append(str(f))
    res = pd.DataFrame({
        'host': hosts,
        'date': dates,
        'log_type': logTypes,
        'file_path': paths
    })
    res.to_csv(dst, index=False)

def parse_all(
    src_df : pd.DataFrame, 
    excluded_types: set = set(), 
    dst: str = 'all_data_content_extracted.csv'
    )->pd.DataFrame:
    """Return dataframe containing all messages in the form `datetime,host,type,content`

    Args:
        src_df (pd.DataFrame): A dataframe containing columns `host,date,log_type`
        excluded_types (set): A set of log types to exclude. Defaults to an empty set.
        dst (str): The file path that the output shoould be saved to. Defaults to 'all_data_content_extracted.csv'.
    
    Returns:
        pd.DataFrame: A dataframe containing all messages in the form `datetime,host,type,content`.
    """
    res_dict = {
        'datetime': list(),
        'host': list(),
        'type': list(),
        'content': list()
    }
    for idx, row in tqdm(src_df.iterrows(), total=len(src_df)):
        host, date, type, path = row['host'], row['date'], row['log_type'], row['file_path']
        if (type in excluded_types):
            continue
        if (path.endswith('.gz')):
            txt_path = path[:-3]
            #gzip this file if it doesn't yet exist
            os.system('gzip -d %s' % path)
        else:
            txt_path = path
        fd = open(txt_path, 'r')
        try:
            file_content = fd.read()
        except:
            fd.close(); del fd
            continue
        fd.close(); del fd
        for line in file_content.splitlines():
            linesplit = line.split()
            datetime = ' '.join(linesplit[:3])
            content = ' '.join(linesplit[3:])
            try:
                parsed_dt = dt.datetime.strptime(datetime, '%b %d %H:%M:%S')
            except:
                continue
            parsed_dt = parsed_dt.replace(year=date.year)

            # Find the ending index of the host name in the content
            start_index = content.index(' ') + 1

            res_dict['datetime'].append(parsed_dt)
            res_dict['host'].append(host)
            res_dict['type'].append(type)
            # Don't include the hostname in the content
            res_dict['content'].append(content[start_index:])
            
    result = pd.DataFrame(res_dict)
    result.to_csv(dst, index=False)
    return result

def filter_logs_by_host(
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

def filter_logs_by_timeframe(
    src_df:pd.DataFrame, 
    start_time:dt.datetime, 
    end_time:dt.datetime
    )->pd.DataFrame:
    """_summary_

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