"""
Lorem
"""
import os
import sys
from pathlib import Path
import datetime as dt
import rrdtool
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel,delayed

def check_none(
    rows
    ):
    """_summary_

    Args:
        rows (_type_): _description_

    Returns:
        _type_: _description_
    """
    row_rep = None
    first = True

    for elems in rows:
        if elems[0]:
            if first:
                first = False
                row_rep = [elems[0]]
            else:
                row_rep.append(elems[0])

    return row_rep

def get_min_and_max_date(
    i:int,
    step:int,
    dir_path:str
    )->tuple[dt.datetime,dt.datetime]:
    """Retrieves the earliest and latest date and times across all files.

    Args:
        i (int): _description_
        step (int): _description_
        dir_path (str): _description_

    Returns:
        tuple[datetime.datetime,datetime.datetime]: _description_
    """
    # Function to get the earliest and latest date and times across all files
    min_start_date, max_start_date = sys.maxsize, 0

    for file_path in os.listdir(dir_path):
        rrd = os.path.join(dir_path, file_path)

        if rrd.endswith("rrd"):
            info = rrdtool.info(rrd)
            end_num = info['last_update']

            rra = f"rra[{i}]"
            num_rows = info[rra+'.rows']
            num_pdp_rows = info[rra+'.pdp_per_row']
            ret_step = 300 * int(num_pdp_rows)

            time_balance = (step * num_rows * num_pdp_rows)
            start_num = end_num - time_balance
            if start_num < min_start_date:
                min_start_date = start_num

            if end_num > max_start_date:
                max_start_date = end_num

    return dt.datetime.utcfromtimestamp(min_start_date), dt.datetime.utcfromtimestamp(max_start_date)

def parse_rrds_for_all_snmp_servers(
    snmp_path:str,
    dst:str=None,
    n_jobs:int=-1
    )->pd.DataFrame:
    """Parses all available SNMP servers' RRDs and
       stores the data in a single dataframe with columns
       for server, rrd, data_source, time, and value.

    Args:
        snmp_path (str): The path to the SNMP directoy. That directory
                         should contain directories for different servers. Those
                         directories should then contain RRD files.
        dst (str, .csv, optional): The file path where the resulting dataframe
                                   should be stored. Defaults to None.
        n_jobs (int, optional): The maximum number of concurrently running jobs to use
                                when calling Parallel. Defaults to -1, which uses all
                                CPUs.

    Returns:
        pd.DataFrame: A dataframe with columns
                      for server, rrd, data_source, time, and value.
    """

    def helper(snmp_server_path):

        # Indicates which RRA to use
        rrd_idx = 1

        # Excludes poller-wrapper_count.rrd and poller-wrapper.rrd
        if not os.path.isdir(snmp_server_path):
            return None

        result = {
            'server': [],
            'rrd': [],
            'data_source': [],
            'time': [],
            'value':[]
        }

        for file_path in os.listdir(snmp_server_path):
            rrd = os.path.join(snmp_server_path, file_path)

            # Check if the file is an rrd file
            if rrd.endswith("rrd"):
                # Get header information about the RRD
                info = rrdtool.info(rrd)

                # Only get CDPs from a single RRA
                rra = f"rra[{rrd_idx}]"
                step = info['step']
                end_num = info['last_update']
                n_rows = info[rra+'.rows']
                n_pdp_per_row = info[rra+'.pdp_per_row']
                time_balance = (step * n_rows * n_pdp_per_row)
                start_num = end_num - time_balance
                start_time_stamp = dt.datetime.utcfromtimestamp(start_num)

                # Get averaged data between start_num and end_num
                rrd_data = rrdtool.fetch(rrd, "AVERAGE", '-s', str(start_num),'-e', str(end_num))

                data_sources = rrd_data[1]

                # rows will be a list of n_rows tuples.
                # Remove the last element because it is always filled
                # with None.
                rows = rrd_data[2][:-1]

                # Store the rrd timeframe and values
                if rows:
                    for k, row in enumerate(rows):
                        for j, value in enumerate(row):
                            time = start_time_stamp + dt.timedelta(seconds=step)*k
                            result['server'].append(Path(snmp_server_path).name.replace('.int.autonlab.org',''))
                            result['rrd'].append(Path(rrd).name.replace('.rrd', ''))
                            result['data_source'].append(data_sources[j])
                            result['time'].append(time)
                            result['value'].append(value)

        print(f"Finished parsing {Path(snmp_server_path).name.replace('.int.autonlab.org','')}...")
        return result

    results = Parallel(n_jobs=-1)(
                delayed(helper)(
                    snmp_path + f'/{snmp_server}'
                ) for snmp_server in os.listdir(snmp_path))

    results = pd.DataFrame([i for i in results if i is not None])
    results = results.apply(pd.Series.explode)
    results = results.reset_index(drop=True)
    if dst is not None:
        results.to_csv(dst, index=False)
    return results

def parse_rrd_files_for_snmp_server(
    snmp_server_path:str,
    dst:str=None
    )->pd.DataFrame:
    """Parses a single SNMP server's RRDs and
       stores the data in a single dataframe with columns
       for server, rrd, data_source, time, and value.

    Args:
        snmp_server_path (str): The path to the SNMP server directoy. The
                                directory should contain RRD files.
        dst (str, .csv, optional): The file path where the resulting dataframe
                                   should be stored. Defaults to None.

    Returns:
        pd.DataFrame: A dataframe with columns
                      for server, rrd, data_source, time, and value.
    """

    rrd_idx = 1
    snmp_server = Path(snmp_server_path).name

    # Iterate through snmp files
    result = {
        'server': [],
        'rrd': [],
        'data_source': [],
        'time': [],
        'value':[]
    }
    for file_path in os.listdir(snmp_server_path):
        rrd = os.path.join(snmp_server_path, file_path)

        # Check if the file is an rrd file
        if rrd.endswith("rrd"):
            # Get header information about the RRD
            info = rrdtool.info(rrd)

            # Only get CDPs from a single RRA
            rra = f"rra[{rrd_idx}]"
            step = info['step']
            end_num = info['last_update']
            n_rows = info[rra+'.rows']
            n_pdp_per_row = info[rra+'.pdp_per_row']
            time_balance = (step * n_rows * n_pdp_per_row)
            start_num = end_num - time_balance
            start_time_stamp = dt.datetime.utcfromtimestamp(start_num)

            # Get averaged data between start_num and end_num
            rrd_data = rrdtool.fetch(rrd, "AVERAGE", '-s', str(start_num),'-e', str(end_num))

            data_sources = rrd_data[1]

            # rows will be a list of n_rows tuples. Each tuple contains 4 values
            # corresponding to read, written, reads, writes.
            rows = rrd_data[2]

            # Store the rrd timeframe and values
            if rows:
                for k, row in enumerate(rows):
                    for j, value in enumerate(row):
                        time = start_time_stamp + dt.timedelta(seconds=step)*k
                        result['server'].append(snmp_server.replace('.int.autonlab.org',''))
                        result['rrd'].append(Path(rrd).name.replace('.rrd', ''))
                        result['data_source'].append(data_sources[j])
                        result['time'].append(time)
                        result['value'].append(value)

    result = pd.DataFrame(result)
    if dst is not None:
        result.to_csv(dst, index=False)
    return result

def get_available_rrd_names_for_server(
    rrd_path:str,
    monitor:str,
    server:str
    )->np.array:
    """Retrieves the names of available RRDs for a server.

    Args:
        rrd_path (str): The path to the rrd directoy.
        monitor (str): The type of monitoring system that the data
                       was gathered by. Options are snmp or collectd.
        server (str): The name of the server that the data should be retrieved from.

    Returns:
        np.array: An array containing the available RRDs for the server.
    """
    parsed_rrd = pd.read_csv(rrd_path + f'/parsed/{monitor}/parsed_data.csv')
    return parsed_rrd[parsed_rrd['server'] == server]['rrd'].unique()

def get_time_series_data_for_single_rrd(
    rrd_path:str,
    monitor:str,
    server:str,
    rrd:str
    )->pd.DataFrame:
    """Builds a dataframe of time series data where
       indices are timestamps and columns are
       the data sources contained in the RRD.

    Args:
        rrd_path (str): The path to the rrd directoy.
        monitor (str): The type of monitoring system that the data
                       was gathered by. Options are snmp or collectd.
        server (str): The name of the server that the data should be retrieved from.
        rrd (str): The name of the rrd whose data should be loaded into a time series.

    Returns:
        pd.DataFrame: A dataframe of time series data where
                      indices are timestamps and columns are
                      the data sources contained in the RRD.
    """
    parsed_df = pd.read_csv(rrd_path + f'/parsed/{monitor}/parsed_data.csv')
    pivoted_df = parsed_df[(parsed_df['server'] == server) & (parsed_df['rrd'] == rrd)].pivot(index='time', columns='data_source', values='value')
    return pivoted_df

def get_number_of_features_for_each_snmp_server(
    rrd_path:str,
    monitor:str
    )->pd.DataFrame:
    """Retrieves the number of features per server.

    Args:
        rrd_path (str): The path to the rrd directoy.
        monitor (str): The type of monitoring system that the data
                       was gathered by. Options are snmp or collectd.

    Returns:
        pd.DataFrame: A dataframe mapping servers to their number of features.
    """
    parsed_df = pd.read_csv(rrd_path + f'/parsed/{monitor}/parsed_data.csv')

    n_features_per_server = {}
    for server in parsed_df['server'].unique():
        n_features = parsed_df[parsed_df['server'] == server].drop_duplicates(subset=['rrd', 'data_source']).shape[0]
        n_features_per_server[server] = n_features

    n_features_per_server_df = pd.DataFrame.from_dict(n_features_per_server, orient='index', columns=['n_features'])
    n_features_per_server_df.to_csv(rrd_path + f'/parsed/{monitor}/n_features_per_server.csv')
    return n_features_per_server_df

def select_rrd_data_by_server(
    rrd_path:str,
    monitor:str,
    servers:list[str],
    lazy_load:bool=False,
    chunk_size:int=10000
    )->pd.DataFrame:
    """Filters the parsed data by server so that the resulting
       dataframe contains data from a single server's RRDs.

    Args:
        rrd_path (str): The path to the rrd directoy.
        monitor (str): The type of monitoring system that the data
                       was gathered by. Options are snmp or collectd.
        server (list[str]): A list of the servers that the data should be retrieved from.
        lazy_load (bool, optional): A boolean indicating whether to perform a lazy load.
                                    The lazy load will load the parsed data by chunks and
                                    filter each chunk, this is ideal when RAM is limited.
                                    Defaults to False.
        chunk_size (int, optional): The number of rows to load at a time. Defaults to 10000.

    Returns:
        pd.DataFrame: A dataframe containing data from a single server's RRDs.
                      There are four columns: server, rrd, time, and value.
    """
    if lazy_load:
        dataframes = []
        # Define a generator to lazily read the CSV file in chunks
        reader = pd.read_csv(rrd_path + f'/parsed/{monitor}_parsed_data.csv', chunksize=chunk_size)

        # Iterate over the chunks and filter rows by the server column
        for chunk in reader:
            filtered_chunk = chunk[chunk['server'].isin(servers)]
            if not filtered_chunk.empty:
                dataframes.append(filtered_chunk)

        # Build the full dataframe
        return pd.concat(dataframes)

    parsed_df = pd.read_csv(rrd_path + f'/parsed/{monitor}_parsed_data.csv')
    return parsed_df[parsed_df['server'].isin(servers)]

def get_data_from_collectd_server_rrd_files(
    i:int,
    step:int,
    collectd_server_path:str
    )->tuple[set,dict,list,dict,int]:
    """Extracts data from RRD files from every component directory
       within the given colllectd_server_path directory.

    Args:
        i (int): _description_
        step (int): _description_
        collectd_server_path (str): The path to a given collectd server directory which contains
                           various component directories, such as cpu-0, which itself contains
                           rrd files.

    Returns:
        tuple[set,dict,list,dict,int]: _description_
    """
    file_list = set() # only
    value_dict = {}
    garbage_files = []
    date_dict = {}
    ret_step = 0

    # Iterate through the component directories within the collectd_server_path
    for component_directory in os.listdir(collectd_server_path):
        dir_name = os.path.join(collectd_server_path, component_directory)
        # Iterate through each file within the component_directory
        for file_path in os.listdir(dir_name):

            rrd = os.path.join(dir_name, file_path)

            # Split the rrd by hyphens, and join the splits except the last one
            # For example, if rrd = perf-pollermodule-applications,
            # then comp = perf-pollermodule
            comp = ''.join(file_path.split('.')[:-1])
            file_list.add(comp)

            # Check if the file is an RRD file
            if rrd.endswith("rrd"):
                # Get information about the RRD
                info = rrdtool.info(rrd)
                rra = f"rra[{i}]"

                end_num = info['last_update']
                num_rows = info[rra+'.rows']
                num_pdp_rows = info[rra+'.pdp_per_row']
                ret_step = 300 * int(num_pdp_rows)
                time_balance = (step * num_rows * num_pdp_rows)
                start_num = end_num - time_balance
                start_time_stamp = dt.datetime.utcfromtimestamp(start_num)
                end_time_stamp = dt.datetime.utcfromtimestamp(end_num)
                getrrd = rrdtool.fetch(rrd, "AVERAGE", '-s', str(start_num),'-e', str(end_num))
                row = getrrd[2]

                # Store the rrd timeframe and values
                if row:
                    value_dict[os.path.join(component_directory,file_path)] = row
                    date_dict[os.path.join(component_directory,file_path)] = (start_time_stamp, end_time_stamp)
                else:
                    garbage_files.append(file_path)

    return file_list, value_dict, garbage_files, date_dict, ret_step

def generate_snmp_plots(
    value_dict:dict,
    plot_path:str,
    dir_name:str
    ):
    """_summary_

    Args:
        value_dict (dict): _description_
        plot_path (str): _description_
        dir_name (str): _description_
    """
    dir_plot_path = os.path.join(plot_path, dir_name)

    if not os.path.exists(dir_plot_path):
        os.mkdir(dir_plot_path)

    for key, values in value_dict.items():
        title = dir_name + ": " + ''.join(key.split('.')[:-1])
        file_path = os.path.join(dir_plot_path, ''.join(key.split('.')[:-1]))

        plt.plot(values)
        plt.title(title)
        plt.savefig(file_path)