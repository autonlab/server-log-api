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

def get_data_from_rrd(
    rrd:str,
    )->tuple[tuple, list[tuple]]:
    """Extracts data from an RRD.

    Args:
        rrd (str): The name of the RRD file.

    Returns:
        tuple[int, dt.datetime, tuple, list[tuple]]: The integer indicates the step the RRD used.
                                                     The datetime is the start time stamp
                                                     of the RRD. The tuple contains the data source names.
                                                     Each tuple in the list of tuples contains the values
                                                     of the data sources at a timestamp.

    """
    # Indicates which RRA to use
    rrd_idx = 1

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

    # Remove the last element because it is always filled
    # with None. rows is a list of n_rows tuples.
    rows = rrd_data[2][:-1]

    return step, start_time_stamp, data_sources, rows

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

    def helper(
        server_path:str
        )->dict:

        # Excludes poller-wrapper_count.rrd and poller-wrapper.rrd
        if not os.path.isdir(server_path):
            return None

        result = {
            'server': [],
            'rrd': [],
            'data_source': [],
            'time': [],
            'value': []
        }

        # Iterate through the server's files
        for file_path in os.listdir(server_path):
            rrd = os.path.join(server_path, file_path)

            # Check if the file is an rrd
            if rrd.endswith("rrd"):
                step, start_time_stamp, data_sources, rows = get_data_from_rrd(rrd)

                # Store the rrd data
                if rows:
                    for k, row in enumerate(rows):
                        for j, value in enumerate(row):
                            # Exclude values that are None
                            if value is None:
                                break
                            time = start_time_stamp + dt.timedelta(seconds=step)*k
                            result['server'].append(Path(server_path).name.replace('.int.autonlab.org',''))
                            result['rrd'].append(Path(rrd).name.replace('.rrd', ''))
                            result['data_source'].append(data_sources[j])
                            result['time'].append(time)
                            result['value'].append(value)

        print(f"Finished parsing {Path(server_path).name.replace('.int.autonlab.org','')}...")
        return result

    # Extract RRD data from each server in parallel
    results = Parallel(n_jobs=n_jobs)(
                delayed(helper)(
                    snmp_path + f'/{server}'
                ) for server in os.listdir(snmp_path))

    # There may be some results that are None because the
    # server path was not a directory, so filter those out.
    results = pd.DataFrame([i for i in results if i is not None])
    results = results.apply(pd.Series.explode)
    results = results.reset_index(drop=True)
    if dst is not None:
        results.to_csv(dst, index=False)
    return results

def parse_rrds_for_all_collectd_servers(
    collectd_path:str,
    dst:str=None,
    n_jobs:int=-1
    )->pd.DataFrame:
    """Parses all available collectd servers' RRDs and
       stores the data in a single dataframe with columns
       for server, component, rrd, data_source, time, and value.

    Args:
        snmp_path (str): The path to the collectd directoy. That directory
                         should contain directories for different servers. Those
                         directories should then contain component directories, which
                         should contain RRD files.
        dst (str, .csv, optional): The file path where the resulting dataframe
                                   should be stored. Defaults to None.
        n_jobs (int, optional): The maximum number of concurrently running jobs to use
                                when calling Parallel. Defaults to -1, which uses all
                                CPUs.

    Returns:
        pd.DataFrame: A dataframe with columns
                      for server, rrd, data_source, time, and value.
    """

    def helper(
        server_path:str
        )->dict:

        # Excludes poller-wrapper_count.rrd and poller-wrapper.rrd
        if not os.path.isdir(server_path):
            return None

        result = {
            'server': [],
            'component': [],
            'rrd': [],
            'data_source': [],
            'time': [],
            'value': []
        }

        # Iterate through the server's components
        for component in os.listdir(server_path):
            component_path = os.path.join(server_path, component)

            # The xen1 server has nearly 10000 interface-vif
            # components. We exclude those due to memory limitations.
            if 'interface-vif' in component:
                print('Skipping interface-vif component...')
                break

            # Iterate through the component's files
            for file_path in os.listdir(component_path):
                rrd = os.path.join(component_path, file_path)

                # Check if the file is an rrd
                if rrd.endswith("rrd"):
                    step, start_time_stamp, data_sources, rows = get_data_from_rrd(rrd)

                    # Store the rrd data
                    if rows:
                        for k, row in enumerate(rows):
                            for j, value in enumerate(row):
                                # Exclude values that are None
                                if value is None:
                                    break
                                time = start_time_stamp + dt.timedelta(seconds=step)*k
                                result['server'].append(Path(server_path).name.replace('.int.autonlab.org',''))
                                result['component'].append(Path(component_path).name)
                                result['rrd'].append(Path(rrd).name.replace('.rrd', ''))
                                result['data_source'].append(data_sources[j])
                                result['time'].append(time)
                                result['value'].append(value)

        print(f"Finished parsing {Path(server_path).name.replace('.int.autonlab.org','')}...")
        return result

    results = Parallel(n_jobs=n_jobs)(
                delayed(helper)(
                    collectd_path + f'/{server}'
                ) for server in os.listdir(collectd_path))

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

def get_available_rrd_names_for_snmp_server(
    rrd_path:str,
    server:str
    )->np.array:
    """Retrieves the names of available RRDs for an snmp server.

    Args:
        rrd_path (str): The path to the rrd directoy.
        server (str): The name of the server that the data should be retrieved from.

    Returns:
        np.array: An array containing the available RRDs for the server.
    """
    parsed_rrd = pd.read_csv(rrd_path + '/parsed/snmp/parsed_data.csv')
    return parsed_rrd[parsed_rrd['server'] == server]['rrd'].unique()

def get_available_collectd_server_names(
    rrd_path:str,
    parsed_rrd:pd.DataFrame=None
    )->np.array:
    """Retrieves the names of available collectd servers.

    Args:
        rrd_path (str): The path to the rrd directoy.
        server (str): The name of the collectd server that the data should be retrieved from.

    Returns:
        np.array: An array containing the available components for the collectd server.
    """
    if parsed_rrd is None:
        parsed_rrd = pd.read_csv(rrd_path + '/parsed/collectd/parsed_data.csv')
    available_servers = list(parsed_rrd['server'].unique())
    return available_servers

def get_available_component_names_for_collectd_server(
    rrd_path:str,
    collectd_server:str,
    parsed_rrd:pd.DataFrame=None
    )->np.array:
    """Retrieves the names of available components for a collectd server.

    Args:
        rrd_path (str): The path to the rrd directoy.
        server (str): The name of the collectd server that the data should be retrieved from.

    Returns:
        np.array: An array containing the available components for the collectd server.
    """
    if parsed_rrd is None:
        parsed_rrd = pd.read_csv(rrd_path + '/parsed/collectd/parsed_data.csv')
    available_components_for_server = list(parsed_rrd[parsed_rrd['server'] == collectd_server]['component'].unique())
    return available_components_for_server

def get_available_rrd_names_for_collectd_server_component(
    rrd_path:str,
    collectd_server:str,
    component:str,
    parsed_rrd:pd.DataFrame=None
    )->np.array:
    """Retrieves the names of available RRDs for a collectd server's component.

    Args:
        rrd_path (str): The path to the rrd directoy.
        server (str): The name of the server that the data should be retrieved from.
        component (str): The name of the server's component that the data should be retrieved from.

    Returns:
        np.array: An array containing the available RRDs for the collectd server's component.
    """
    if parsed_rrd is None:
        parsed_rrd = pd.read_csv(rrd_path + '/parsed/snmp/parsed_data.csv')
    available_rrds_for_server_component = list(parsed_rrd[(parsed_rrd['server'] == collectd_server) & (parsed_rrd['component'] == component)]['rrd'].unique())
    return available_rrds_for_server_component

def get_time_series_data_for_snmp_server(
    rrd_path:str,
    snmp_server:str
    )->pd.DataFrame:
    """Builds a dataframe of time series data where
       indices are timestamps and columns are
       the data sources contained in the  snmp server's RRDs.

    Args:
        rrd_path (str): The path to the rrd directoy.
        snmp_server (str): The name of the snmp server that the data should be retrieved from.

    Returns:
        pd.DataFrame: A dataframe of time series data where
                      indices are timestamps and columns are
                      the data sources contained in the RRD.
    """
    parsed_rrd = pd.read_csv(rrd_path + '/parsed/snmp/parsed_data.csv')
    pivoted_df = parsed_rrd[parsed_rrd['server'] == snmp_server].pivot(index='time', columns=['rrd', 'data_source'], values='value')
    return pivoted_df

def get_time_series_data_for_collectd_server(
    rrd_path:str,
    collectd_server:str
    )->pd.DataFrame:
    """Builds a dataframe of time series data where
       indices are timestamps and columns are
       the data sources contained in the collectd server's RRDs.

    Args:
        rrd_path (str): The path to the rrd directoy.
        snmp_server (str): The name of the collectd server that the data should be retrieved from.

    Returns:
        pd.DataFrame: A dataframe of time series data where
                      indices are timestamps and columns are
                      the data sources contained in the collectd server's RRDs.
    """
    parsed_rrd = pd.read_csv(rrd_path + '/parsed/collectd/parsed_data.csv')
    pivoted_df = parsed_rrd[parsed_rrd['server'] == collectd_server].pivot(index='time', columns=['component', 'rrd', 'data_source'], values='value')
    return pivoted_df

def get_number_of_features_for_each_snmp_server(
    rrd_path:str,
    parsed_rrd
    )->pd.DataFrame:
    """Retrieves the number of features per server.

    Args:
        rrd_path (str): The path to the rrd directoy.
        monitor (str): The type of monitoring system that the data
                       was gathered by. Options are snmp or collectd.

    Returns:
        pd.DataFrame: A dataframe mapping servers to their number of features.
    """
    parsed_rrd = pd.read_csv(rrd_path + '/parsed/snmp/parsed_data.csv')

    n_features_per_server = {}
    for server in parsed_rrd['server'].unique():
        n_features = parsed_rrd[parsed_rrd['server'] == server].drop_duplicates(subset=['rrd', 'data_source']).shape[0]
        n_features_per_server[server] = n_features

    n_features_per_server_df = pd.DataFrame.from_dict(n_features_per_server, orient='index', columns=['n_features'])
    n_features_per_server_df.to_csv(rrd_path + '/parsed/snmp/n_features_per_server.csv')
    return n_features_per_server_df

def get_number_of_features_for_each_collectd_server(
    rrd_path:str,
    parsed_rrd:pd.DataFrame=None
    )->pd.DataFrame:
    """Retrieves the number of features per server.

    Args:
        rrd_path (str): The path to the rrd directoy.
        monitor (str): The type of monitoring system that the data
                       was gathered by. Options are snmp or collectd.

    Returns:
        pd.DataFrame: A dataframe mapping servers to their number of features.
    """
    if parsed_rrd is None:
        parsed_rrd = pd.read_csv(rrd_path + '/parsed/collectd/parsed_data.csv')

    n_features_per_server = {}
    for server in parsed_rrd['server'].unique():
        n_features = parsed_rrd[parsed_rrd['server'] == server].drop_duplicates(subset=['component', 'rrd', 'data_source']).shape[0]
        n_features_per_server[server] = n_features

    n_features_per_server_df = pd.DataFrame.from_dict(n_features_per_server, orient='index', columns=['n_features'])
    n_features_per_server_df.to_csv(rrd_path + '/parsed/collectd/n_features_per_server.csv')
    return n_features_per_server_df

def select_rrd_data_by_snmp_server(
    rrd_path:str,
    servers:list[str],
    lazy_load:bool=False,
    chunk_size:int=10000,
    parsed_rrd:pd.DataFrame=None
    )->pd.DataFrame:
    """Filters the parsed data by server so that the resulting
       dataframe contains data from a single server's RRDs.

    Args:
        rrd_path (str): The path to the rrd directoy.
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
    if parsed_rrd is None:
        if lazy_load:
            dataframes = []
            # Define a generator to lazily read the CSV file in chunks
            reader = pd.read_csv(rrd_path + '/parsed/snmp_parsed_data.csv', chunksize=chunk_size)

            # Iterate over the chunks and filter rows by the server column
            for chunk in reader:
                filtered_chunk = chunk[chunk['server'].isin(servers)]
                if not filtered_chunk.empty:
                    dataframes.append(filtered_chunk)

            # Build the full dataframe
            return pd.concat(dataframes)

        parsed_rrd = pd.read_csv(rrd_path + '/parsed/snmp_parsed_data.csv')
    return parsed_rrd[parsed_rrd['server'].isin(servers)]

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