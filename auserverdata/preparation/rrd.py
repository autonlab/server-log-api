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
from joblib import Parallel,delayed
import pyarrow.parquet as pq

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
        results.to_parquet(dst, index=False)
    return results

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
        results.to_parquet(dst, index=False)
    return results

def get_available_collectd_server_names(
    rrd_dir:str
    )->list[str]:
    """Retrieves the names of available collectd servers.

    Args:
        rrd_dir (str): The path to the rrd directoy.
        server (str): The name of the collectd server that the data should be retrieved from.

    Returns:
        list[str]: A list containing the available collectd servers.
    """
    parsed_rrd = pq.read_table(rrd_dir + '/parsed/collectd/parsed_data.parquet', columns=['server']).to_pandas()
    return list(parsed_rrd['server'].unique())

def get_available_snmp_server_names(
    rrd_dir:str
    )->list[str]:
    """Retrieves the names of available snmp servers.

    Args:
        rrd_dir (str): The path to the rrd directoy.
        server (str): The name of the snmp server that the data should be retrieved from.

    Returns:
        list[str]: A list containing the available snmp servers.
    """
    parsed_rrd = pq.read_table(rrd_dir + '/parsed/snmp/parsed_data.parquet', columns=['server']).to_pandas()
    return list(parsed_rrd['server'].unique())

def get_available_component_names_for_collectd_server(
    rrd_dir:str,
    collectd_server:str
    )->list[str]:
    """Retrieves the names of available components for a collectd server.

    Args:
        rrd_dir (str): The path to the rrd directoy.
        server (str): The name of the collectd server that the data should be retrieved from.

    Returns:
        list[str]: An array containing the available components for the collectd server.
    """
    condition = ('server','=',collectd_server)
    parsed_rrd = pq.read_table(rrd_dir + '/parsed/collectd/parsed_data.parquet', filters=[condition], columns=['component']).to_pandas()
    return list(parsed_rrd['component'].unique())

def get_available_rrd_names_for_collectd_server_component(
    rrd_dir:str,
    collectd_server:str,
    component:str
    )->list[str]:
    """Retrieves the names of available RRDs for a collectd server's component.

    Args:
        rrd_dir (str): The path to the rrd directoy.
        server (str): The name of the server that the data should be retrieved from.
        component (str): The name of the server's component that the data should be retrieved from.

    Returns:
        list[str]: An array containing the available RRDs for the collectd server's component.
    """
    condition = [('server','=',collectd_server),('component','=',component)]
    parsed_rrd = pq.read_table(rrd_dir + '/parsed/collectd/parsed_data.parquet', filters=[condition], columns=['rrd']).to_pandas()
    return list(parsed_rrd['rrd'].unique())

def get_available_rrd_names_for_snmp_server(
    rrd_dir:str,
    snmp_server:str
    )->list[str]:
    """Retrieves the names of available RRDs for an snmp server.

    Args:
        rrd_dir (str): The path to the rrd directoy.
        snmp_server (str): The name of the server that the data should be retrieved from.

    Returns:
        list[str]: An array containing the available RRDs for the server.
    """
    condition = ('server','=',snmp_server)
    parsed_rrd = pq.read_table(rrd_dir + '/parsed/snmp/parsed_data.parquet', filters=[condition], columns=['rrd']).to_pandas()
    return parsed_rrd['rrd'].unique()

def get_time_series_data_for_collectd_servers(
    rrd_dir:str,
    collectd_servers:list[str]
    )->pd.DataFrame:
    """Builds a dataframe of time series data where
       indices are timestamps and columns are
       the data sources contained in the collectd server's RRDs.

    Args:
        rrd_dir (str): The path to the rrd directoy.
        snmp_server (str): The name of the collectd server that the data should be retrieved from.

    Returns:
        pd.DataFrame: A dataframe of time series data where
                      indices are timestamps and columns are
                      the data sources contained in the collectd server's RRDs.
    """
    conditions = []
    for collectd_server in collectd_servers:
        conditions.append([('server', '==', collectd_server)])
    parsed_rrd = pq.read_table(rrd_dir + '/parsed/collectd/parsed_data.parquet', filters=conditions, columns=['server', 'component','rrd','data_source','time','value']).to_pandas()
    parsed_rrd = parsed_rrd.pivot(index='time', columns=['server','component','rrd','data_source'], values='value')
    return parsed_rrd

def get_time_series_data_for_snmp_servers(
    rrd_dir:str,
    snmp_server:str
    )->pd.DataFrame:
    """Builds a dataframe of time series data where
       indices are timestamps and columns are
       the data sources contained in the  snmp server's RRDs.

    Args:
        rrd_dir (str): The path to the rrd directoy.
        snmp_server (str): The name of the snmp server that the data should be retrieved from.

    Returns:
        pd.DataFrame: A dataframe of time series data where
                      indices are timestamps and columns are
                      the data sources contained in the RRD.
    """
    conditions = []
    for snmp_server in snmp_servers:
        conditions.append([('server', '==', snmp_server)])
    parsed_rrd = pq.read_table(rrd_dir + '/parsed/snmp/parsed_data.parquet', filters=conditions, columns=['server', 'rrd', 'data_source', 'time', 'value']).to_pandas()
    parsed_rrd = parsed_rrd.pivot(index='time', columns=['server', 'rrd', 'data_source'], values='value')
    return parsed_rrd

def get_time_series_data_for_single_collectd_rrd(
    rrd_dir:str,
    collectd_server:str,
    component:str,
    rrd:str
    )->pd.DataFrame:
    """Builds a dataframe of time series data for a single
       collectd server component's rrd where indices are timestamps and columns are
       the component, rrd, and data sources.

    Args:
        rrd_dir (str): The path to the rrd directoy.
        collectd_server (str): The name of the collectd server that the data should be retrieved from.
        component (str): The name of the component that the data should be retrieved from.
        rrd (str): The name of the rrd that the data should be retrieved from.

    Returns:
        pd.DataFrame: A dataframe of time series data for a single snmpp server
                      where indices are timestamps and columns are the
                      component, rrd, and data sources.
    """
    condition = [('server','=',collectd_server),('component','=',component),('rrd','=',rrd)]
    parsed_rrd = pq.read_table(rrd_dir + '/parsed/collectd/parsed_data.parquet', filters=[condition], columns=['component','rrd','data_source','time','value']).to_pandas()
    parsed_rrd = parsed_rrd.pivot(index='time', columns=['component', 'rrd', 'data_source'], values='value')
    return parsed_rrd

def get_time_series_data_for_single_snmp_rrd(
    rrd_dir:str,
    snmp_server:str,
    rrd:str
    )->pd.DataFrame:
    """Builds a dataframe of time series data for a single
       snmp server's rrd where indices are timestamps and columns are
       the rrd and data sources.

    Args:
        rrd_dir (str): The path to the rrd directoy.
        snmp_server (str): The name of the snmp server that the data should be retrieved from.
        rrd (str): The name of the rrd that the data should be retrieved from.

    Returns:
        pd.DataFrame: A dataframe of time series data for a single snmpp server
                      where indices are timestamps and columns are the rrd
                      and data sources.
    """
    condition = [('server','=',snmp_server),('rrd','=',rrd)]
    parsed_rrd = pq.read_table(rrd_dir + '/parsed/snmp/parsed_data.parquet', filters=[condition], columns=['rrd','data_source','time','value']).to_pandas()
    parsed_rrd = parsed_rrd.pivot(index='time', columns=['rrd', 'data_source'], values='value')
    return parsed_rrd

def get_collectd_features_with_matching_timestamps(
    rrd_dir:str,
    collectd_server:str
    )->dict[int,pd.DataFrame]:
    """Builds a dictionary mapping unique counts of non-nan values to
       a dataframe containing features that share timestamps. All data is retrieved
       for a single collectd server.

    Args:
        rrd_dir (str): The path to the rrd directoy.
        collectd_server (str): The name of the collectd server that the data should be retrieved from.

    Returns:
        dict[int,pd.DataFrame]: A dictionary mapping unique counts of non-nan values to
                                a dataframe containing features that share timestamps.
    """
    # Get the time series data for the snmp server
    collectd_ts_df = get_time_series_data_for_collectd_servers(rrd_dir=rrd_dir, collectd_servers=[collectd_server])
    # Count the number of non-nan values per row/timestamp
    # We will find that the unique numbers of non-nan values per timestamp
    # is discrete. This means that some features share the timestamps.
    # For example, if there are 20 timestamps with 250 non-nan values,
    # then that means there are 250 features that share 20 timestamps.
    non_nan_value_count_per_timestamp = collectd_ts_df.count(axis=1)
    # Get the number of timestamps per unique count of non-nan values.
    # For example, if there are 20 timestamps with 250 non-nan values,
    # then we will get 250 mapped to 20.
    value_counts = non_nan_value_count_per_timestamp.value_counts()
    # Create a dictionary that maps the unique counts of non-nan values
    # to a dataframe containing the features sharing timestamps.
    # For example, if there are 20 timestamps with 250 non-nan values,
    # then we will have a key = 250 that maps to a dictionary
    # that has 20 timestamps and columns being the features that share
    # those timestamps.
    features_df_per_non_nan_value_count = {}
    for idx in value_counts.index:
        feature_df = collectd_ts_df.loc[non_nan_value_count_per_timestamp[non_nan_value_count_per_timestamp == idx].index]
        features_df_per_non_nan_value_count[idx] = feature_df.dropna(axis=1,how='all')

    return features_df_per_non_nan_value_count

def get_snmp_features_with_matching_timestamps(
    rrd_dir:str,
    snmp_server:str
    )->dict[int,pd.DataFrame]:
    """Builds a dictionary mapping unique counts of non-nan values to
       a dataframe containing features that share timestamps. All data is retrieved
       for a single snmp server.

    Args:
        rrd_dir (str): The path to the rrd directoy.
        snmp_server (str): The name of the snmp server that the data should be retrieved from.

    Returns:
        dict[int,pd.DataFrame]: A dictionary mapping unique counts of non-nan values to
                                a dataframe containing features that share timestamps.
    """
    # Get the time series data for the snmp server
    snmp_ts_df = get_time_series_data_for_snmp_servers(rrd_dir=rrd_dir, snmp_servers=[snmp_server])
    # Count the number of non-nan values per row/timestamp
    # We will find that the unique numbers of non-nan values per timestamp
    # is discrete. This means that some features share the timestamps.
    # For example, if there are 20 timestamps with 250 non-nan values,
    # then that means there are 250 features that share 20 timestamps.
    non_nan_value_count_per_timestamp = snmp_ts_df.count(axis=1)
    # Get the number of timestamps per unique count of non-nan values.
    # For example, if there are 20 timestamps with 250 non-nan values,
    # then we will get 250 mapped to 20.
    value_counts = non_nan_value_count_per_timestamp.value_counts()
    # Create a dictionary that maps the unique counts of non-nan values
    # to a dataframe containing the features sharing timestamps.
    # For example, if there are 20 timestamps with 250 non-nan values,
    # then we will have a key = 250 that maps to a dataframe
    # that has 20 timestamps and columns being the features that share
    # those timestamps.
    features_df_per_non_nan_value_count = {}
    for idx in value_counts.index:
        feature_df = snmp_ts_df.loc[non_nan_value_count_per_timestamp[non_nan_value_count_per_timestamp == idx].index]
        features_df_per_non_nan_value_count[idx] = feature_df.dropna(axis=1,how='all')

    return features_df_per_non_nan_value_count

def get_unique_component_rrd_pairs_from_features_with_matching_timestamps(features_with_matching_timestamps):
    component_rrd_df = pd.DataFrame(
        pd.Series(
            features_with_matching_timestamps.columns.droplevel(['data_source', 'server']).unique().to_numpy()
            ).apply(lambda x: pd.Series(x))
        )
    component_rrd_df.columns = ['component', 'rrd']
    return component_rrd_df

def get_sorted_value_counts_of_unique_component_rrd_pairs(component_rrd_df):
    return component_rrd_df['component'].value_counts().sort_index()

def get_number_of_features_for_each_collectd_server(
    rrd_dir:str
    )->pd.DataFrame:
    """Retrieves the number of features per server.

    Args:
        rrd_dir (str): The path to the rrd directoy.
        monitor (str): The type of monitoring system that the data
                       was gathered by. Options are snmp or collectd.

    Returns:
        pd.DataFrame: A dataframe mapping servers to their number of features.
    """
    parsed_rrd = pd.read_parquet(rrd_dir + '/parsed/collectd/parsed_data.parquet')

    n_features_per_server = {}
    for server in parsed_rrd['server'].unique():
        n_features = parsed_rrd[parsed_rrd['server'] == server].drop_duplicates(subset=['component', 'rrd', 'data_source']).shape[0]
        n_features_per_server[server] = n_features

    n_features_per_server_df = pd.DataFrame.from_dict(n_features_per_server, orient='index', columns=['n_features'])
    n_features_per_server_df.to_csv(rrd_dir + '/parsed/collectd/n_features_per_server.csv')
    return n_features_per_server_df

def get_number_of_features_for_each_snmp_server(
    rrd_dir:str
    )->pd.DataFrame:
    """Retrieves the number of features per server.

    Args:
        rrd_dir (str): The path to the rrd directoy.
        monitor (str): The type of monitoring system that the data
                       was gathered by. Options are snmp or collectd.

    Returns:
        pd.DataFrame: A dataframe mapping servers to their number of features.
    """
    parsed_rrd = pd.read_parquet(rrd_dir + '/parsed/snmp/parsed_data.parquet')

    n_features_per_server = {}
    for server in parsed_rrd['server'].unique():
        n_features = parsed_rrd[parsed_rrd['server'] == server].drop_duplicates(subset=['rrd', 'data_source']).shape[0]
        n_features_per_server[server] = n_features

    n_features_per_server_df = pd.DataFrame.from_dict(n_features_per_server, orient='index', columns=['n_features'])
    n_features_per_server_df.to_csv(rrd_dir + '/parsed/snmp/n_features_per_server.csv')
    return n_features_per_server_df