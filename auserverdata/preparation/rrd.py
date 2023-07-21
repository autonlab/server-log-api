"""
Lorem
"""
import os
import sys
import datetime as dt
import rrdtool
import matplotlib.pyplot as plt

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
        file_name = os.path.join(dir_path, file_path)

        if file_name.endswith("rrd"):
            info = rrdtool.info(file_name)
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

def get_data_from_snmp_server_rrd_files(
    i:int,
    step:int,
    snmp_server_path:str
    )->tuple[set,dict,list,dict,int]:
    """Extracts data from RRD files within the given
       colllectd_server_path directory.

    Args:
        i (int): _description_
        step (_type_): _description_
        snmp_server_path (str): _description_

    Returns:
        tuple[set,dict,list,dict,int]: _description_
    """
    file_list = set()
    value_dict = {}
    garbage_files = []
    date_dict = {}
    ret_step = 0

    # Iterate through snmp files
    for file_path in os.listdir(snmp_server_path):

        file_name = os.path.join(snmp_server_path, file_path)

        # Split the file_name by hyphens, and join the splits except the last one
        # For example, if file_name = perf-pollermodule-applications,
        # then comp = perf-pollermodule
        comp = '-'.join(file_path.split('-')[:-1])
        file_list.add(comp)

        # Check if the file is an rrd file
        if file_name.endswith("rrd"):
            # Get information about the RRD
            info = rrdtool.info(file_name)
            rra = f"rra[{i}]"
            end_num = info['last_update']
            num_rows = info[rra+'.rows']
            num_pdp_rows = info[rra+'.pdp_per_row']
            ret_step = 300 * int(num_pdp_rows)
            time_balance = (step * num_rows * num_pdp_rows)
            start_num = end_num - time_balance
            start_time_stamp = dt.datetime.utcfromtimestamp(start_num)
            end_time_stamp = dt.datetime.utcfromtimestamp(end_num)

            # Get averaged data between start_num and end_num
            getrrd = rrdtool.fetch(file_name, "AVERAGE", '-s', str(start_num),'-e', str(end_num))
            row = getrrd[2]

            # Store the rrd timeframe and values
            if row:
                value_dict[file_path] = row
                date_dict[file_path] = (start_time_stamp, end_time_stamp)
            else:
                garbage_files.append(file_path)

    return file_list, value_dict, garbage_files, date_dict, ret_step

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

            file_name = os.path.join(dir_name, file_path)

            # Split the file_name by hyphens, and join the splits except the last one
            # For example, if file_name = perf-pollermodule-applications,
            # then comp = perf-pollermodule
            comp = ''.join(file_path.split('.')[:-1])
            file_list.add(comp)

            # Check if the file is an RRD file
            if file_name.endswith("rrd"):
                # Get information about the RRD
                info = rrdtool.info(file_name)
                rra = f"rra[{i}]"

                end_num = info['last_update']
                num_rows = info[rra+'.rows']
                num_pdp_rows = info[rra+'.pdp_per_row']
                ret_step = 300 * int(num_pdp_rows)
                time_balance = (step * num_rows * num_pdp_rows)
                start_num = end_num - time_balance
                start_time_stamp = dt.datetime.utcfromtimestamp(start_num)
                end_time_stamp = dt.datetime.utcfromtimestamp(end_num)
                getrrd = rrdtool.fetch(file_name, "AVERAGE", '-s', str(start_num),'-e', str(end_num))
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