import os
import sys
import rrdtool 
import pickle
import datetime as dt
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
from audata import File

data_path = "/home/ubuntu/exs/pmx_cpu/rrd_examples"
plot_path = "/home/ubuntu/exs/pmx_cpu/plots"
i, step = 1, 300

def check_none(rows):
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

def get_date(dirpath):
    # Function to get start date of file
    min_start_date, max_start_date = sys.maxsize, 0

    for f in os.listdir(dirpath):
        filename = os.path.join(dirpath, f)
        
        if filename.endswith("rrd"):
            info = rrdtool.info(filename)
            end_num = info['last_update']

            rra = "rra[{}]".format(i)
            num_rows = info[rra+'.rows']
            num_pdp_rows = info[rra+'.pdp_per_row']
            ret_step = 300 * int(num_pdp_rows)

            time_balance = (step * num_rows * num_pdp_rows)
            start_num = end_num - time_balance
            if start_num < min_start_date:
                min_start_date = start_num
            
            if end_num > max_start_date:
                max_start_date = end_num
            
    #print(len(garbage_files))
    return dt.datetime.utcfromtimestamp(min_start_date), dt.datetime.utcfromtimestamp(max_start_date)

def snmp_data(dirpath):
    file_list = set()
    garbage_files = []
    value_dict = {}
    date_dict = {}
    ret_step = 0

    for f in os.listdir(dirpath):
        filename = os.path.join(dirpath, f)
        comp = '-'.join(f.split('-')[:-1])
        file_list.add(comp)
        if filename.endswith("rrd"):
            info = rrdtool.info(filename)
            end_num = info['last_update']

            rra = "rra[{}]".format(i)
            num_rows = info[rra+'.rows']
            num_pdp_rows = info[rra+'.pdp_per_row']
            ret_step = 300 * int(num_pdp_rows)

            time_balance = (step * num_rows * num_pdp_rows)
            start_num = end_num - time_balance
            start_time_stamp = dt.datetime.utcfromtimestamp(start_num)
            end_time_stamp = dt.datetime.utcfromtimestamp(end_num)
            getrrd = rrdtool.fetch(filename, "AVERAGE", '-s', str(start_num),'-e', str(end_num))
            row = getrrd[2]
            
            if row:
                # Handle the correct file 
                value_dict[f] = row
                date_dict[f] = (start_time_stamp, end_time_stamp)
            else:
                garbage_files.append(f)
    #print(len(garbage_files))
    return file_list, value_dict, garbage_files, date_dict, ret_step

def collectd_data(dirpath):
    file_list = set()
    garbage_files = []
    value_dict = {}
    date_dict = {}
    ret_step = 0

    for d in os.listdir(dirpath):
        dirname = os.path.join(dirpath, d)
        for f in os.listdir(dirname):
            filename = os.path.join(dirname, f)
            comp = ''.join(f.split('.')[:-1])
            file_list.add(comp)
            if filename.endswith("rrd"):
                info = rrdtool.info(filename)
                end_num = info['last_update']
                
                rra = "rra[{}]".format(i)
                num_rows = info[rra+'.rows']
                num_pdp_rows = info[rra+'.pdp_per_row']
                ret_step = 300 * int(num_pdp_rows)

                time_balance = (step * num_rows * num_pdp_rows)
                start_num = end_num - time_balance
                start_time_stamp = dt.datetime.utcfromtimestamp(start_num)
                end_time_stamp = dt.datetime.utcfromtimestamp(end_num)
                getrrd = rrdtool.fetch(filename, "AVERAGE", '-s', str(start_num),'-e', str(end_num))
                row = getrrd[2]

                if row:
                    # Handle the correct file 
                    value_dict[os.path.join(d,f)] = row
                    date_dict[os.path.join(d,f)] = (start_time_stamp, end_time_stamp)
                else:
                    garbage_files.append(f)
    
    return file_list, value_dict, garbage_files, date_dict, ret_step

def gen_plots_snmp(value_dict, dir_name):
    dir_plot_path = os.path.join(plot_path, dir_name)

    if not os.path.exists(dir_plot_path):
        os.mkdir(dir_plot_path)

    for key, values in value_dict.items():
        title = dir_name + ": " + ''.join(key.split('.')[:-1])
        file_path = os.path.join(dir_plot_path, ''.join(key.split('.')[:-1]))

        plt.plot(values)
        plt.title(title)
        plt.savefig(file_path)


fold_dict = {"collectd":collectd_data, "snmp":{"data":snmp_data, "plot":gen_plots_snmp}}


if __name__=="__main__":
    file_dict = {}
    f_name = os.path.join(os.getcwd(), 'rrd_data.h5')
    if os.path.exists(f_name):
        os.remove(f_name)
    date_time, _ = get_date(os.path.join(data_path, "snmp"))
    
    for d in os.listdir(data_path):
        dir_path = os.path.join(data_path,d)

        # Get features from snmp
        if "snmp" in dir_path:
            for subdirs in os.listdir(dir_path):
                subdir_path = os.path.join(dir_path, subdirs)
                if os.path.isdir(subdir_path):
                    f = File.new('h5s/snmp_{}.h5'.format(subdirs), time_reference=date_time)
                    print(subdir_path)
                    file_list, value_dict, garbage_file, date_dict, ret_step = fold_dict["snmp"]["data"](subdir_path)
                    file_dict[d] = {"file-list":file_list, "value-dict":value_dict, "garbage-files":garbage_file}
                    #fold_dict["snmp"]["plot"](file_dict[d]["value-dict"], d)
                    for k in value_dict.keys():
                        pd_len = len(value_dict[k])
                        df_f = pd.DataFrame(data={
                            'time': f.time_reference + dt.timedelta(seconds=ret_step)*np.arange(pd_len),
                            'values': list(value_dict[k])
                        })
                        f['data/{0}'.format(k)] = df_f
                    f.close()
                    #print(f['data'])

        # Get features from collectd
        elif "collectd" in dir_path:
            for subdirs in os.listdir(dir_path):
                f = File.new('h5s/collectd_{}.h5'.format(subdirs), time_reference=date_time)
                subdir_path = os.path.join(dir_path, subdirs)
                print(subdir_path)
                file_list, value_dict, garbage_file, date_dict, ret_step = fold_dict["collectd"](subdir_path)
                file_dict[d] = {"file-list":file_list, "value-dict":value_dict, "garbage-files":garbage_file}
                for k in value_dict.keys():
                    pd_len = len(value_dict[k])
                    df_f = pd.DataFrame(data={
                        'time': f.time_reference + dt.timedelta(seconds=ret_step)*np.arange(pd_len),
                        'values': list(value_dict[k])
                    })
                    f['data/{0}'.format(k)] = df_f

                f.close()

    with open("rrd_dict_lov_low.p", "wb") as p:
        pickle.dump(file_dict, p, protocol=pickle.HIGHEST_PROTOCOL)

    print(file_dict.keys())