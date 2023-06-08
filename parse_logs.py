import datetime as dt
import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm


def determineAvailableData(src = Path(__file__).parent / 'data', dst: Path = 'available_data.csv'):
    """Writes to local directory a csv enumerating possible log data from which to draw from. Writes it to `dst`, ensure it is a csv path.

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

def parse_all(src_df : pd.DataFrame, excluded_types: set = set(), dst: str = 'all_data_content_extracted.csv'):
    """Return dataframe containing all messages in the form `datetime,host,type,content`

    Args:
        src_df (pd.DataFrame): containing columns `host,date,log_type`
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

            res_dict['datetime'].append(parsed_dt)
            res_dict['host'].append(host)
            res_dict['type'].append(type)
            res_dict['content'].append(content)
    result = pd.DataFrame(res_dict)
    result.to_csv(dst, index=False)
    return result

if __name__=='__main__':
    determineAvailableData()
    parse_all(
        pd.read_csv('available_data.csv', parse_dates=['date']),
        excluded_types= set(['daemon'])
    )