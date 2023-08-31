from pytest import approx
from auserverdata.preparation.rrd import (
    get_available_snmp_server_names,
    get_available_collectd_server_names,
    get_available_component_names_for_collectd_server,
    get_available_rrd_names_for_snmp_server,
    get_available_rrd_names_for_collectd_server_component,
    get_time_series_data_for_snmp_server,
    get_time_series_data_for_collectd_server
)

rrd_dir = '/home/bshook/Projects/server-log-api/rrd'

def test_get_available_snmp_server_names():
    available_snmp_server_names = get_available_snmp_server_names(rrd_dir)
    assert len(available_snmp_server_names) == 74, 'The number of available snmp server names was not expected.'

def test_get_available_collectd_server_names():
    available_collectd_server_names = get_available_collectd_server_names(rrd_dir)
    assert len(available_collectd_server_names) == 72, 'The number of available snmp server names was not expected.'

def test_get_available_component_names_for_collectd_server():
    available_component_names = get_available_component_names_for_collectd_server(
        rrd_dir,
        collectd_server='gpu1'
        )
    assert len(available_component_names) == 34, 'The number of available component names was not expected.'

def test_get_available_rrd_names_for_snmp_server():
    available_rrd_names = get_available_rrd_names_for_snmp_server(
                            rrd_dir,
                            'gpu1'
                            )
    assert len(available_rrd_names) == 134, 'The number of available rrd names was not expected.'

def test_get_available_rrd_names_for_collectd_server_component():
    available_rrd_names = get_available_rrd_names_for_collectd_server_component(
                            rrd_dir,
                            collectd_server='gpu1',
                            component='cpu-1'
                            )
    assert len(available_rrd_names) == 8, 'The number of available rrd names was not expected.'

def test_get_time_series_data_for_snmp_server():
    snmp_ts = get_time_series_data_for_snmp_server(rrd_dir, snmp_server='gpu1')
    assert snmp_ts['ucd_diskio-dm-0']['read'][1] == approx(6.915904139433553), 'The value in the time series was not expected.'

def test_get_time_series_data_for_collectd_server():
    collectd_ts = get_time_series_data_for_collectd_server(rrd_dir, collectd_server='gpu1')
    assert collectd_ts['cpu-0']['cpu-idle']['value'][-2] == approx(99.6), 'The value in the time series was not expected.'