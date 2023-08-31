import platform
import os

class LocalConfig:
    def __init__(self, rrd_dir):
        self.rrd_dir = rrd_dir

class BradLisaLocalConfig(LocalConfig):
    def __init__(self):
        self.rrd_dir = '/home/bshook/Projects/server-log-api/rrd'

def get_local_config():
    username = os.getlogin()
    computer_name = platform.node()

    if username == 'bshook' and 'lisa.auton.cs.cmu.edu' in computer_name:
        return BradLisaLocalConfig()
    else:
        raise OSError('Your username and computer name does not have a corresponding LocalConfig class. Create one '\
                      'and add an elif statement to the get_local_config() function that matches your username and computer name.')