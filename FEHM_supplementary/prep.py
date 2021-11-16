import os
import pandas as pd

def setup_test_folders(param_file):
    D = pd.read_csv(param_file)
    for x in D['prefix']:
        cmd = 'mkdir %s' % x
        os(cmd)
        cmd = 'cp /mnt/c/FEHM_template_files/FEHM_v3.3.0linUb.04Dec15.exe %s/FEHM_lin.exe' % x
        os(cmd)
        cmd = 'cp /mnt/c/FEHM_template_files/driver.py %s/driver.py' % x