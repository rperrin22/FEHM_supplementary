import os
import pandas as pd

def setup_test_folders(param_file):
    D = pd.read_csv(param_file)
    for x in range(len(D)):
        cmd = 'mkdir %s' % D.prefix[x]
        os.system(cmd)
        cmd = 'cp FEHM_v3.3.0linUb.04Dec15.exe %s/FEHM_lin.exe' % D.prefix[x]
        os.system(cmd)

        pather = os.getcwd()

        filename = '%s/driver.py' % D.prefix[x]
        RZ = open(filename,'w+')
        RZ.write('from FEHM_supplementary.create_setpoints_class import *\n')
        RZ.write('\n')
        RZ.write('# initialize the object\n')
        RZ.write("D = create_FEHM_run(%d,'%s/%s')\n" % (int(D.test_num[x]),pather,param_file))
        RZ.write('\n')
        RZ.write('# build surfaces\n')
        RZ.write('D.build_surfaces_real()\n')
        RZ.write('# save the csv file with coordinates\n')
        RZ.write('D.save_coords_csv()\n')
        RZ.write('# read the boundary file\n')
        RZ.write('D.read_boundary_file()\n')
        RZ.write('# make zone files\n')
        RZ.write('D.build_zones()\n')
        RZ.write('D.build_mat_prop_files()\n')
        RZ.write('D.run_lagrit()\n')
        RZ.write('D.write_input_file()\n')
        RZ.write('D.write_control_file()')
        RZ.close()

def build_run_script(param_file):
    D = pd.read_csv(param_file)

    RZ = open('FEHM_test_run.sh','w+')

    RZ.write('#!/bin/bash\n')
    RZ.write('\n')

    for x in range(len(D)):
        RZ.write('cd %s\n' % D.prefix[x])
        RZ.write('python3 driver.py\n')
        RZ.write('chmod 755 FEHM_lin.exe\n')
        RZ.write('./FEHM_lin.exe\n')
        RZ.write('cd ..\n')
        RZ.write('\n')

    RZ.close()