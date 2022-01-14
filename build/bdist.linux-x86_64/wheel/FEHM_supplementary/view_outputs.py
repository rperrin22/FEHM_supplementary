import plotly.express as px
import pandas as pd
import numpy as np
import os
import matplotlib.tri as tri
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def view_output_nodes_single(file,param_num):
    """
    This script reads in a single output file from FEHM and plots the
    node locations colored by the chosen parameter.  Parameter nodes are:
        1 - Temperature
        2 - Pressure
        3 - Saturation
        4 - x-permeability
        5 - y-permeability
        6 - z-permeability
    """

    D = pd.read_csv(file)
    if param_num == 1:
        fig = px.scatter_3d(D,x=' X coordinate (m)',y = ' Y coordinate (m)', z = ' Z coordinate (m)',color=' Temperature (deg C)')
        fig.show()
    elif param_num == 2:
        fig = px.scatter_3d(D,x=' X coordinate (m)',y = ' Y coordinate (m)', z = ' Z coordinate (m)',color=' Liquid Pressure (MPa)')
        fig.show()
    elif param_num == 3:
        fig = px.scatter_3d(D,x=' X coordinate (m)',y = ' Y coordinate (m)', z = ' Z coordinate (m)',color=' Saturation')
        fig.show()
    elif param_num == 4:
        fig = px.scatter_3d(D,x=' X coordinate (m)',y = ' Y coordinate (m)', z = ' Z coordinate (m)',color=' X Permeability (log m**2)')
        fig.show()
    elif param_num == 5:
        fig = px.scatter_3d(D,x=' X coordinate (m)',y = ' Y coordinate (m)', z = ' Z coordinate (m)',color=' Y Permeability (log m**2)')
        fig.show()
    elif param_num == 6:
        fig = px.scatter_3d(D,x=' X coordinate (m)',y = ' Y coordinate (m)', z = ' Z coordinate (m)',color=' Z Permeability (log m**2)')
        fig.show()
    else:
        print('that is not an acceptable answer')

def plot_convergence(path):
    files = os.listdir(path)
    file_list = []
    for f in files:
        if f.endswith("days_sca_node.csv"):
            file_list.append((f,f[6:16],float(f[6:16])/365,0))

    DF = pd.DataFrame(file_list,columns=['filename','days','years','dT'])
    DF=DF.sort_values('years')

    filename_old = DF.filename[0]
    temp_old = pd.read_csv('%s/%s' % (path,filename_old))
    year_old = 0
    for x in range(len(DF)):
        temp_new = pd.read_csv('%s/%s' % (path,DF.loc[x,'filename']))
        year_new = DF.loc[x,'years']
        ddt = year_new - year_old
        bob = (temp_new[' Temperature (deg C)'] - temp_old[' Temperature (deg C)'])**2
        DF.loc[x,'dT'] = np.sqrt(bob.sum())/ddt
        temp_old = temp_new.copy()
        year_old = year_new.copy()
    
    fig = px.line(DF,x='years',y='dT')
    fig.show()

def plot_vert_profiles(param_file,xpos,ypos):
    """
        plot vertical temperature and pressure profiles for a given location
        in the survey area.  
    """
    D = pd.read_csv(param_file)

    plt.figure(figsize=(12,8))
    plt.subplot(1,2,1)
    for x in range(len(D)):
        filename = '%s/%s.36500000.0_days_sca_node.csv' % (D.prefix[x],D.prefix[x])
        D1 = pd.read_csv(filename)
        DS1 = D1[(D1[' X coordinate (m)']==xpos) & (D1[' Y coordinate (m)']==ypos)]
        plt.plot(DS1[' Temperature (deg C)'],DS1[' Z coordinate (m)'],'*:',label='%s'%D.prefix[x])

    plt.grid()
    plt.legend()
    plt.ylabel('Elevation (m)')
    plt.xlabel('Temperature (deg C)')

    D2 = pd.read_csv(D.surf_filename[0],sep=' ',names=['x','y','z'])
    triang = tri.Triangulation(D2.x, D2.y)

    D2.z = D2.z+D.zbulk[0]

    plt.subplot(2,2,2)
    levels = np.arange(min(D2.z), max(D2.z), 10)
    cmap = cm.get_cmap(name='terrain', lut=None)
    tcf = plt.tricontourf(triang, D2.z, levels=levels, cmap=cmap)
    plt.tricontour(triang, D2.z, levels=levels,
                colors=['0.25', '0.5', '0.5', '0.5', '0.5'],
                linewidths=[1.0, 0.5, 0.5, 0.5, 0.5])

    plt.scatter(x=xpos,y=ypos,c='k')
    plt.title("Crust Surface")
    plt.xlabel('Easting (m)')
    plt.ylabel('Northing (m)')
    cbar = plt.colorbar(tcf)
    cbar.set_label('Elevation (m)')

    plt.show()
