import plotly.express as px
import pandas as pd
import numpy as np

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

    