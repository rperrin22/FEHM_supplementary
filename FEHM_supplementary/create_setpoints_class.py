import numpy as np
import pandas as pd
from pylagrit import PyLaGriT
from matplotlib import pyplot as plt
from scipy import interpolate

class create_FEHM_run:
    def __init__(self,file_prefix,param_file):

        # read in the parameter file
        headings = ['parameter','value']
        params = pd.read_csv(param_file,names=headings)

        # initialize the geometry
        self.upper_numlayers = int(params.loc[params.parameter=='upper_numlayers', 'value'].values[0])
        self.middle_numlayers = int(params.loc[params.parameter=='middle_numlayers', 'value'].values[0])
        self.lower_numlayers = int(params.loc[params.parameter=='lower_numlayers', 'value'].values[0])
        self.min_x = float(params.loc[params.parameter=='min_x', 'value'].values[0])
        self.min_y = float(params.loc[params.parameter=='min_y', 'value'].values[0])
        self.min_z = float(params.loc[params.parameter=='min_z', 'value'].values[0])
        self.max_x = float(params.loc[params.parameter=='max_x', 'value'].values[0])
        self.max_y = float(params.loc[params.parameter=='max_y', 'value'].values[0])
        self.max_z = float(params.loc[params.parameter=='max_z', 'value'].values[0])
        self.dx = float(params.loc[params.parameter=='dx', 'value'].values[0])
        self.dy = float(params.loc[params.parameter=='dy', 'value'].values[0])
        self.dz = (self.max_z - self.min_z + 1)/(self.upper_numlayers + self.middle_numlayers + self.lower_numlayers)
        self.xvec = np.arange(self.min_x,self.max_x,self.dx)
        self.yvec = np.arange(self.min_y,self.max_y,self.dy)
        self.zvec = np.arange(self.min_z,self.max_z,self.dz)
        self.XX,self.YY,self.ZZ = np.meshgrid(self.xvec,self.yvec,self.zvec)

        # create output filenames
        self.csv_filename = '%s_grid_coords.csv' % file_prefix
        self.inp_filename = '%s_grid_coords.inp' % file_prefix
        self.fehm_filename = '%s_grid_coords' % file_prefix
        self.material_zones_filename = '%s_materials.zone' % file_prefix
        self.boundary_zones_filename = '%s_boundary.zone' % file_prefix
        self.input_filename = '%s_input.dat' % file_prefix
        self.control_filename = 'fehmn.files'
        self.prefix_name = '%s' % file_prefix
        self.lagrit_exec_filename = params.loc[params.parameter=='lagrit_exec', 'value'].values[0]

        # populate model parameters
        self.title = '%s_input.dat' % file_prefix
        self.grad_cond1 = float(params.loc[params.parameter=='grad_cond1', 'value'].values[0])
        self.grad_cond2 = float(params.loc[params.parameter=='grad_cond2', 'value'].values[0])
        self.grad_cond3 = float(params.loc[params.parameter=='grad_cond3', 'value'].values[0])
        self.grad_ref_loc = float(params.loc[params.parameter=='grad_ref_loc', 'value'].values[0])
        self.grad_direction = float(params.loc[params.parameter=='grad_direction', 'value'].values[0])
        self.grad_ref_sat = float(params.loc[params.parameter=='grad_ref_sat', 'value'].values[0])
        self.grad_sat_slope = float(params.loc[params.parameter=='grad_sat_slope', 'value'].values[0])
        self.grad_ref_temp = float(params.loc[params.parameter=='grad_ref_temp', 'value'].values[0])
        self.grad_temp_slope = float(params.loc[params.parameter=='grad_temp_slope', 'value'].values[0])
        self.grad_ref_pres = float(params.loc[params.parameter=='grad_ref_pres', 'value'].values[0])
        self.grad_pres_slope = float(params.loc[params.parameter=='grad_pres_slope', 'value'].values[0])
        self.perm_lower = float(params.loc[params.parameter=='perm_lower', 'value'].values[0])
        self.perm_middle_ocean = float(params.loc[params.parameter=='perm_middle_ocean', 'value'].values[0])
        self.perm_middle_continental = float(params.loc[params.parameter=='perm_middle_continental', 'value'].values[0])
        self.perm_upper = float(params.loc[params.parameter=='perm_upper', 'value'].values[0])
        self.temp_lower = float(params.loc[params.parameter=='temp_lower', 'value'].values[0])
        self.temp_upper = float(params.loc[params.parameter=='temp_upper', 'value'].values[0])
        self.mult_lower = float(params.loc[params.parameter=='mult_lower', 'value'].values[0])
        self.mult_upper = float(params.loc[params.parameter=='mult_upper', 'value'].values[0])
        self.cond_lower = float(params.loc[params.parameter=='cond_lower', 'value'].values[0])
        self.cond_middle_ocean = float(params.loc[params.parameter=='cond_middle_ocean', 'value'].values[0])
        self.cond_middle_continental = float(params.loc[params.parameter=='cond_middle_continental', 'value'].values[0])
        self.cond_upper = float(params.loc[params.parameter=='cond_upper', 'value'].values[0])
        self.rock_density = float(params.loc[params.parameter=='rock_density', 'value'].values[0])
        self.rock_spec_heat = float(params.loc[params.parameter=='rock_spec_heat', 'value'].values[0])
        self.rock_porosity = float(params.loc[params.parameter=='rock_porosity', 'value'].values[0])
        self.init_time_step = float(params.loc[params.parameter=='init_time_step', 'value'].values[0])
        self.final_sim_time = float(params.loc[params.parameter=='final_sim_time', 'value'].values[0])
        self.max_time_steps = float(params.loc[params.parameter=='max_time_steps', 'value'].values[0])
        self.info_print_int = float(params.loc[params.parameter=='info_print_int', 'value'].values[0])
        self.crust_thickness = float(params.loc[params.parameter=='crust_thickness', 'value'].values[0])

        self.max_iterations = float(params.loc[params.parameter=='max_iterations', 'value'].values[0])
        self.newton_tol = float(params.loc[params.parameter=='newton_tol', 'value'].values[0])
        self.num_orth = float(params.loc[params.parameter=='num_orth', 'value'].values[0])
        self.max_solve = float(params.loc[params.parameter=='max_solve', 'value'].values[0])
        self.acc_method = params.loc[params.parameter=='acc_method', 'value'].values[0]
        self.ja = float(params.loc[params.parameter=='ja', 'value'].values[0])
        self.jb = float(params.loc[params.parameter=='jb', 'value'].values[0])
        self.jc = float(params.loc[params.parameter=='jc', 'value'].values[0])
        self.nards = float(params.loc[params.parameter=='nards', 'value'].values[0])
        self.implicitness_factor = float(params.loc[params.parameter=='implicitness_factor', 'value'].values[0])
        self.grav_direction = float(params.loc[params.parameter=='grav_direction', 'value'].values[0])
        self.upstream_weight = float(params.loc[params.parameter=='upstream_weight', 'value'].values[0])
        self.max_iter_mult = float(params.loc[params.parameter=='max_iter_mult', 'value'].values[0])
        self.time_step_mult = float(params.loc[params.parameter=='time_step_mult', 'value'].values[0])
        self.max_time_step_size = float(params.loc[params.parameter=='max_time_step_size', 'value'].values[0])
        self.min_time_step_size = float(params.loc[params.parameter=='min_time_step_size', 'value'].values[0])
        self.geom_id = float(params.loc[params.parameter=='geom_id', 'value'].values[0])
        self.lda = float(params.loc[params.parameter=='lda', 'value'].values[0])

        self.G1 = float(params.loc[params.parameter=='G1', 'value'].values[0])
        self.G2 = float(params.loc[params.parameter=='G2', 'value'].values[0])
        self.G3 = float(params.loc[params.parameter=='G3', 'value'].values[0])
        self.TMCH = float(params.loc[params.parameter=='TMCH', 'value'].values[0])
        self.OVERF = float(params.loc[params.parameter=='OVERF', 'value'].values[0])
        self.IRDOF = float(params.loc[params.parameter=='IRDOF', 'value'].values[0])
        self.ISLORD = float(params.loc[params.parameter=='ISLORD', 'value'].values[0])
        self.IBACK = float(params.loc[params.parameter=='IBACK', 'value'].values[0])
        self.ICOUPL = float(params.loc[params.parameter=='ICOUPL', 'value'].values[0])
        self.RNMAX = float(params.loc[params.parameter=='RNMAX', 'value'].values[0])

        self.zbulk = float(params.loc[params.parameter=='zbulk', 'value'].values[0])
    
    def build_surfaces_real(self,surf_filename):
        # Brings in an externally-generated surface for the top of the crust.
        # The surface is space-delimited ascii file with 3 columns - X,Y,Z

        # in the future, add an option here to generate within the script
        # based on some function like a sine-wave
        self.XXtemp,self.YYtemp = np.meshgrid(self.xvec,self.yvec)

        header_list = ["X","Y","Z"]
        self.D = pd.read_csv(surf_filename,sep=' ',names=header_list)

        self.surf_upper = self.D['Z'].to_numpy() + self.zbulk
        self.surf_upper = np.reshape(self.surf_upper,self.XXtemp.shape)
        self.surf_lower = self.surf_upper - self.crust_thickness
        self.bound_upper = np.ones(self.XXtemp.shape)*self.max_z
        self.bound_lower = np.ones(self.XXtemp.shape)*self.min_z

        self.bottom_zone = np.linspace(self.bound_lower,self.surf_lower - self.dz/4,self.lower_numlayers)
        self.middle_zone = np.linspace(self.surf_lower + self.dz/4,self.surf_upper - self.dz/4,self.middle_numlayers)
        self.upper_zone = np.linspace(self.surf_upper + self.dz/4,self.bound_upper,self.upper_numlayers)

        self.bottom_zone = np.transpose(self.bottom_zone,(1,2,0))
        self.middle_zone = np.transpose(self.middle_zone,(1,2,0))
        self.upper_zone = np.transpose(self.upper_zone,(1,2,0))

        self.ZZ = np.concatenate((self.bottom_zone,self.middle_zone,self.upper_zone),axis=2)



    def plot_upper_surface(self):
        plt.imshow(self.surf_upper)
        plt.colorbar()
        plt.show()

    def plot_lower_surface(self):
        plt.imshow(self.surf_lower)
        plt.colorbar()
        plt.show()


    def save_coords_csv(self):
        # export a csv with the XYZ's for the mesh nodes.  This will be used later
        # for running LaGriT but can also be used to externally plot the mesh nodes
        # for QC purposes.
        self.XX_out = self.XX.flatten(order = 'F')
        self.YY_out = self.YY.flatten(order = 'F')
        self.ZZ_out = self.ZZ.flatten(order = 'F')

        self.DF = pd.DataFrame({'x':self.XX_out, 'y':self.YY_out, 'z':self.ZZ_out})
        self.DF.to_csv(self.csv_filename,index=False,header=False)

    def read_boundary_file(self,boundary_filename):
        # load in the boundary file describing the location of the border between
        # the oceanic and continental crust
        colnames = ['Easting','Northing','Elevation']
        self.FF = pd.read_csv(boundary_filename,skiprows=1,names=colnames,header=None)
        self.FF.Elevation = self.FF.Elevation + self.zbulk
    
    def build_zones(self):
        self.fsurf = interpolate.interp2d(self.xvec,self.yvec,self.surf_upper,kind='linear')
        self.DF['upp_surf'] = 0
        self.DF['low_surf'] = 0

        for index,row in self.DF.iterrows():
            self.DF.upp_surf[index] = self.fsurf(self.DF.x[index],self.DF.y[index])

        self.DF.low_surf = self.DF.upp_surf - self.crust_thickness
        self.x_boun_vec = np.zeros(self.xvec.shape)
        self.f = interpolate.interp1d(self.FF.Northing,self.FF.Easting, fill_value=(self.FF.Easting.iloc[-1]+10,self.FF.Easting.iloc[0]-10), bounds_error=False)

        # first make an x-boundary column
        self.DF['x_boun'] = self.f(self.DF['y'])

        # initialize a zone column to zeros
        #   zones will be as follows:
        #      1 - below the crust
        #      2 - oceanic crust
        #      3 - continental crust
        #      4 - sediments
        self.DF['mat_zone'] = self.DF['x_boun']*0

        self.DF.mat_zone[self.DF.x[:] > self.DF.x_boun[:]] = 3  # setting continental crust
        self.DF.mat_zone[self.DF.x[:] <= self.DF.x_boun[:]] = 2  # setting oceanic crust
        self.DF.mat_zone[self.DF.z[:] < self.DF.low_surf[:]] = 1  # setting below the crust
        self.DF.mat_zone[self.DF.z[:] > self.DF.upp_surf[:]] = 4  # setting the sediment zone

        self.DF['orig_ind'] = self.DF.index*1


        # create materials zone
        testerbob = self.DF[self.DF.mat_zone==1]
        self.node_nums_bottom = testerbob[['orig_ind']].to_numpy()+1
        testerbob = self.DF[self.DF.mat_zone==2]
        self.node_nums_middle_ocean = testerbob[['orig_ind']].to_numpy()+1
        testerbob = self.DF[self.DF.mat_zone==3]
        self.node_nums_middle_continental = testerbob[['orig_ind']].to_numpy()+1
        testerbob = self.DF[self.DF.mat_zone==4]
        self.node_nums_upper = testerbob[['orig_ind']].to_numpy()+1
        
        MZ = open(self.material_zones_filename,'w+')

        zonecounter = 1
        MZ.write('zone\n')
        MZ.write('%05d\n' % zonecounter)
        zonecounter = zonecounter + 1
        MZ.write('nnum\n')
        MZ.write('     %d\n' % self.node_nums_bottom.size)
        col_ind = 1
        for x in range(self.node_nums_bottom.size):
            if x == self.node_nums_bottom.size-1:
                MZ.write(' %d\n' % self.node_nums_bottom[x])
            elif col_ind % 10 != 0:
                MZ.write(' %d' % self.node_nums_bottom[x])
            else:
                MZ.write(' %d\n' % self.node_nums_bottom[x])
            col_ind = col_ind + 1
        MZ.write('%05d\n' % zonecounter)
        zonecounter = zonecounter + 1
        MZ.write('nnum\n')
        MZ.write('     %d\n' % self.node_nums_middle_ocean.size)
        col_ind = 1
        for x in range(self.node_nums_middle_ocean.size):
            if x == self.node_nums_middle_ocean.size-1:
                MZ.write(' %d\n' % self.node_nums_middle_ocean[x])
            elif col_ind % 10 != 0:
                MZ.write(' %d' % self.node_nums_middle_ocean[x])
            else:
                MZ.write(' %d\n' % self.node_nums_middle_ocean[x])
            col_ind = col_ind + 1
        MZ.write('%05d\n' % zonecounter)
        zonecounter = zonecounter + 1
        MZ.write('nnum\n')
        MZ.write('     %d\n' % self.node_nums_middle_continental.size)
        col_ind = 1
        for x in range(self.node_nums_middle_continental.size):
            if x == self.node_nums_middle_continental.size-1:
                MZ.write(' %d\n' % self.node_nums_middle_continental[x])
            elif col_ind % 10 != 0:
                MZ.write(' %d' % self.node_nums_middle_continental[x])
            else:
                MZ.write(' %d\n' % self.node_nums_middle_continental[x])
            col_ind = col_ind + 1
        MZ.write('%05d\n' % zonecounter)
        zonecounter = zonecounter + 1
        MZ.write('nnum\n')
        MZ.write('     %d\n' % self.node_nums_upper.size)
        col_ind = 1
        for x in range(self.node_nums_upper.size):
            if x == self.node_nums_upper.size-1:
                MZ.write(' %d\n' % self.node_nums_upper[x])
            elif col_ind % 10 != 0:
                MZ.write(' %d' % self.node_nums_upper[x])
            else:
                MZ.write(' %d\n' % self.node_nums_upper[x])
            col_ind = col_ind + 1
        MZ.write('\n')
        MZ.write('stop')

        MZ.close()

        # create boundary zones
        # boundary zones will be:
        #   00005 - bottom boundary
        #   00006 - top boundary

        # in the future add one here to create a vertical internal boundary
        # that will be used to generate heat along the fault.

        self.node_nums_domain_bottom = np.where(self.ZZ_out == min(self.ZZ_out))
        self.node_nums_domain_bottom = np.asarray(self.node_nums_domain_bottom).flatten()+1

        self.node_nums_domain_top = np.where(self.ZZ_out == max(self.ZZ_out))
        self.node_nums_domain_top = np.asarray(self.node_nums_domain_top).flatten()+1

        BZ = open(self.boundary_zones_filename,'w+')

        BZ.write('zone\n')
        BZ.write('%05d\n' % zonecounter)
        zonecounter = zonecounter + 1
        BZ.write('nnum\n')
        BZ.write('     %d\n' % self.node_nums_domain_bottom.size)
        col_ind = 1
        for x in range(self.node_nums_domain_bottom.size):
            if x == self.node_nums_domain_bottom.size-1:
                BZ.write(' %d\n' % self.node_nums_domain_bottom[x])
            elif col_ind % 10 != 0:
                BZ.write(' %d' % self.node_nums_domain_bottom[x])
            else:
                BZ.write(' %d\n' % self.node_nums_domain_bottom[x])
            col_ind = col_ind + 1
        BZ.write('%05d\n' % zonecounter)
        zonecounter = zonecounter + 1
        BZ.write('nnum\n')
        BZ.write('     %d\n' % self.node_nums_domain_top.size)
        col_ind = 1
        for x in range(self.node_nums_domain_top.size):
            if x == self.node_nums_domain_top.size-1:
                BZ.write(' %d\n' % self.node_nums_domain_top[x])
            elif col_ind % 10 != 0:
                BZ.write(' %d' % self.node_nums_domain_top[x])
            else:
                BZ.write(' %d\n' % self.node_nums_domain_top[x])
            col_ind = col_ind + 1
        BZ.write('\n')
        BZ.write('stop')
        BZ.close()

    def run_lagrit(self):
        lg = PyLaGriT(lagrit_exe=self.lagrit_exec_filename)
        d = np.genfromtxt(self.csv_filename, delimiter=",")
        surf_pts = lg.points(d,elem_type='tet',connect=True)
        surf_pts.dump(self.inp_filename)
        surf_pts.dump_fehm(self.fehm_filename)

    def write_input_file(self):
        RZ = open(self.input_filename,'w+')

        RZ.write('# %s\n' % self.title)
        RZ.write('# ----------------------------SOLUTION TYPE---------------------------\n')
        RZ.write('sol\n')
        RZ.write('1  -1\n')
        RZ.write('# -----------------------CONTOUR OUTPUT REQUESTS----------------------\n')
        RZ.write('cont\n')
        RZ.write('surf   1   1e+30   time\n')
        RZ.write('xyz\n')
        RZ.write('temperature\n')
        RZ.write('pressure\n')
        RZ.write('liquid\n')
        RZ.write('saturation\n')
        RZ.write('permeability\n')
        RZ.write('velocity\n')
        RZ.write('end\n')
        RZ.write('# ----------------------INITIAL VARIABLE GRADIENTS--------------------\n')
        RZ.write('grad\n')
        RZ.write('3\n')
        RZ.write('all %1.1f %d %d %.2f %.5f\n' % (self.grad_ref_loc,self.grad_direction,self.grad_cond1,self.grad_ref_sat,self.grad_sat_slope))
        RZ.write('all %1.1f %d %d %.2f %.5f\n' % (self.grad_ref_loc,self.grad_direction,self.grad_cond2,self.grad_ref_temp,self.grad_temp_slope))
        RZ.write('all %1.1f %d %d %.2f %.5f\n' % (self.grad_ref_loc,self.grad_direction,self.grad_cond3,self.grad_ref_pres,self.grad_pres_slope))
        RZ.write('# -----------------------------PERMEABILITY---------------------------\n')
        RZ.write('zone\n')
        RZ.write('file\n')
        RZ.write('%s\n' % self.material_zones_filename)
        RZ.write('perm\n')
        RZ.write('-00001  0  0  %.2e  %.2e  %.2e\n' % (self.perm_lower,self.perm_lower,self.perm_lower))
        RZ.write('-00002  0  0  %.2e  %.2e  %.2e\n' % (self.perm_middle_ocean,self.perm_middle_ocean,self.perm_middle_ocean))
        RZ.write('-00003  0  0  %.2e  %.2e  %.2e\n' % (self.perm_middle_continental,self.perm_middle_continental,self.perm_middle_continental))
        RZ.write('-00004  0  0  %.2e  %.2e  %.2e\n' % (self.perm_upper,self.perm_upper,self.perm_upper))
        RZ.write('\n')
        RZ.write('# -----------------------------ROCK CONDUCTIVITY----------------------\n')
        RZ.write('cond\n')
        RZ.write('-00001  0  0  %.2e  %.2e  %.2e\n' % (self.cond_lower,self.cond_lower,self.cond_lower))
        RZ.write('-00002  0  0  %.2e  %.2e  %.2e\n' % (self.cond_middle_ocean,self.cond_middle_ocean,self.cond_middle_ocean))
        RZ.write('-00003  0  0  %.2e  %.2e  %.2e\n' % (self.cond_middle_continental,self.cond_middle_continental,self.cond_middle_continental))
        RZ.write('-00004  0  0  %.2e  %.2e  %.2e\n' % (self.cond_upper,self.cond_upper,self.cond_upper))
        RZ.write('\n')
        RZ.write('# ----------------------PRODUCTION------------------------------------\n')
        RZ.write('zone\n')
        RZ.write('file\n')
        RZ.write('%s\n' % self.boundary_zones_filename)
        RZ.write('hflx\n')
        RZ.write('-00005  0  0  %.2f  %.2f\n' % (self.temp_lower,self.mult_lower))
        RZ.write('-00006  0  0  %.2f  %.2f\n' % (self.temp_upper,self.mult_upper))
        RZ.write('\n')
        RZ.write('# ----------------------MAT PROPERTIES--------------------------------\n')
        RZ.write('rock\n')
        RZ.write('1  0  0  %d  %d  %.1f\n' % (self.rock_density,self.rock_spec_heat,self.rock_porosity))
        RZ.write('\n')
        RZ.write('# -----------------------TIME STEPPING PARAMETERS---------------------\n')
        RZ.write('time\n')
        RZ.write('%.1f  %.1f  %d  %d\n' % (self.init_time_step,self.final_sim_time,self.max_time_steps,self.info_print_int))
        RZ.write('\n')
        RZ.write('# --------------------SIM CONTROL PARAMETERS--------------------------\n')
        RZ.write('ctrl\n')
        RZ.write('%d  %.2e  %d  %d  %s\n' % (self.max_iterations,self.newton_tol,self.num_orth,self.max_solve,self.acc_method))
        RZ.write('%d  %d  %d  %d\n' % (self.ja,self.jb,self.jc,self.nards))
        RZ.write('\n')
        RZ.write('%d  %d  %.1f\n' % (self.implicitness_factor,self.grav_direction,self.upstream_weight))
        RZ.write('%d  %.1f  %.1e  %.1f\n' % (self.max_iter_mult,self.time_step_mult,self.min_time_step_size,self.max_time_step_size))
        RZ.write('%d  %d\n' % (self.geom_id,self.lda))
        RZ.write('# -------------------------SOLVER PARAMETERS--------------------------\n')
        RZ.write('iter\n')
        RZ.write('%.1e  %.1e  %.3f  %.1e  %.1f\n' % (self.G1,self.G2,self.G3,self.TMCH,self.OVERF))
        RZ.write('%d  %d  %d  %d  %d\n' % (self.IRDOF,self.ISLORD,self.IBACK,self.ICOUPL,self.RNMAX))
        RZ.write('stop\n')

        RZ.close()

    def write_control_file(self):
        CF = open(self.control_filename,'w+')

        CF.write('input: %s\n' % (self.input_filename))
        CF.write('grida: %s.fehmn\n' % (self.fehm_filename))
        CF.write('rsto: %s.rsto\n' % (self.prefix_name))
        CF.write('outp: %s.outp\n' % (self.prefix_name))
        CF.write('root: %s\n' % (self.prefix_name))
        CF.write('\n')
        CF.write('all\n')

        CF.close()
