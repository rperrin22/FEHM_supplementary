import numpy as np
import pandas as pd
from pylagrit import PyLaGriT
from matplotlib import pyplot as plt
from scipy import interpolate

class create_FEHM_run:
    def __init__(self,test_number,param_file):

        # read in the parameter file
        temp = pd.read_csv(param_file)
        params = temp[temp.test_num==test_number]

        # initialize the geometry
        file_prefix = params['prefix'].values[0]
        self.upper_numlayers = int(params['upper_numlayers'].values[0])
        self.middle_numlayers = int(params['middle_numlayers'].values[0])
        self.lower_numlayers = int(params['lower_numlayers'].values[0])
        self.min_x = float(params['min_x'].values[0])
        self.min_y = float(params['min_y'].values[0])
        self.min_z = float(params['min_z'].values[0])
        self.max_x = float(params['max_x'].values[0])
        self.max_y = float(params['max_y'].values[0])
        self.max_z = float(params['max_z'].values[0])
        self.dx = float(params['dx'].values[0])
        self.dy = float(params['dy'].values[0])
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
        self.lagrit_exec_filename = params['lagrit_exec'].values[0]

        # populate model parameters
        self.title = '%s_input.dat' % file_prefix
        self.grad_cond1 = float(params['grad_cond1'].values[0])
        self.grad_cond2 = float(params['grad_cond2'].values[0])
        self.grad_cond3 = float(params['grad_cond3'].values[0])
        self.grad_ref_loc = float(params['grad_ref_loc'].values[0])
        self.grad_direction = float(params['grad_direction'].values[0])
        self.grad_ref_sat = float(params['grad_ref_sat'].values[0])
        self.grad_sat_slope = float(params['grad_sat_slope'].values[0])
        self.grad_ref_temp = float(params['grad_ref_temp'].values[0])
        self.grad_temp_slope = float(params['grad_temp_slope'].values[0])
        self.grad_ref_pres = float(params['grad_ref_pres'].values[0])
        self.grad_pres_slope = float(params['grad_pres_slope'].values[0])
        self.perm_lower = float(params['perm_lower'].values[0])
        self.perm_middle_ocean = float(params['perm_middle_ocean'].values[0])
        self.perm_middle_continental = float(params['perm_middle_continental'].values[0])
        self.perm_upper = float(params['perm_upper'].values[0])
        self.temp_lower = float(params['temp_lower'].values[0])
        self.temp_upper = float(params['temp_upper'].values[0])
        self.mult_lower = float(params['mult_lower'].values[0])
        self.mult_upper = float(params['mult_upper'].values[0])
        self.cond_lower = float(params['cond_lower'].values[0])
        self.cond_middle_ocean = float(params['cond_middle_ocean'].values[0])
        self.cond_middle_continental = float(params['cond_middle_continental'].values[0])
        self.cond_upper = float(params['cond_upper'].values[0])
        self.rock_density = float(params['rock_density'].values[0])
        self.rock_spec_heat = float(params['rock_spec_heat'].values[0])
        self.rock_porosity = float(params['rock_porosity'].values[0])
        self.solids_cond = 2
        self.water_cond = 0.604
        self.init_time_step = float(params['init_time_step'].values[0])
        self.final_sim_time = float(params['final_sim_time'].values[0])
        self.max_time_steps = float(params['max_time_steps'].values[0])
        self.info_print_int = float(params['info_print_int'].values[0])
        self.crust_thickness = float(params['crust_thickness'].values[0])
        self.rp_mult = float(params['athy_multiplier'].values[0])
        self.rp_exp = float(params['athy_exp'].values[0])

        self.max_iterations = float(params['max_iterations'].values[0])
        self.newton_tol = float(params['newton_tol'].values[0])
        self.num_orth = float(params['num_orth'].values[0])
        self.max_solve = float(params['max_solve'].values[0])
        self.acc_method = params['acc_method'].values[0]
        self.ja = float(params['ja'].values[0])
        self.jb = float(params['jb'].values[0])
        self.jc = float(params['jc'].values[0])
        self.nards = float(params['nards'].values[0])
        self.implicitness_factor = float(params['implicitness_factor'].values[0])
        self.grav_direction = float(params['grav_direction'].values[0])
        self.upstream_weight = float(params['upstream_weight'].values[0])
        self.max_iter_mult = float(params['max_iter_mult'].values[0])
        self.time_step_mult = float(params['time_step_mult'].values[0])
        self.max_time_step_size = float(params['max_time_step_size'].values[0])
        self.min_time_step_size = float(params['min_time_step_size'].values[0])
        self.geom_id = float(params['geom_id'].values[0])
        self.lda = float(params['lda'].values[0])

        self.G1 = float(params['G1'].values[0])
        self.G2 = float(params['G2'].values[0])
        self.G3 = float(params['G3'].values[0])
        self.TMCH = float(params['TMCH'].values[0])
        self.OVERF = float(params['OVERF'].values[0])
        self.IRDOF = float(params['IRDOF'].values[0])
        self.ISLORD = float(params['ISLORD'].values[0])
        self.IBACK = float(params['IBACK'].values[0])
        self.ICOUPL = float(params['ICOUPL'].values[0])
        self.RNMAX = float(params['RNMAX'].values[0])

        self.zbulk = float(params['zbulk'].values[0])

        self.surf_filename = params['surf_filename'].values[0]
        self.boundary_filename = params['boundary_filename'].values[0]
    
    def build_surfaces_real(self):
        # Brings in an externally-generated surface for the top of the crust.
        # The surface is space-delimited ascii file with 3 columns - X,Y,Z

        # in the future, add an option here to generate within the script
        # based on some function like a sine-wave
        self.XXtemp,self.YYtemp = np.meshgrid(self.xvec,self.yvec)

        header_list = ["X","Y","Z"]
        self.D = pd.read_csv(self.surf_filename,sep=' ',names=header_list)

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

    def build_mat_prop_files(self):
        self.mat_prop_filename = '%s.rock' % self.prefix_name
        self.cond_filename = '%s.cond' % self.prefix_name
        PZ = open(self.mat_prop_filename,'w+')
        CZ = open(self.cond_filename,'w+')

        PZ.write('rock\n')
        CZ.write('cond\n')

        for x in self.node_nums_upper:
            temp_depth = self.max_z - self.ZZ_out[x-1]
            temp_porosity = self.rp_mult*np.exp(self.rp_exp*temp_depth/1000)
            temp_cond = self.solids_cond**(1 - temp_porosity) * self.water_cond**(temp_porosity)
            PZ.write('  %d %d 1 %d %d %.1f\n' % (x,x,self.rock_density,self.rock_spec_heat,temp_porosity))
            CZ.write('  %d %d 1 %.2f %.2f %.2f\n' % (x,x,temp_cond,temp_cond,temp_cond))
        for x in self.node_nums_middle_ocean:
            PZ.write('  %d  %d  1  %d  %d  %.1f\n' % (x,x,self.rock_density,self.rock_spec_heat,self.rock_porosity))
            CZ.write('  %d  %d  1  %.2f  %.2f  %.2f\n' % (x,x,self.cond_middle_ocean,self.cond_middle_ocean,self.cond_middle_ocean))
        for x in self.node_nums_middle_continental:
            PZ.write('  %d  %d  1  %d  %d  %.1f\n' % (x,x,self.rock_density,self.rock_spec_heat,self.rock_porosity))
            CZ.write('  %d  %d  1  %.2f  %.2f  %.2f\n' % (x,x,self.cond_middle_continental,self.cond_middle_continental,self.cond_middle_continental))
        for x in self.node_nums_bottom:
            PZ.write('  %d  %d  1  %d  %d  %.1f\n' % (x,x,self.rock_density,self.rock_spec_heat,self.rock_porosity))
            CZ.write('  %d  %d  1  %.2f  %.2f  %.2f\n' % (x,x,self.cond_lower,self.cond_lower,self.cond_lower))
        PZ.write('\nstop')
        PZ.close()
        CZ.write('\nstop')
        CZ.close()

    def save_coords_csv(self):
        # export a csv with the XYZ's for the mesh nodes.  This will be used later
        # for running LaGriT but can also be used to externally plot the mesh nodes
        # for QC purposes.
        self.XX_out = self.XX.flatten(order = 'F')
        self.YY_out = self.YY.flatten(order = 'F')
        self.ZZ_out = self.ZZ.flatten(order = 'F')

        self.DF = pd.DataFrame({'x':self.XX_out, 'y':self.YY_out, 'z':self.ZZ_out})
        self.DF.to_csv(self.csv_filename,index=False,header=False)

    def read_boundary_file(self):
        # load in the boundary file describing the location of the border between
        # the oceanic and continental crust
        colnames = ['Easting','Northing','Elevation']
        self.FF = pd.read_csv(self.boundary_filename,skiprows=1,names=colnames,header=None)
        self.FF.Elevation = self.FF.Elevation + self.zbulk
    
    def build_zones(self):
        self.fsurf = interpolate.interp2d(self.xvec,self.yvec,self.surf_upper,kind='linear')
        self.DF['upp_surf'] = 0
        self.DF['low_surf'] = 0

        for index,row in self.DF.iterrows():
            self.DF.upp_surf[index] = self.fsurf(self.DF.x[index].copy(),self.DF.y[index].copy())

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

        self.DF.mat_zone[self.DF.x[:].copy() > self.DF.x_boun[:].copy()] = 3  # setting continental crust
        self.DF.mat_zone[self.DF.x[:].copy() <= self.DF.x_boun[:].copy()] = 2  # setting oceanic crust
        self.DF.mat_zone[self.DF.z[:].copy() < self.DF.low_surf[:].copy()] = 1  # setting below the crust
        self.DF.mat_zone[self.DF.z[:].copy() > self.DF.upp_surf[:].copy()] = 4  # setting the sediment zone

        self.DF['orig_ind'] = self.DF.index*1


        # create materials zone
        # check the .copy() part
        testerbob = self.DF[self.DF.mat_zone==1].copy()
        self.node_nums_bottom = testerbob[['orig_ind']].to_numpy()+1
        testerbob = self.DF[self.DF.mat_zone==2].copy()
        self.node_nums_middle_ocean = testerbob[['orig_ind']].to_numpy()+1
        testerbob = self.DF[self.DF.mat_zone==3].copy()
        self.node_nums_middle_continental = testerbob[['orig_ind']].to_numpy()+1
        testerbob = self.DF[self.DF.mat_zone==4].copy()
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
        RZ.write('file\n')
        RZ.write('%s\n' % self.cond_filename)
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
        RZ.write('file\n')
        RZ.write('%s\n' % self.mat_prop_filename)
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
