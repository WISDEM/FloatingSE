from openmdao.api import Problem, ScipyOptimizer, pyOptSparseDriver, DumpRecorder
import numpy as np

#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
from mayavi import mlab


NSECTIONS = 5
NPTS = 100

def vecOption(x, in1s):
    myones = in1s if type(in1s) == type(np.array([])) else np.ones((in1s,))
    return (x*myones) if type(x)==type(0.0) or len(x) == 1 else x


def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

class FloatingInstance(object):
    def __init__(self):
        self.prob = Problem()
        self.params = {}

        # Environmental parameters
        self.params['water_depth'] = 200.0
        #self.params['air_density'] = 1.198
        self.params['base.windLoads.rho'] = 1.198
        #self.params['air_viscosity'] = 1.81e-5
        self.params['base.windLoads.mu'] = 1.81e-5
        self.params['water_density'] = 1025.0
        #self.params['water_viscosity'] = 8.9e-4
        self.params['base.waveLoads.mu'] = 8.9e-4
        #self.params['wave_height'] = 10.8
        self.params['hmax'] = 10.8
        #self.params['wave_period'] = 9.8
        self.params['T'] = 9.8
        self.params['Uc'] = 0.0
        #self.params['wind_reference_speed'] = 11.0
        self.params['Uref'] = 11.0
        #self.params['wind_reference_height'] = 90.0
        self.params['zref'] = 119.0
        #self.params['alpha'] = 0.11
        self.params['shearExp'] = 0.11
        #self.params['morison_mass_coefficient'] = 2.0
        self.params['cm'] = 2.0
        self.params['z0'] = 0.0
        self.params['yaw'] = 0.0
        self.params['beta'] = 0.0
        self.params['cd_usr'] = np.inf

        # Mooring parameters
        self.params['mooring_max_offset'] = 0.1*self.params['water_depth'] # Assumption        
        self.params['mooring_max_heel'] = 10.0
        self.params['number_of_mooring_lines'] = 3
        self.params['mooring_type'] = 'chain'
        self.params['anchor_type'] = 'suctionpile'
        self.params['mooring_cost_rate'] = 1.1
        self.params['drag_embedment_extra_length'] = 300.0
        self.params['number_of_mooring_lines'] = 3

        # Steel properties
        self.params['material_density'] = 7850.0
        self.params['E'] = 200e9
        self.params['G'] = 79.3e9
        self.params['nu'] = 0.26
        self.params['yield_stress'] = 3.45e8
        self.params['loading'] = 'hydrostatic'

        # Design constraints
        self.params['min_taper_ratio'] = 0.4
        self.params['min_diameter_thickness_ratio'] = 120.0

        # Safety factors
        self.params['gamma_f'] = 1.35
        self.params['gamma_b'] = 1.1
        self.params['gamma_m'] = 1.1
        self.params['gamma_n'] = 1.0
        self.params['gamma_fatigue'] = 1.755
        
        # Typically static- set defaults
        self.params['permanent_ballast_density'] = 4492.0
        self.params['bulkhead_mass_factor'] = 1.0
        self.params['ring_mass_factor'] = 1.0
        self.params['shell_mass_factor'] = 1.0
        self.params['column_mass_factor'] = 1.05
        self.params['outfitting_mass_fraction'] = 0.06
        self.params['ballast_cost_rate'] = 100.0
        self.params['tapered_col_cost_rate'] = 4720.0
        self.params['outfitting_cost_rate'] = 6980.0
        self.params['cross_attachment_pontoons'] = True
        self.params['lower_attachment_pontoons'] = True
        self.params['upper_attachment_pontoons'] = True
        self.params['lower_ring_pontoons'] = True
        self.params['upper_ring_pontoons'] = True
        self.params['outer_cross_pontoons'] = True #False
        self.params['pontoon_cost_rate'] = 6.250

        # OC4 Tower
        self.params['hub_height'] = 87.6
        self.params['tower_outer_diameter']    = np.linspace(6.5, 3.87, NSECTIONS+1)
        self.params['tower_section_height']    = vecOption(77.6/NSECTIONS, NSECTIONS)
        self.params['tower_wall_thickness']    = np.linspace(0.027, 0.019, NSECTIONS+1)
        self.params['tower_buckling_length']   = 30.0
        self.params['tower_outfitting_factor'] = 1.07
        self.params['rna_mass'] = 350e3 #285598.8
        self.params['rna_I'] = np.array([1.14930678e+08, 2.20354030e+07, 1.87597425e+07, 0.0, 5.03710467e+05, 0.0])
        self.params['rna_cg'] = np.array([-1.13197635, 0.0, 0.50875268])
        # Max thrust
        self.params['rna_force']  = np.array([1284744.196, 0.0,  -112400.5527])
        self.params['rna_moment'] = np.array([3963732.762, 896380.8464,  -346781.6819])
        # Max wind speed
        #self.params['rna_force']  = np.array([188038.8045, 0,  -16451.2637])
        #self.params['rna_moment'] = np.array([0.0, 131196.8431,  0.0])
        
        # Typically design (start at OC4 semi)
        self.params['radius_to_auxiliary_column'] = 28.867513459481287
        self.params['number_of_auxiliary_columns'] = 3
        self.params['base_freeboard'] = 10.0
        self.params['auxiliary_freeboard'] = 12.0
        self.params['fairlead'] = 14.0
        self.params['fairlead_offset_from_shell'] = 40.868-28.867513459481287-6.0
        self.params['base_outer_diameter'] = 6.5
        self.params['base_wall_thickness'] = 0.03
        self.params['auxiliary_wall_thickness'] = 0.06
        self.params['base_permanent_ballast_height'] = 1.0
        self.params['base_stiffener_web_height'] = 0.1
        self.params['base_stiffener_web_thickness'] = 0.04
        self.params['base_stiffener_flange_width'] = 0.1
        self.params['base_stiffener_flange_thickness'] = 0.02
        self.params['base_stiffener_spacing'] = 0.4
        self.params['auxiliary_permanent_ballast_height'] = 0.1
        self.params['auxiliary_stiffener_web_height'] = 0.1
        self.params['auxiliary_stiffener_web_thickness'] = 0.04
        self.params['auxiliary_stiffener_flange_width'] = 0.1
        self.params['auxiliary_stiffener_flange_thickness'] = 0.02
        self.params['auxiliary_stiffener_spacing'] = 0.4
        self.params['pontoon_outer_diameter'] = 2*1.6
        self.params['pontoon_wall_thickness'] = 0.0175
        self.params['connection_ratio_max'] = 0.25
        self.params['base_pontoon_attach_lower'] = -20.0
        self.params['base_pontoon_attach_upper'] = 10.0
        
        self.set_length_base( 30.0 )
        self.set_length_aux( 32.0 )

        self.params['auxiliary_section_height'] = np.array([6.0, 0.1, 7.9, 8.0, 10.0])
        self.params['auxiliary_outer_diameter'] = 2*np.array([12.0, 12.0, 6.0, 6.0, 6.0, 6.0])

        self.params['scope_ratio'] = 835.5 / (self.params['water_depth']-self.params['fairlead']) 
        self.params['anchor_radius'] = 837.6
        self.params['mooring_diameter'] = 0.0766

    def set_length_base(self, inval):
        self.params['base_section_height'] =  vecOption(inval/NSECTIONS, NSECTIONS)
        
    def set_length_aux(self, inval):
        self.params['auxiliary_section_height'] =  vecOption(inval/NSECTIONS, NSECTIONS)
        
    def check_vectors(self):
        self.params['tower_outer_diameter']            = vecOption(self.params['tower_outer_diameter'], NSECTIONS+1)
        self.params['tower_wall_thickness']            = vecOption(self.params['tower_wall_thickness'], NSECTIONS+1)
        self.params['tower_section_height']            = vecOption(self.params['tower_section_height'], NSECTIONS+1)
        self.params['base_outer_diameter']             = vecOption(self.params['base_outer_diameter'], NSECTIONS+1)
        self.params['base_wall_thickness']             = vecOption(self.params['base_wall_thickness'], NSECTIONS+1)
        self.params['base_stiffener_web_height']       = vecOption(self.params['base_stiffener_web_height'], NSECTIONS)
        self.params['base_stiffener_web_thickness']    = vecOption(self.params['base_stiffener_web_thickness'], NSECTIONS)
        self.params['base_stiffener_flange_width']     = vecOption(self.params['base_stiffener_flange_width'], NSECTIONS)
        self.params['base_stiffener_flange_thickness'] = vecOption(self.params['base_stiffener_flange_thickness'], NSECTIONS)
        self.params['base_stiffener_spacing']          = vecOption(self.params['base_stiffener_spacing'], NSECTIONS)
        self.params['base_bulkhead_nodes']             = [False] * (NSECTIONS+1)
        self.params['base_bulkhead_nodes'][0]          = True
        self.params['base_bulkhead_nodes'][1]          = True
        
        self.params['auxiliary_outer_diameter']             = vecOption(self.params['auxiliary_outer_diameter'], NSECTIONS+1)
        self.params['auxiliary_wall_thickness']             = vecOption(self.params['auxiliary_wall_thickness'], NSECTIONS+1)
        self.params['auxiliary_stiffener_web_height']       = vecOption(self.params['auxiliary_stiffener_web_height'], NSECTIONS)
        self.params['auxiliary_stiffener_web_thickness']    = vecOption(self.params['auxiliary_stiffener_web_thickness'], NSECTIONS)
        self.params['auxiliary_stiffener_flange_width']     = vecOption(self.params['auxiliary_stiffener_flange_width'], NSECTIONS)
        self.params['auxiliary_stiffener_flange_thickness'] = vecOption(self.params['auxiliary_stiffener_flange_thickness'], NSECTIONS)
        self.params['auxiliary_stiffener_spacing']          = vecOption(self.params['auxiliary_stiffener_spacing'], NSECTIONS)
        self.params['auxiliary_bulkhead_nodes']             = [False] * (NSECTIONS+1)
        self.params['auxiliary_bulkhead_nodes'][0]          = True
        self.params['auxiliary_bulkhead_nodes'][1]          = True
        
    def get_assembly(self):
        raise NotImplementedError("Subclasses should implement this!")

    def get_design_variables(self):
        raise NotImplementedError("Subclasses should implement this!")
        
    def init_optimization(self, optimizer=None):
        # Establish the optimization driver
        validStr = 'Valid options are: [COBYLA, SLSQP, CONMIN, PSQP]'
        if optimizer is None:
            return
        elif optimizer.upper() in ['COBYLA','SLSQP']:
            self.prob.driver = ScipyOptimizer()
        elif optimizer.upper() in ['CONMIN', 'PSQP','SNOPT', 'NSGA2']:
            self.prob.driver = pyOptSparseDriver()
        elif optimizer.upper() in ['ALPSO', 'SLSQP']:
            print 'WARNING: These optimizers run but jump to infeasible values. '+validStr
            self.prob.driver = pyOptSparseDriver()
        else:
            raise ValueError('Unknown or unworking optimizer. '+validStr)

        # Optimizer specific parameters
        self.prob.driver.options['optimizer'] = optimizer.upper()
        if optimizer.upper() == 'CONMIN':
            self.prob.driver.opt_settings['ITMAX'] = 1000
        elif optimizer.upper() in ['PSQP']:
            self.prob.driver.opt_settings['MIT'] = 100
        elif optimizer.upper() in ['NSGA2']:
            self.prob.driver.opt_settings['PopSize'] = 200
            self.prob.driver.opt_settings['maxGen'] = 2000
        elif optimizer.upper() in ['SNOPT']:
            self.prob.driver.opt_settings['Major iterations limit'] = 500
            self.prob.driver.opt_settings['Minor iterations limit'] = 250
            #self.prob.driver.opt_settings['Major optimality tolerance'] = 1e-5
            #self.prob.driver.opt_settings['Major feasibility tolerance'] = 1e-6
            #self.prob.driver.opt_settings['Minor feasibility tolerance'] = 1e-6
            #self.prob.driver.opt_settings['Function precision'] = 1e-12
            #self.prob.driver.opt_settings['Linesearch tolerance'] = 0.4
            #self.prob.driver.opt_settings['LU singularity tolerance'] = 1e30
        elif optimizer.upper() in ['COBYLA','SLSQP']:
            self.prob.driver.options['tol'] = 1e-6
            self.prob.driver.options['maxiter'] = 100000

        # Add in design variables
        desvarList = self.get_design_variables()
        if optimizer.upper() in ['CONMIN','PSQP','ALPSO','NSGA2','SLSQP','SNOPT']:
            for ivar in desvarList:
                self.prob.driver.add_desvar(ivar[0], lower=ivar[1], upper=ivar[2])
        else:
            for ivar in desvarList:
                iscale=ivar[3]
                self.prob.driver.add_desvar(ivar[0], lower=iscale*ivar[1], upper=iscale*ivar[2], scaler=iscale)

    def get_constraints(self):
        raise NotImplementedError("Subclasses should implement this!")

    def add_objective(self):
        raise NotImplementedError("Subclasses should implement this!")

    def set_inputs(self):
        # Load all variables from local params dictionary
        localnames = self.params.keys()
        for ivar in localnames:
            try:
                self.prob[ivar] = self.params[ivar]
            except KeyError:
                print 'Cannot set: ', ivar
                continue
            except AttributeError as e:
                print 'Vector issues?: ', ivar
                print e
                raise e
            except ValueError as e:
                print 'Badding setting of: ', ivar
                print e
                raise e

        # Check that everything got set correctly
        namesAssembly = self.prob.root._unknowns_dict.keys()
        for ivar in namesAssembly:
            if self.prob.root._unknowns_dict[ivar].has_key('_canset_') and self.prob.root._unknowns_dict[ivar]['_canset_']:
                selfvar = ivar.split('.')[-1]
                if not selfvar in localnames:
                    print 'WARNING:', selfvar, 'has not been set!'
                    

        '''
        namesAssembly = self.prob.root._unknowns_dict.keys()
        for ivar in namesAssembly:
            if self.prob.root._unknowns_dict[ivar].has_key('_canset_') and self.prob.root._unknowns_dict[ivar]['_canset_']:
                selfvar = ivar.split('.')[-1]

                selfval = self.prob[selfvar]
                if ( (type(selfval) == type(0.0)) or (type(selfval) == np.float64) or (type(selfval) == np.float32) ) and selfval == 0.0:
                    print selfvar, 'is zero! Did you set it?'
                if ( (type(selfval) == type(0)) or (type(selfval) == np.int64) or (type(selfval) == np.int32) ) and selfval == 0:
                    print selfvar, 'is zero! Did you set it?'
                elif type(selfval) == type(np.array([])) and not np.any(selfval):
                    print selfvar, 'is zero! Did you set it?'

                selfval = getattr(self, selfvar, None)
                if selfval is None:
                    print 'Variable not found:', ivar, selfvar, self.prob[ivar]
                else:
                    self.prob[ivar] = selfval
        #raise NotImplementedError("Subclasses should implement this!")
        '''

    def store_results(self):
        localnames = self.params.keys()
        optDict = self.prob.driver.get_desvars()
        for ivar in optDict.keys():
            ival = optDict[ivar]
            if type(ival) == type(np.array([])) and len(ival) == 1: ival=ival[0]
            selfvar = ivar.split('.')[-1]
            self.prob[ivar] = ival
            if selfvar in localnames:
                self.params[ivar] = ival
                ivalstr = 'np.array( '+str(ival.tolist())+' )' if type(ival)==type(np.array([])) else str(ival)
                print ivar, '=', ivalstr
                
    def init_problem(self, optimizer=None):
        self.prob = Problem()
        self.prob.root = self.get_assembly()

        self.init_optimization(optimizer=optimizer)

        constlist = self.get_constraints()
        for con in constlist:
            self.prob.driver.add_constraint(con[0], lower=con[1], upper=con[2], equals=con[3])

        self.add_objective()

        # Recorder
        #recorder = DumpRecorder('floatingOptimization.dat')
        #recorder.options['record_params'] = True
        #recorder.options['record_metadata'] = False
        #recorder.options['record_derivs'] = False
        #self.prob.driver.add_recorder(recorder)
        
        # Note this command must be done after the constraints, design variables, and objective have been set,
        # but before the initial conditions are specified (unless we use the default initial conditions )
        # After setting the intial conditions, running setup() again will revert them back to default values
        self.prob.setup()

        self.set_inputs()

        # Checks
        self.prob.check_setup()
        self.prob.pre_run_check()
        #self.prob.check_total_derivatives()

    def run(self, optimizer=None):
        self.init_problem(optimizer)
        self.prob.run()
        if not optimizer is None:
            self.store_results()
            print self.prob.driver.get_constraints()
            #print self.prob.driver.get_desvars()
            print self.prob.driver.get_objectives()
        
    def evaluate(self, optimizer=None):
        self.init_problem(optimizer)
        self.prob.run_once()
        if not optimizer is None:
            #self.store_results()
            print self.prob.driver.get_constraints()
            print self.prob.driver.get_desvars()
            print self.prob.driver.get_objectives()

    def visualize(self, fname=None):
        raise NotImplementedError("Subclasses should implement this!")

    def init_figure(self):
        mysky   = np.array([135, 206, 250]) / 255.0
        mysky   = tuple(mysky.tolist())
        #fig = plt.figure()
        #ax = fig.add_subplot(111, projection='3d')
        #fig = mlab.figure(bgcolor=(1,)*3, size=(1600,1100))
        #fig = mlab.figure(bgcolor=mysky, size=(1600,1100))
        fig = mlab.figure(bgcolor=(0,)*3, size=(1600,1100))
        return fig

    def draw_ocean(self, fig=None):
        if fig is None: fig=self.init_figure()
        npts = 100
        
        #mybrown = np.array([244, 170, 66]) / 255.0
        #mybrown = tuple(mybrown.tolist())
        mywater = np.array([95, 158, 160 ]) / 255.0 #(0.0, 0.0, 0.8) [143, 188, 143]
        mywater = tuple(mywater.tolist())
        alpha   = 0.3

        # Waterplane box
        x = y = 50 * np.linspace(-1, 1, npts)
        X,Y = np.meshgrid(x,y)
        Z   = np.sin(100*X*Y) #np.zeros(X.shape)
        #ax.plot_surface(X, Y, Z, alpha=alpha, color=mywater)
        mlab.mesh(X, Y, Z, opacity=alpha, color=mywater, figure=fig)
        
        # Sea floor
        Z = -self.params['water_depth'] * np.ones(X.shape)
        #ax.plot_surface(10*X, 10*Y, Z, alpha=1.0, color=mybrown)
        #mlab.mesh(10*X,10*Y,Z, opacity=1.0, color=mybrown, figure=fig)

        # Sides
        #x = 500 * np.linspace(-1, 1, npts)
        #z = self.params['water_depth'] * np.linspace(-1, 0, npts)
        #X,Z = np.meshgrid(x,z)
        #Y = x.max()*np.ones(Z.shape)
        ##ax.plot_surface(X, Y, Z, alpha=alpha, color=mywater)
        #mlab.mesh(X,Y,Z, opacity=alpha, color=mywater, figure=fig)
        #mlab.mesh(X,-Y,Z, opacity=alpha, color=mywater, figure=fig)
        #mlab.mesh(Y,X,Z, opacity=alpha, color=mywater, figure=fig)
        ##mlab.mesh(-Y,X,Z, opacity=alpha, color=mywater, figure=fig)

    def draw_mooring(self, fig, mooring):
        mybrown = np.array([244, 170, 66]) / 255.0
        mybrown = tuple(mybrown.tolist())
        npts    = 100
        
        # Sea floor
        r  = np.linspace(0, self.params['anchor_radius'], npts)
        th = np.linspace(0, 2*np.pi, npts)
        R, TH = np.meshgrid(r, th)
        X = R*np.cos(TH)
        Y = R*np.sin(TH)
        Z = -self.params['water_depth'] * np.ones(X.shape)
        #ax.plot_surface(X, Y, Z, alpha=1.0, color=mybrown)
        mlab.mesh(X,Y,Z, opacity=1.0, color=mybrown, figure=fig)

        cmoor = (0,1,0)
        for k in xrange(self.params['number_of_mooring_lines']):
            #ax.plot(mooring[k,:,0], mooring[k,:,1], mooring[k,:,2], 'k', lw=2)
            mlab.plot3d(mooring[k,:,0], mooring[k,:,1], mooring[k,:,2], color=cmoor, tube_radius=0.5*self.params['mooring_diameter'], figure=fig)

            
    def draw_pontoons(self, fig, truss, R, freeboard):
        nE = truss.shape[0]
        c = (0.5,0,0)
        for k in xrange(nE):
            if np.any(truss[k,2,:] > freeboard): continue
            mlab.plot3d(truss[k,0,:], truss[k,1,:], truss[k,2,:], color=c, tube_radius=R, figure=fig)

            
    def draw_column(self, fig, centerline, freeboard, h_section, r_nodes, spacingVec=None, ckIn=None):
        npts = 20
        
        z_nodes = np.flipud( freeboard - np.r_[0.0, np.cumsum(np.flipud(h_section))] )
        
        th = np.linspace(0, 2*np.pi, npts)
        for k in xrange(NSECTIONS):
            rk = np.linspace(r_nodes[k], r_nodes[k+1], npts)
            z  = np.linspace(z_nodes[k], z_nodes[k+1], npts)
            R, TH = np.meshgrid(rk, th)
            Z, _  = np.meshgrid(z, th)
            X = R*np.cos(TH) + centerline[0]
            Y = R*np.sin(TH) + centerline[1]

            # Draw parameters
            if ckIn is None:
                ck = (0.6,)*3 if np.mod(k,2) == 0 else (0.4,)*3
            else:
                ck = ckIn
            #ax.plot_surface(X, Y, Z, alpha=0.5, color=ck)
            mlab.mesh(X, Y, Z, opacity=0.9, color=ck, figure=fig)

            if spacingVec is None: continue
            
            z = z_nodes[k] + spacingVec[k]
            while z < z_nodes[k+1]:
                rk = np.interp(z, z_nodes[k:], r_nodes[k:])
                #print z, z_nodes[k], z_nodes[k+1], rk, r_nodes[k], r_nodes[k+1]
                #ax.plot(rk*np.cos(th), rk*np.sin(th), z*np.ones(th.shape), 'r', lw=0.25)
                mlab.plot3d(rk*np.cos(th) + centerline[0], rk*np.sin(th) + centerline[1], z*np.ones(th.shape), color=(0.5,0,0), figure=fig)
                z += spacingVec[k]
                
                '''
                # Web
                r   = np.linspace(rk - self.params['stiffener_web_height'][k], rk, npts)
                R, TH = np.meshgrid(r, th)
                Z, _  = np.meshgrid(z, th)
                X = R*np.cos(TH)
                Y = R*np.sin(TH)
                ax.plot_surface(X, Y, Z, alpha=0.7, color='r')

                # Flange
                r = r[0]
                h = np.linspace(0, self.params['stiffener_flange_width'][k], npts)
                zflange = z + h - 0.5*self.params['stiffener_flange_width'][k]
                R, TH = np.meshgrid(r, th)
                Z, _  = np.meshgrid(zflange, th)
                X = R*np.cos(TH)
                Y = R*np.sin(TH)
                ax.plot_surface(X, Y, Z, alpha=0.7, color='r')
                '''

    def set_figure(self, fig, fname=None):
        #ax.set_aspect('equal')
        #set_axes_equal(ax)
        #ax.autoscale_view(tight=True)
        #ax.set_xlim([-125, 125])
        #ax.set_ylim([-125, 125])
        #ax.set_zlim([-220, 30])
        #plt.axis('off')
        #plt.show()
        mlab.move([-517.16728532, -87.0711504, 5.60826224], [1.35691603e+01, -2.84217094e-14, -1.06547500e+02])
        mlab.view(-170.68320804213343, 78.220729198686854, 549.40101471336777, [1.35691603e+01,  0.0, -1.06547500e+02])
        if not fname is None: mlab.savefig(fname, figure=fig)
        mlab.show(stop=True)

        
    
