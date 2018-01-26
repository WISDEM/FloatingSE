from openmdao.api import Problem, ScipyOptimizer, pyOptSparseDriver
import numpy as np

#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
from mayavi import mlab


NSECTIONS = 5
NPTS = 100

def nodal2sectional(x):
    """Averages nodal data to be length-1 vector of sectional data

    INPUTS:
    ----------
    x   : float vector, nodal data

    OUTPUTS:
    -------
    y   : float vector,  sectional data
    """
    return 0.5*(x[:-1] + x[1:])

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

        # Environmental parameters
        self.water_depth = 218.0
        self.air_density = 1.198
        self.air_viscosity = 1.81e-5
        self.water_density = 1025.0
        self.water_viscosity = 8.9e-4
        self.wave_height = 10.8
        self.wave_period = 9.8
        self.wind_reference_speed = 11.0
        self.wind_reference_height = 119.0
        self.alpha = 0.11
        self.morison_mass_coefficient = 2.0

        # Mooring parameters
        self.max_offset  = 0.1*self.water_depth # Assumption        
        self.number_of_mooring_lines = 3
        self.mooring_type = 'chain'
        self.anchor_type = 'suctionpile'
        self.mooring_cost_rate = 1.1
        self.scope_ratio = 2.6
        self.anchor_radius = 420.0
        self.drag_embedment_extra_length = 300.0
        
        self.mooring_diameter = 0.14

        # Turbine parameters
        self.turbine_mass = 371690.0 + 285599.0
        self.turbine_center_of_gravity = np.array([0.0, 0.0, 43.4])
        self.turbine_surge_force = 1.3e6
        self.turbine_pitch_moment = 107850803.0
        self.tower_diameter = 6.5

        # Steel properties
        self.material_density = 7850.0
        self.E = 200e9
        self.G = 79.3e9
        self.nu = 0.26
        self.yield_stress = 3.45e8

        # Design parameters
        self.min_taper_ratio = 0.4
        self.min_diameter_thickness_ratio = 120.0
        
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
            self.prob.driver.opt_settings['MIT'] = 10000
        elif optimizer.upper() in ['NSGA2']:
            self.prob.driver.opt_settings['PopSize'] = 200
            self.prob.driver.opt_settings['maxGen'] = 2000
        elif optimizer.upper() in ['SNOPT']:
            self.prob.driver.opt_settings['Major iterations limit'] = 10000
            self.prob.driver.opt_settings['Minor iterations limit'] = 1000
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

    def add_constraints_objective(self):
        raise NotImplementedError("Subclasses should implement this!")

    def set_inputs(self):
        namesAssembly = self.prob.root._unknowns_dict.keys()
        for ivar in namesAssembly:
            if self.prob.root._unknowns_dict[ivar].has_key('_canset_') and self.prob.root._unknowns_dict[ivar]['_canset_']:
                selfvar = ivar.split('.')[0]
                selfval = getattr(self, selfvar, None)
                if selfval is None:
                    print 'Variable not found:', ivar, selfvar, self.prob[ivar]
                else:
                    self.prob[ivar] = selfval
        #raise NotImplementedError("Subclasses should implement this!")

    def store_results(self):
        optDict = self.prob.driver.get_desvars()
        for ivar in optDict.keys():
            ival = optDict[ivar]
            if type(ival) == type(np.array([])) and len(ival) == 1: ival=ival[0]
            selfvar = ivar.split('.')[0]
            setattr(self, selfvar, ival)
            self.prob[ivar] = ival

    def init_problem(self, optimizer=None):
        self.prob = Problem()
        self.prob.root = self.get_assembly()

        self.init_optimization(optimizer=optimizer)

        self.add_constraints_objective()

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
            print self.prob.driver.get_desvars()
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
        alpha   = 0.6

        # Waterplane box
        x = y = 50 * np.linspace(-1, 1, npts)
        X,Y = np.meshgrid(x,y)
        Z   = np.sin(100*X*Y) #np.zeros(X.shape)
        #ax.plot_surface(X, Y, Z, alpha=alpha, color=mywater)
        mlab.mesh(X, Y, Z, opacity=alpha, color=mywater, figure=fig)
        
        # Sea floor
        Z = -self.water_depth * np.ones(X.shape)
        #ax.plot_surface(10*X, 10*Y, Z, alpha=1.0, color=mybrown)
        #mlab.mesh(10*X,10*Y,Z, opacity=1.0, color=mybrown, figure=fig)

        # Sides
        #x = 500 * np.linspace(-1, 1, npts)
        #z = self.water_depth * np.linspace(-1, 0, npts)
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
        r  = np.linspace(0, self.anchor_radius, npts)
        th = np.linspace(0, 2*np.pi, npts)
        R, TH = np.meshgrid(r, th)
        X = R*np.cos(TH)
        Y = R*np.sin(TH)
        Z = -self.water_depth * np.ones(X.shape)
        #ax.plot_surface(X, Y, Z, alpha=1.0, color=mybrown)
        mlab.mesh(X,Y,Z, opacity=1.0, color=mybrown, figure=fig)

        cmoor = (0,1,0)
        for k in xrange(self.number_of_mooring_lines):
            #ax.plot(mooring[k,:,0], mooring[k,:,1], mooring[k,:,2], 'k', lw=2)
            mlab.plot3d(mooring[k,:,0], mooring[k,:,1], mooring[k,:,2], color=cmoor, tube_radius=0.5*self.mooring_diameter, figure=fig)

            
    def draw_pontoons(self, fig, truss, R, freeboard):
        nE = truss.shape[0]
        c = (0.5,0,0)
        for k in xrange(nE):
            if np.any(truss[k,2,:] > freeboard): continue
            mlab.plot3d(truss[k,0,:], truss[k,1,:], truss[k,2,:], color=c, tube_radius=R, figure=fig)

            
    def draw_cylinder(self, fig, centerline, freeboard, h_section, r_nodes, spacingVec=None):
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
            ck = (0.6,)*3 if np.mod(k,2) == 0 else (0.4,)*3
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
                r   = np.linspace(rk - self.stiffener_web_height[k], rk, npts)
                R, TH = np.meshgrid(r, th)
                Z, _  = np.meshgrid(z, th)
                X = R*np.cos(TH)
                Y = R*np.sin(TH)
                ax.plot_surface(X, Y, Z, alpha=0.7, color='r')

                # Flange
                r = r[0]
                h = np.linspace(0, self.stiffener_flange_width[k], npts)
                zflange = z + h - 0.5*self.stiffener_flange_width[k]
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

        
    
