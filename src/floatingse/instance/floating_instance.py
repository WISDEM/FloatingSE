from __future__ import print_function
from floatingse.floating import FloatingSE
from openmdao.api import Problem, ScipyOptimizer, pyOptSparseDriver, HeuristicDriver, HeuristicDriverParallel, DumpRecorder
import numpy as np
import cPickle as pickle        
from StringIO import StringIO

from mayavi import mlab

NSECTIONS = 5
NPTS = 100
Ten_strings = ['DTU', 'DTU10', 'DTU10MW', '10', '10MW', 'DTU-10', 'DTU-10MW']
Five_strings = ['NREL', 'NREL5', 'NREL5MW', '5', '5MW', 'NREL-5', 'NREL-5MW']


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
        self.optimizerSet = False
        self.desvarSet = False
        self.constraintSet = False
        self.optimizer = None

        # Environmental parameters
        self.params['water_depth']               = 200.0
        #self.params['air_density']              = 1.198
        self.params['base.windLoads.rho']        = 1.198
        #self.params['air_viscosity']            = 1.81e-5
        self.params['base.windLoads.mu']         = 1.81e-5
        self.params['water_density']             = 1025.0
        #self.params['water_viscosity']          = 8.9e-4
        self.params['base.waveLoads.mu']         = 8.9e-4
        #self.params['wave_height']              = 10.8
        self.params['Hs']                      = 10.8
        #self.params['wave_period']              = 9.8
        self.params['T']                         = 9.8
        self.params['Uc']                        = 0.0
        #self.params['wind_reference_speed']     = 11.0
        self.params['Uref']                      = 11.0
        #self.params['wind_reference_height']    = 90.0
        self.params['zref']                      = 119.0
        #self.params['alpha']                    = 0.11
        self.params['shearExp']                  = 0.11
        #self.params['morison_mass_coefficient'] = 2.0
        self.params['cm']                        = 2.0
        self.params['z0']                        = 0.0
        self.params['yaw']                       = 0.0
        self.params['beta']                      = 0.0
        self.params['cd_usr']                    = np.inf
        self.params['z_offset'] = 0.0
        
        # Encironmental constaints
        self.params['wave_period_range_low']                = 2.0
        self.params['wave_period_range_high']               = 20.0
        
        # Mooring parameters
        self.params['mooring_max_offset']                   = 0.1*self.params['water_depth'] # Assumption        
        self.params['mooring_operational_heel']             = 6.0
        self.params['max_survival_heel']                    = 15.0
        self.params['mooring_type']                         = 'chain'
        self.params['anchor_type']                          = 'suctionpile'
        self.params['mooring_cost_rate']                    = 1.1
        self.params['number_of_mooring_connections']        = 3
        self.params['mooring_lines_per_connection']         = 1

        # Steel properties
        self.params['material_density']                     = 7850.0
        self.params['E']                                    = 200e9
        self.params['G']                                    = 79.3e9
        self.params['nu']                                   = 0.26
        self.params['yield_stress']                         = 3.45e8
        self.params['loading']                              = 'hydrostatic'

        # Design constraints
        self.params['max_taper_ratio']                      = 0.2
        self.params['min_diameter_thickness_ratio']         = 120.0

        # Safety factors
        self.params['gamma_f']                              = 1.35
        self.params['gamma_b']                              = 1.1
        self.params['gamma_m']                              = 1.1
        self.params['gamma_n']                              = 1.0
        self.params['gamma_fatigue']                        = 1.755
        
        # Typically static- set defaults
        self.params['permanent_ballast_density']            = 4492.0
        self.params['bulkhead_mass_factor']                 = 1.0
        self.params['ring_mass_factor']                     = 1.0
        self.params['shell_mass_factor']                    = 1.0
        self.params['column_mass_factor']                   = 1.05
        self.params['outfitting_mass_fraction']             = 0.06
        self.params['ballast_cost_rate']                    = 100.0
        self.params['tapered_col_cost_rate']                = 4720.0
        self.params['outfitting_cost_rate']                 = 6980.0
        self.params['cross_attachment_pontoons_int']        = 1
        self.params['lower_attachment_pontoons_int']        = 1
        self.params['upper_attachment_pontoons_int']        = 1
        self.params['lower_ring_pontoons_int']              = 1
        self.params['upper_ring_pontoons_int']              = 1
        self.params['outer_cross_pontoons_int']             = 1 #False
        self.params['pontoon_cost_rate']                    = 6.250

        # OC4 Tower
        self.params['hub_height']                           = 90.0
        self.params['tower_outer_diameter']                 = np.linspace(6.5, 3.87, NSECTIONS+1)
        self.params['tower_section_height']                 = vecOption(77.6/NSECTIONS, NSECTIONS)
        self.params['tower_wall_thickness']                 = np.linspace(0.027, 0.019, NSECTIONS+1)
        self.params['tower_buckling_length']                = 30.0
        self.params['tower_outfitting_factor']              = 1.07
        self.params['rna_mass']                             = 350e3 #285598.8
        self.params['rna_I']                                = np.array([1.14930678e+08, 2.20354030e+07, 1.87597425e+07, 0.0, 5.03710467e+05, 0.0])
        self.params['rna_cg']                               = np.array([-1.13197635, 0.0, 0.50875268])
        # Max thrust
        self.params['rna_force']                            = np.array([1284744.196, 0.0,  -112400.5527])
        self.params['rna_moment']                           = np.array([3963732.762, 896380.8464,  -346781.6819])
        # Max wind speed
        #self.params['rna_force']                           = np.array([188038.8045, 0,  -16451.2637])
        #self.params['rna_moment']                          = np.array([0.0, 131196.8431,  0.0])
        self.params['base_bulkhead_thickness']              = 0.05*np.array([1, 1, 0, 0, 0, 1]) # Locations/thickness of internal bulkheads at section interfaces [m]
        self.params['auxiliary_bulkhead_thickness']         = 0.05*np.array([1, 1, 0, 0, 0, 1]) # Locations/thickness of internal bulkheads at section interfaces [m]
        self.params['sg.Rhub']                              = 1.125
        
        # Typically design (start at OC4 semi)
        self.params['radius_to_auxiliary_column']           = 28.867513459481287
        self.params['number_of_auxiliary_columns']          = 3
        self.params['base_freeboard']                       = 10.0
        self.params['auxiliary_freeboard']                  = 12.0
        self.params['fairlead']                             = 14.0
        self.params['fairlead_offset_from_shell']           = 40.868-28.867513459481287-6.0
        self.params['base_outer_diameter']                  = 6.5
        self.params['base_wall_thickness']                  = 0.03
        self.params['auxiliary_wall_thickness']             = 0.06
        self.params['base_permanent_ballast_height']        = 1.0
        self.params['base_stiffener_web_height']            = 0.1
        self.params['base_stiffener_web_thickness']         = 0.04
        self.params['base_stiffener_flange_width']          = 0.1
        self.params['base_stiffener_flange_thickness']      = 0.02
        self.params['base_stiffener_spacing']               = 0.4
        self.params['auxiliary_permanent_ballast_height']   = 0.1
        self.params['auxiliary_stiffener_web_height']       = 0.1
        self.params['auxiliary_stiffener_web_thickness']    = 0.04
        self.params['auxiliary_stiffener_flange_width']     = 0.1
        self.params['auxiliary_stiffener_flange_thickness'] = 0.02
        self.params['auxiliary_stiffener_spacing']          = 0.4
        self.params['fairlead_support_outer_diameter']      = 2*1.6
        self.params['fairlead_support_wall_thickness']      = 0.0175
        self.params['pontoon_outer_diameter']               = 2*1.6
        self.params['pontoon_wall_thickness']               = 0.0175
        self.params['connection_ratio_max']                 = 0.25
        self.params['base_pontoon_attach_lower']            = 0.1
        self.params['base_pontoon_attach_upper']            = 1.0
        self.params['base_heave_plate_diameter']            = 0.0
        self.params['auxiliary_heave_plate_diameter']       = 0.0
        
        self.set_length_base( 30.0 )
        self.set_length_aux( 32.0 )

        self.params['auxiliary_outer_diameter']             = 2*6.0
        self.params['auxiliary_heave_plate_diameter']       = 24.0

        self.params['mooring_line_length']                  = 835.5
        self.params['anchor_radius']                        = 837.6
        self.params['mooring_diameter']                     = 0.0766

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
        self.params['base_bulkhead_thickness']         = vecOption(self.params['base_bulkhead_thickness'], NSECTIONS+1)
        #self.params['base_bulkhead_thickness'][:2] = 0.05
        
        self.params['auxiliary_outer_diameter']             = vecOption(self.params['auxiliary_outer_diameter'], NSECTIONS+1)
        self.params['auxiliary_wall_thickness']             = vecOption(self.params['auxiliary_wall_thickness'], NSECTIONS+1)
        self.params['auxiliary_stiffener_web_height']       = vecOption(self.params['auxiliary_stiffener_web_height'], NSECTIONS)
        self.params['auxiliary_stiffener_web_thickness']    = vecOption(self.params['auxiliary_stiffener_web_thickness'], NSECTIONS)
        self.params['auxiliary_stiffener_flange_width']     = vecOption(self.params['auxiliary_stiffener_flange_width'], NSECTIONS)
        self.params['auxiliary_stiffener_flange_thickness'] = vecOption(self.params['auxiliary_stiffener_flange_thickness'], NSECTIONS)
        self.params['auxiliary_stiffener_spacing']          = vecOption(self.params['auxiliary_stiffener_spacing'], NSECTIONS)
        self.params['auxiliary_bulkhead_thickness']         = vecOption(self.params['auxiliary_bulkhead_thickness'], NSECTIONS+1)
        #self.params['auxiliary_bulkhead_thickness'][:2] = 0.05


    def set_reference(self, instr):
        if instr.upper() in Five_strings:

            self.params['hub_height']              = 90.0
            self.params['tower_outer_diameter']    = np.linspace(6.5, 3.87, NSECTIONS+1)
            self.params['tower_section_height']    = vecOption(77.6/NSECTIONS, NSECTIONS)
            self.params['tower_wall_thickness']    = np.linspace(0.027, 0.019, NSECTIONS+1)
            self.params['sg.Rhub']                 = 1.125
            self.params['base_freeboard'] = 10.0

            if self.params.has_key('rna_mass'):
                self.params['rna_mass'] = 350e3 #285598.8
                self.params['rna_I'] = np.array([1.14930678e+08, 2.20354030e+07, 1.87597425e+07, 0.0, 5.03710467e+05, 0.0])
                self.params['rna_cg'] = np.array([-1.13197635, 0.0, 0.50875268])
                # Max thrust
                self.params['rna_force']      = np.array([ 1.03086517e+06, 0.0, -3.53463639e+06])
                self.params['rna_moment']     = np.array([9817509.35136043, 566675.9644231, -858920.77230378])
                # Max wind speed
                #self.params['rna_force']     = np.array([188038.8045, 0,  -16451.2637])
                #self.params['rna_moment']    = np.array([0.0, 131196.8431,  0.0])
                
            
        elif instr.upper() in Ten_strings:

            #z Do Di Ixx= Iyy J
            #[m] [m] [m] [m4] [m4]
            dtuTowerData = StringIO("""
            145.63 5.50 5.44 2.03 4.07
            134.55 5.79 5.73 2.76 5.52
            124.04 6.07 6.00 3.63 7.26
            113.54 6.35 6.26 4.67 9.35
            103.03 6.63 6.53 5.90 11.80
            92.53 6.91 6.80 7.33 14.67
            82.02 7.19 7.07 8.99 17.98
            71.52 7.46 7.34 10.9 21.80
            61.01 7.74 7.61 13.08 26.15
            50.51 8.02 7.88 15.55 31.09
            40.00 8.30 8.16 15.55 31.09
            40.00 9.00 8.70 34.84 69.68
            38.00 9.00 8.70 35.74 71.48
            36.00 9.00 8.70 36.66 73.32
            34.00 9.00 8.70 37.59 75.18
            32.00 9.00 8.69 38.54 77.08
            30.00 9.00 8.69 39.51 79.01
            28.00 9.00 8.69 40.49 80.97
            26.00 9.00 8.69 41.48 82.97
            24.00 9.00 8.69 42.5 85.00
            22.00 9.00 8.69 43.53 87.05
            20.00 9.00 8.69 44.05 88.10
            16.00 9.00 8.80 28.09 56.18
            12.00 9.00 8.80 28.09 56.18
            8.00 9.00 8.80 28.09 56.18
            4.00 9.00 8.80 28.09 56.18
            0.00 9.00 8.80 28.09 56.18
            -8.400 9.00 8.80 28.09 56.18
            -16.800 9.00 8.80 28.09 56.18
            -25.20 9.00 8.80 28.09 56.18
            -33.60 9.00 8.80 28.09 56.18
            -42.60 9.00 8.80 28.09 56.18
            """)

            
            self.params['hub_height'] = 149.0
            self.params['base_freeboard'] = 30.0
            self.params['base_section_height'] += 3.0
            towerData = np.loadtxt(dtuTowerData)
            towerData = towerData[(towerData[:,0] >= 30.0),:]
            towerData = np.vstack((towerData[0,:], towerData))
            towerData[0,0] = self.params['hub_height']
            towerData[(towerData[:,0] == 40.0),0] += np.array([0.01, 0.0])
            trans_idx = np.where(towerData[:,1] == towerData[-1,1])[0]
            idx = [0, 1, trans_idx[0]/2, trans_idx[0]-1, trans_idx[0], trans_idx[-1]]
            self.params['tower_section_height'] = np.diff( np.flipud( towerData[idx,0] ) )
            self.params['tower_outer_diameter'] = np.flipud( towerData[idx, 1] )
            self.params['tower_wall_thickness'] = np.flipud( towerData[idx, 1] - towerData[idx, 2] )

            if self.params.has_key('rna_mass'):
                self.params['rna_mass'] = 350e3 #285598.8
                self.params['rna_I'] = np.array([1.14930678e+08, 2.20354030e+07, 1.87597425e+07, 0.0, 5.03710467e+05, 0.0])
                self.params['rna_cg'] = np.array([-1.13197635, 0.0, 0.50875268])
                self.params['rna_force']   = np.array([ 2.11271060e+06, 0.0, -7.25225356e+06])
                self.params['rna_moment']  = np.array([29259007.24076359,  1422330.08481948, -3075245.58067333])
                self.params['sg.Rhub']     = 2.3
            
        else:
            raise ValueError('Inputs must be either NREL5MW or DTU10MW')

    def save(self, fname):
        assert type(fname) == type(''), 'Input filename must be a string'
        with open(fname,'wb') as fp:
            pickle.dump(self.params, fp)
            
    def load(self, fname):
        assert type(fname) == type(''), 'Input filename must be a string'
        with open(fname,'rb') as fp:
            newparams = pickle.load(fp)
        for k in newparams.keys():
            self.params[k] = newparams[k]
        
    def get_assembly(self):
        return FloatingSE(NSECTIONS)

    
    def add_design_variable(self, varStr, lowVal, highVal):
        if not self.optimizerSet:
            raise RuntimeError('Must set the optimizer to set the driver first')
        
        assert type(varStr) == type(''), 'Input variable must be a string'
        assert isinstance(lowVal, (float, int, np.float32, np.float64, np.int32, np.int64)), 'Invalid lower bound'
        assert isinstance(highVal, (float, int, np.float32, np.float64, np.int32, np.int64)), 'Invalid upper bound'
        
        if self.optimizer in ['COBLYA']:
            iscale=min(1.0 / lowVal, highVal / 1.0)
            self.prob.driver.add_desvar(varStr, lower=lowVal, upper=highVal, scaler=iscale)
        else:
            self.prob.driver.add_desvar(varStr, lower=lowVal, upper=highVal)

        self.desvarSet = True

        
    def add_constraint(self, varStr, lowVal, highVal, eqVal):
        if not self.optimizerSet:
            raise RuntimeError('Must set the optimizer to set the driver first')
        
        assert type(varStr) == type(''), 'Input variable must be a string'
        if not lowVal is None:
            assert isinstance(lowVal, (float, int, np.float32, np.float64, np.int32, np.int64)), 'Invalid lower bound'
        if not highVal is None:
            assert isinstance(highVal, (float, int, np.float32, np.float64, np.int32, np.int64)), 'Invalid upper bound'
        if not eqVal is None:
            assert isinstance(eqVal, (float, int, np.float32, np.float64, np.int32, np.int64)), 'Invalid equality value'

        self.prob.driver.add_constraint(varStr, lower=lowVal, upper=highVal, equals=eqVal)

        self.constraintSet = True
        
            
    def set_optimizer(self, optStr):
        assert type(optStr) == type(''), 'Input optimizer must be a string'
        self.optimizer = optStr.upper()
        
        # Establish the optimization driver
        if self.optimizer in ['SOGA','SOPSO']:
            self.prob.driver = HeuristicDriverParallel()
        elif self.optimizer in ['COBYLA','SLSQP']:
            self.prob.driver = ScipyOptimizer()
        elif self.optimizer in ['CONMIN', 'PSQP','SNOPT','NSGA2','ALPSO']:
            self.prob.driver = pyOptSparseDriver()
        else:
            raise ValueError('Unknown or unworking optimizer. '+validStr)

        self.optimizerSet = True
        
        # Set default options
        self.prob.driver.options['optimizer'] = self.optimizer
        if self.optimizer == 'CONMIN':
            self.prob.driver.opt_settings['ITMAX'] = 1000
        elif self.optimizer in ['PSQP']:
            self.prob.driver.opt_settings['MIT'] = 1000
        elif self.optimizer in ['SOGA','SOPSO']:
            self.prob.driver.options['population'] = 50
            self.prob.driver.options['generations'] = 500
        elif self.optimizer in ['NSGA2']:
            self.prob.driver.opt_settings['PopSize'] = 200
            self.prob.driver.opt_settings['maxGen'] = 500
        elif self.optimizer in ['SNOPT']:
            self.prob.driver.opt_settings['Iterations limit'] = 500
            self.prob.driver.opt_settings['Major optimality tolerance'] = 1e-4
            self.prob.driver.opt_settings['Major feasibility tolerance'] = 1e-6
            self.prob.driver.opt_settings['Minor feasibility tolerance'] = 1e-6
            self.prob.driver.opt_settings['Function precision'] = 1e-8
        elif self.optimizer in ['COBYLA','SLSQP']:
            self.prob.driver.options['tol'] = 1e-6
            self.prob.driver.options['maxiter'] = 1000


    def set_options(self, indict):
        if not self.optimizerSet:
            raise RuntimeError('Must set the optimizer to set the driver first')
        assert isinstance(indict, dict), 'Options must be passed as a string:value dictionary'
        
        for k in indict.keys():
            if self.optimizer in ['SOGA','SOPSO','COBYLA','SLSQP']:
                self.prob.driver.options[k] = indict[k]
            elif self.optimizer in ['CONMIN', 'PSQP','SNOPT','NSGA2','ALPSO']:
                if k in ['title','print_results','gradient method']:
                    self.prob.driver.options[k] = indict[k]
                else:
                    self.prob.driver.opt_settings[k] = indict[k]
            
    def get_constraints(self):

        conlist = [
            # Try to get tower height to match desired hub height
            ['tow.height_constraint', -1e-2, 1e-2, None],
            
            # Ensure that draft is greater than 0 (spar length>0) and that less than water depth
            # Ensure that fairlead attaches to draft
            ['base.draft', 0.0, 100.0, None],
            ['aux.draft', 0.0, 100.0, None],
            ['base.draft_depth_ratio', None, 0.9, None],
            ['aux.draft_depth_ratio', None, 0.9, None],
            ['base.wave_height_freeboard_ratio', None, 1.0, None],
            ['aux.wave_height_freeboard_ratio', None, 1.0, None],
            
            #['aux.fairlead_draft_ratio', 0.0, 1.0, None],
            ['sg.base_auxiliary_spacing', 1.0, None, None],
            
            # Ensure that the radius doesn't change dramatically over a section
            ['base.manufacturability', 0.0, None, None],
            ['base.weldability', None, 0.0, None],
            ['aux.manufacturability', 0.0, None, None],
            ['aux.weldability', None, 0.0, None],
            ['tow.manufacturability', 0.0, None, None],
            ['tow.weldability', None, 0.0, None],
            
            # Ensure that the spar top matches the tower base
            ['sg.tower_transition_buffer', -1.0, 1.0, None],
            ['sg.nacelle_transition_buffer', 0.0, None, None],

            # Make sure semisub columns don't get submerged
            ['sg.auxiliary_freeboard_heel_margin', 0.0, None, None],
            
            # Ensure max mooring line tension is less than X% of MBL: 60% for intact mooring, 80% for damanged
            ['mm.axial_unity', 0.0, 1.0, None],
            
            # Ensure there is sufficient mooring line length, MAP doesn't give an error about this
            ['mm.mooring_length_max', None, 1.0, None],
            
            # API Bulletin 2U constraints
            ['base.flange_spacing_ratio', None, 1.0, None],
            ['base.stiffener_radius_ratio', None, 0.5, None],
            ['base.flange_compactness', 1.0, None, None],
            ['base.web_compactness', 1.0, None, None],
            ['base.axial_local_api', None, 1.0, None],
            ['base.axial_general_api', None, 1.0, None],
            ['base.external_local_api', None, 1.0, None],
            ['base.external_general_api', None, 1.0, None],
            
            ['aux.flange_spacing_ratio', None, 1.0, None],
            ['aux.stiffener_radius_ratio', None, 0.5, None],
            ['aux.flange_compactness', 1.0, None, None],
            ['aux.web_compactness', 1.0, None, None],
            ['aux.axial_local_api', None, 1.0, None],
            ['aux.axial_general_api', None, 1.0, None],
            ['aux.external_local_api', None, 1.0, None],
            ['aux.external_general_api', None, 1.0, None],
            
            # Pontoon tube radii
            #['load.base_connection_ratio', 0.0, None, None],
            #['load.auxiliary_connection_ratio', 0.0, None, None],
            
            # Pontoon stress safety factor
            ['load.pontoon_stress', None, 1.0, None],
            ['load.tower_stress', None, 1.0, None],
            ['load.tower_shell_buckling', None, 1.0, None],
            ['load.tower_global_buckling', None, 1.0, None],
            
            # Achieving non-zero variable ballast height means the semi can be balanced with margin as conditions change
            ['subs.variable_ballast_height_ratio', 0.0, 1.0, None],
            ['subs.variable_ballast_mass', 0.0, None, None],
            
            # Metacentric height should be positive for static stability
            ['subs.metacentric_height', 0.1, None, None],
            
            # Center of buoyancy should be above CG (difference should be positive, None],
            #['subs.buoyancy_to_gravity', 0.1, None, None],
            
            # Surge restoring force should be greater than wave-wind forces (ratio < 1, None],
            ['subs.offset_force_ratio', None, 1.0, None],
            
            # Heel angle should be less than 6deg for ordinary operation, less than 10 for extreme conditions
            ['subs.heel_moment_ratio', None, 1.0, None],

            # Wave forcing period should be different than natural periods and structural modes
            ['subs.period_margin_low', None, 1.0, None],
            ['subs.period_margin_high', 1.0, None, None],
            ['subs.modal_margin_low', None, 1.0, None],
            ['subs.modal_margin_high', 1.0, None, None]
        ]
        #raise NotImplementedError("Subclasses should implement this!")
        return conlist


    def constraint_report(self):
        passStr = 'yes'
        noStr = 'NO'
        
        #print('Status\tLow\tName\tHigh\tEq\tValue')
        conlist = self.get_constraints()
        for k in conlist:
            lowStr = ''
            highStr = ''
            eqStr = ''
            passFlag = True
            if not k[1] is None:
                lowStr = str(k[1])+' <\t'
                passFlag = passFlag and np.all(self.prob[k[0]] >= k[1])
                
            if not k[2] is None:
                highStr = '< '+str(k[2])+'\t'
                passFlag = passFlag and np.all(self.prob[k[0]] <= k[2])

            if not k[3] is None:
                highStr = '= '+str(k[3])+'\t'
                passFlag = passFlag and np.all(self.prob[k[0]] == k[3])

            conStr = passStr if passFlag else noStr
            valStr = '' if passFlag else str(self.prob[k[0]])
            print(conStr, '\t', lowStr, k[0], '\t', highStr, eqStr, '\t', valStr)

            
    def add_objective(self):
        if (len(self.prob.driver._objs) == 0):
            self.prob.driver.add_objective('total_cost', scaler=1e-9)


    def set_inputs(self):
        # Load all variables from local params dictionary
        localnames = self.params.keys()

        for ivar in localnames:
            try:
                self.prob[ivar] = self.params[ivar]
            except KeyError:
                print('Cannot set: ', ivar, '=', self.params[ivar])
                continue
            except AttributeError as e:
                print('Vector issues?: ', ivar)
                print(e)
                raise e
            except ValueError as e:
                print('Badding setting of: ', ivar)
                print(e)
                raise e

        # Check that everything got set correctly
        namesAssembly = self.prob.root._unknowns_dict.keys()
        for ivar in namesAssembly:
            if self.prob.root._unknowns_dict[ivar].has_key('_canset_') and self.prob.root._unknowns_dict[ivar]['_canset_']:
                selfvar = ivar.split('.')[-1]
                if not selfvar in localnames:
                    print('WARNING:', selfvar, 'has not been set!')
                    

        '''
        namesAssembly = self.prob.root._unknowns_dict.keys()
        for ivar in namesAssembly:
            if self.prob.root._unknowns_dict[ivar].has_key('_canset_') and self.prob.root._unknowns_dict[ivar]['_canset_']:
                selfvar = ivar.split('.')[-1]

                selfval = self.prob[selfvar]
                if ( (type(selfval) == type(0.0)) or (type(selfval) == np.float64) or (type(selfval) == np.float32) ) and selfval == 0.0:
                    print(selfvar, 'is zero! Did you set it?')
                if ( (type(selfval) == type(0)) or (type(selfval) == np.int64) or (type(selfval) == np.int32) ) and selfval == 0:
                    print(selfvar, 'is zero! Did you set it?')
                elif type(selfval) == type(np.array([])) and not np.any(selfval):
                    print(selfvar, 'is zero! Did you set it?')

                selfval = getattr(self, selfvar, None)
                if selfval is None:
                    print('Variable not found:', ivar, selfvar, self.prob[ivar])
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
                print(ivar, '=', ivalstr)

                
    def init_problem(self, optFlag=False):
        self.prob.root = self.get_assembly()
        
        if optFlag:
            if not self.optimizerSet:
                raise RuntimeError('Must set the optimizer to set the driver first')
            if not self.desvarSet:
                raise RuntimeError('Must set design variables before running optimization')
            if not self.constraintSet:
                print('Warning: no constraints set')

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
        #self.prob.check_total_derivatives()

    def run(self):
        if not self.optimizerSet:
            print('WARNING: executing once because optimizer is not set')
            self.evaluate()
        else:
            self.init_problem(optFlag=True)
            self.prob.run()
            self.store_results()
            print(self.prob.driver.get_constraints())
            #print(self.prob.driver.get_desvars())
            print(self.prob.driver.get_objectives())
        
    def evaluate(self):
        self.init_problem(optFlag=False)
        self.prob.run_once()


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
        x = y = 100 * np.linspace(-1, 1, npts)
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

        cmoor = (0,0.8,0)
        nlines = int( self.params['number_of_mooring_connections'] * self.params['mooring_lines_per_connection'] )
        for k in xrange(nlines):
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
            mlab.mesh(X, Y, Z, opacity=0.7, color=ck, figure=fig)

            if spacingVec is None: continue
            
            z = z_nodes[k] + spacingVec[k]
            while z < z_nodes[k+1]:
                rk = np.interp(z, z_nodes[k:], r_nodes[k:])
                #print(z, z_nodes[k], z_nodes[k+1], rk, r_nodes[k], r_nodes[k+1])
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

    def draw_ballast(self, fig, centerline, freeboard, h_section, r_nodes, h_perm, h_water):
        npts = 40
        th = np.linspace(0, 2*np.pi, npts)
        z_nodes = np.flipud( freeboard - np.r_[0.0, np.cumsum(np.flipud(h_section))] )

        # Permanent ballast
        z_perm = z_nodes[0] + np.linspace(0, h_perm, npts)
        r_perm = np.interp(z_perm, z_nodes, r_nodes)
        R, TH = np.meshgrid(r_perm, th)
        Z, _  = np.meshgrid(z_perm, th)
        X = R*np.cos(TH) + centerline[0]
        Y = R*np.sin(TH) + centerline[1]
        ck = np.array([122, 85, 33]) / 255.0
        ck = tuple(ck.tolist())
        mlab.mesh(X, Y, Z, color=ck, figure=fig)

        # Water ballast
        z_water = z_perm[-1] + np.linspace(0, h_water, npts)
        r_water = np.interp(z_water, z_nodes, r_nodes)
        R, TH = np.meshgrid(r_water, th)
        Z, _  = np.meshgrid(z_water, th)
        X = R*np.cos(TH) + centerline[0]
        Y = R*np.sin(TH) + centerline[1]
        ck = (0.0, 0.1, 0.8) # Dark blue
        mlab.mesh(X, Y, Z, color=ck, figure=fig)
        
        
    def set_figure(self, fig, fname=None):
        #ax.set_aspect('equal')
        #set_axes_equal(ax)
        #ax.autoscale_view(tight=True)
        #ax.set_xlim([-125, 125])
        #ax.set_ylim([-125, 125])
        #ax.set_zlim([-220, 30])
        #plt.axis('off')
        #plt.show()
        #mlab.move([-517.16728532, -87.0711504, 5.60826224], [1.35691603e+01, -2.84217094e-14, -1.06547500e+02])
        #mlab.view(-170.68320804213343, 78.220729198686854, 549.40101471336777, [1.35691603e+01,  0.0, -1.06547500e+02])
        if not fname is None: mlab.savefig(fname, figure=fig)
        mlab.show(stop=True)

        
    
