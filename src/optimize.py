from openmdao.api import Problem, ScipyOptimizer, pyOptSparseDriver
from sparAssembly import SparAssembly
from sparGeometry import NSECTIONS
import numpy as np
import time

def vecOption(x, in1s):
    myones = in1s if type(in1s) == type(np.array([])) else np.ones((in1s,))
    return (x*myones) if type(x)==type(0.0) or len(x) == 1 else x

def init_problem():
    prob = Problem()
    prob.root = SparAssembly()
    return prob

def solve_spar(prob, params):
    # Note this command must be done after the constraints, design variables, and objective have been set,
    # but before the initial conditions are specified (unless we use the default initial conditions )
    # After setting the intial conditions, running setup() again will revert them back to default values
    prob.setup()

    # INPUT PARAMETER SETTING
    # Parameters heading into Spar Geometry first
    prob['water_depth.x']                = params['water_depth']
    prob['freeboard.x']                  = params['freeboard']
    prob['fairlead.x']                   = params['fairlead']
    prob['fairlead_offset_from_shell.x'] = params['fairlead_offset_from_shell']
    if params.has_key('spar_length'):
        prob['section_height.x']         = vecOption(params['spar_length']/NSECTIONS, NSECTIONS)
    elif params.has_key('section_height'):
        prob['section_height.x']         = params['section_height']
    else:
        raise ValueError('Need to include spar_length or section_height')

    prob['outer_radius.x']               = vecOption(params['outer_radius'], NSECTIONS+1)
    prob['wall_thickness.x']             = vecOption(params['wall_thickness'], NSECTIONS+1)

    # Parameters heading into Turbine first
    prob['rna_mass.x']                = params['rna_mass']
    prob['rna_center_of_gravity.x']   = params['rna_center_of_gravity']
    prob['rna_center_of_gravity_x.x'] = params['rna_center_of_gravity_x']
    prob['rna_wind_force.x']          = params['rna_wind_force']
    prob['tower_mass.x']              = params['tower_mass']
    prob['tower_center_of_gravity.x'] = params['tower_center_of_gravity']
    prob['tower_wind_force.x']        = params['tower_wind_force']
    
    # Parameters heading into MAP Mooring second
    prob['water_density.x']           = params['water_density']
    prob['scope_ratio.x']             = params['scope_ratio']
    prob['anchor_radius.x']           = params['anchor_radius']
    prob['mooring_diameter.x']        = params['mooring_diameter']
    prob['number_of_mooring_lines.x'] = params['number_of_mooring_lines']
    prob['mooring_type.x']            = params['mooring_type']
    prob['anchor_type.x']             = params['anchor_type']
    prob['max_offset.x']              = 0.1*params['water_depth'] # Assumption!
    prob['mooring_cost_rate.x']       = params['mooring_cost_rate']

    # Parameters heading into Cylinder first
    prob['air_density.x']               = params['air_density']
    prob['air_viscosity.x']             = params['air_viscosity']
    prob['water_viscosity.x']           = params['water_viscosity']
    prob['wave_height.x']               = params['wave_height']
    prob['wave_period.x']               = params['wave_period']
    prob['wind_reference_speed.x']      = params['wind_reference_speed']
    prob['wind_reference_height.x']     = params['wind_reference_height']
    prob['alpha.x']                     = params['alpha']
    prob['morison_mass_coefficient.x']  = params['morison_mass_coefficient']
    prob['material_density.x']          = params['material_density']
    prob['E.x']                         = params['E']
    prob['nu.x']                        = params['nu']
    prob['yield_stress.x']              = params['yield_stress']
    prob['permanent_ballast_density.x'] = params['permanent_ballast_density']
    
    prob['stiffener_web_height.x']       = vecOption(params['stiffener_web_height'], NSECTIONS)
    prob['stiffener_web_thickness.x']    = vecOption(params['stiffener_web_thickness'], NSECTIONS)
    prob['stiffener_flange_width.x']     = vecOption(params['stiffener_flange_width'], NSECTIONS)
    prob['stiffener_flange_thickness.x'] = vecOption(params['stiffener_flange_thickness'], NSECTIONS)
    prob['stiffener_spacing.x']          = vecOption(params['stiffener_spacing'], NSECTIONS)
    
    prob['bulkhead_nodes.x']             = [False] * (NSECTIONS+1)
    prob['bulkhead_nodes.x'][0]          = True
    prob['bulkhead_nodes.x'][1]          = True
    prob['permanent_ballast_height.x']   = params['permanent_ballast_height']
    prob['bulkhead_mass_factor.x']     = params['bulkhead_mass_factor']
    prob['ring_mass_factor.x']         = params['ring_mass_factor']
    prob['shell_mass_factor.x']        = params['shell_mass_factor']
    prob['spar_mass_factor.x']         = params['spar_mass_factor']
    prob['outfitting_mass_fraction.x'] = params['outfitting_mass_fraction']
    prob['ballast_cost_rate.x']        = params['ballast_cost_rate']
    prob['tapered_col_cost_rate.x']    = params['tapered_col_cost_rate']
    prob['outfitting_cost_rate.x']     = params['outfitting_cost_rate']
    
    # Execute the optimization
    prob.check_setup()
    prob.pre_run_check()
    #prob.check_total_derivatives()
    return prob
    
def optimize_spar(prob):
    # Establish the optimization driver, then set design variables and constraints
    #prob.driver = ScipyOptimizer()
    #prob.driver.options['optimizer'] = 'COBYLA'
    #prob.driver.options['optimizer'] = 'SLSQP'

    prob.driver = pyOptSparseDriver()
    # Working
    #prob.driver.options['optimizer'] = 'CONMIN' # Takes too long and gets stuck
    prob.driver.options['optimizer'] = 'PSQP'
    
    # Jumps to infeasible values
    #prob.driver.options['optimizer'] = 'ALPSO'
    #prob.driver.options['optimizer'] = 'NSGA2'
    #prob.driver.options['optimizer'] = 'SLSQP'

    # Don't import correctly
    #prob.driver.options['optimizer'] = 'FSQP'
    #prob.driver.options['optimizer'] = 'NLPY_AUGLAG'

    if prob.driver.options['optimizer'] == 'CONMIN':
        prob.driver.opt_settings['ITMAX'] = 1000
    elif prob.driver.options['optimizer'] in ['COBYLA','SLSQP']:
        prob.driver.options['tol'] = 1e-6
        prob.driver.options['maxiter'] = 100000
        
    # Make a neat list of design variables, lower bound, upper bound, scalar
    desvarList = [('freeboard.x',0.0, 50.0, 1.0),
                  ('fairlead.x',0.0, 100.0, 1.0),
                  ('fairlead_offset_from_shell.x',0.0, 5.0, 1e2),
                  ('section_height.x',1e-1, 100.0, 1e1),
                  ('outer_radius.x',1.1, 25.0, 10.0),
                  ('wall_thickness.x',5e-3, 1.0, 1e3),
                  ('scope_ratio.x', 1.0, 5.0, 1.0),
                  ('anchor_radius.x', 1.0, 1e3, 1e-2),
                  ('mooring_diameter.x', 0.05, 1.0, 1e1),
                  ('stiffener_web_height.x', 1e-2, 1.0, 1e2),
                  ('stiffener_web_thickness.x', 1e-3, 5e-1, 1e2),
                  ('stiffener_flange_width.x', 1e-2, 5.0, 1e2),
                  ('stiffener_flange_thickness.x', 1e-3, 5e-1, 1e2),
                  ('stiffener_spacing.x', 1e-1, 1e2, 1e1),
                  ('permanent_ballast_height.x', 1e-1, 50.0, 1.0)]

    # TODO: Integer and Boolean design variables
    #prob.driver.add_desvar('number_of_mooring_lines.x', lower=1)
    #prob.driver.add_desvar('mooring_type.x')
    #prob.driver.add_desvar('anchor_type.x')
    #prob.driver.add_desvar('bulkhead_nodes.x')

    if prob.driver.options['optimizer'] in ['CONMIN','PSQP','ALPSO','NSGA2','SLSQP']:
        for ivar in desvarList:
            prob.driver.add_desvar(ivar[0], lower=ivar[1], upper=ivar[2])
    else:
        for ivar in desvarList:
            iscale=ivar[3]
            prob.driver.add_desvar(ivar[0], lower=iscale*ivar[1], upper=iscale*ivar[2], scaler=iscale)

    # CONSTRAINTS
    # These are mostly the outputs that were not connected to another model
    
    # Ensure that draft is greater than 0 (spar length>0) and that less than water depth
    # Ensure that fairlead attaches to draft
    prob.driver.add_constraint('sg.draft_depth_ratio',lower=0.0, upper=0.75)
    prob.driver.add_constraint('sg.fairlead_draft_ratio',lower=0.0, upper=1.0)

    # Ensure that the radius doesn't change dramatically over a section
    prob.driver.add_constraint('sg.taper_ratio',upper=0.1)

    # Ensure max mooring line tension is less than X% of MBL: 60% for intact mooring, 80% for damanged
    prob.driver.add_constraint('mm.safety_factor',lower=0.0, upper=0.8)

    # Ensure there is sufficient mooring line length, MAP doesn't give an error about this
    prob.driver.add_constraint('mm.mooring_length_min',lower=1.0)
    prob.driver.add_constraint('mm.mooring_length_max',upper=1.0)
    
    # API Bulletin 2U constraints
    prob.driver.add_constraint('cyl.flange_spacing_ratio', upper=0.5)
    prob.driver.add_constraint('cyl.web_radius_ratio', upper=0.5)
    prob.driver.add_constraint('cyl.flange_compactness', lower=1.0)
    prob.driver.add_constraint('cyl.web_compactness', lower=1.0)
    prob.driver.add_constraint('cyl.axial_local_unity', upper=1.0)
    prob.driver.add_constraint('cyl.axial_general_unity', upper=1.0)
    prob.driver.add_constraint('cyl.external_local_unity', upper=1.0)
    prob.driver.add_constraint('cyl.external_general_unity', upper=1.0)

    # Achieving non-zero variable ballast height means the spar can be balanced with margin as conditions change
    prob.driver.add_constraint('sp.variable_ballast_height', lower=2.0, upper=100.0)
    prob.driver.add_constraint('sp.variable_ballast_mass', lower=0.0)

    # Metacentric height should be positive for static stability
    prob.driver.add_constraint('sp.metacentric_height', lower=0.1)

    # Center of buoyancy should be above CG (difference should be positive)
    prob.driver.add_constraint('sp.static_stability', lower=0.1)

    # Surge restoring force should be greater than wave-wind forces (ratio < 1)
    prob.driver.add_constraint('sp.offset_force_ratio',lower=0.0, upper=1.0)

    # Heel angle should be less than 6deg for ordinary operation, less than 10 for extreme conditions
    prob.driver.add_constraint('sp.heel_angle',lower=0.0, upper=10.0)

    
    # OBJECTIVE FUNCTION: Minimize total cost!
    prob.driver.add_objective('sp.total_cost', scaler=1e-9)
    return prob


def get_static_params():
    params = {}
    params['water_depth'] = 218.0
    params['number_of_mooring_lines'] = 3
    params['mooring_type'] = 'chain'
    params['anchor_type'] = 'pile'
    params['mooring_cost_rate'] = 1.1
    params['air_density'] = 1.198
    params['air_viscosity'] = 1.81e-5
    params['water_density'] = 1025.0
    params['water_viscosity'] = 8.9e-4
    params['wave_height'] = 10.8
    params['wave_period'] = 9.8
    params['wind_reference_speed'] = 11.0
    params['wind_reference_height'] = 119.0
    params['alpha'] = 0.11
    params['morison_mass_coefficient'] = 2.0
    params['material_density'] = 7850.0
    params['E'] = 200e9
    params['nu'] = 0.3
    params['yield_stress'] = 3.45e8
    params['permanent_ballast_density'] = 4492.0
    params['bulkhead_mass_factor'] = 1.0
    params['ring_mass_factor'] = 1.0
    params['shell_mass_factor'] = 1.0
    params['spar_mass_factor'] = 1.05
    params['outfitting_mass_fraction'] = 0.06
    params['ballast_cost_rate'] = 100.0
    params['tapered_col_cost_rate'] = 4720.0
    params['outfitting_cost_rate'] = 6980.0
    params['rna_mass']= 180e3
    params['rna_center_of_gravity'] = 3.5 + 80.0
    params['rna_center_of_gravity_x'] = 5.75
    params['tower_mass'] = 180e3
    params['tower_center_of_gravity'] = 35.0
    params['rna_wind_force'] = 820818.0
    params['tower_wind_force'] = 33125.0
    return params


def example_spar():
    #tt = time.time()
    params = get_static_params()
    params['freeboard'] = 5.0
    params['fairlead'] = 7.57
    params['spar_length'] = 95.0
    params['outer_radius'] = 7.0
    params['wall_thickness'] = 0.05
    params['fairlead_offset_from_shell'] = 0.05
    params['scope_ratio'] = 2.41
    params['anchor_radius'] = 450.0
    params['mooring_diameter'] = 0.19
    params['number_of_mooring_lines'] = 3
    params['permanent_ballast_height'] = 10.0
    params['stiffener_web_height']= 0.1
    params['stiffener_web_thickness'] = 0.04
    params['stiffener_flange_width'] = 0.1
    params['stiffener_flange_thickness'] = 0.02
    params['stiffener_spacing'] = 0.4

    prob = init_problem()
    prob = optimize_spar(prob)
    prob = solve_spar(prob, params)
    prob.run()
    #prob.run_once()
    print prob.driver.get_constraints()
    print prob.driver.get_desvars()
    print prob.driver.get_objectives()
    
def psqp_optimal():
    #OrderedDict([('sp.total_cost', array([ 3.68125491]))])
    params = get_static_params()
    params['freeboard'] = 0.0
    params['fairlead'] = 28.98778705
    params['fairlead_offset_from_shell'] = 4.15123014
    params['section_height'] = np.array([ 49.01203711,  52.29873288,  33.77996355,   1.98476181,  26.42465958])
    params['outer_radius'] = np.array([ 3.76031187,  3.78714959,  4.16586455,  3.74927809,  3.37435028, 3.03691526])
    params['wall_thickness'] = np.array([ 0.00661516,  0.01453792,  0.005     ,  0.0097314 ,  0.0059745 , 0.005     ])
    params['scope_ratio'] = 2.60758833
    params['anchor_radius'] = 448.0246413
    params['mooring_diameter'] = 0.0724271
    params['stiffener_web_height']= np.array([ 0.12411858,  0.12090302,  0.11733809,  0.07447426,  0.09330525])
    params['stiffener_web_thickness'] = np.array([ 0.00515503,  0.00502148,  0.00487342,  0.00309315,  0.00387526 ])
    params['stiffener_flange_width'] = np.array([ 0.01      ,  0.01      ,  0.01      ,  0.04525972,  0.01 ])
    params['stiffener_flange_thickness'] = np.array([ 0.27650071,  0.21402134,  0.13546971,  0.00276745,  0.03429803])
    params['stiffener_spacing'] = np.array([ 0.21849442,  0.26591885,  0.2661029 ,  0.32705785,  0.20720817])
    params['permanent_ballast_height'] = 30.06223745

    prob = init_problem()
    prob = optimize_spar(prob)
    prob = solve_spar(prob, params)
    prob.run_once()
    print prob.driver.get_constraints()
    print prob.driver.get_desvars()
    print prob.driver.get_objectives()
    '''
OrderedDict([('sg.draft_depth_ratio', array([ 0.75])), ('sg.fairlead_draft_ratio', array([ 0.17729533])), ('sg.taper_ratio', array([ 0.0071371,  0.1      ,  0.1      ,  0.1      ,  0.1      ])), ('mm.safety_factor', array([ 0.79959284])), ('mm.mooring_length_min', array([ 1.02897572])), ('mm.mooring_length_max', array([ 0.77368529])), ('cyl.flange_spacing_ratio', array([ 0.04576776,  0.03760546,  0.03757945,  0.13838446,  0.04826065])), ('cyl.web_radius_ratio', array([ 0.03289015,  0.03040433,  0.02964902,  0.02090908,  0.02910666])), ('cyl.flange_compactness', array([ 499.30160594,  386.47712046,  244.62955708,    1.10416659,
         61.93496328])), ('cyl.web_compactness', array([ 1.        ,  1.        ,  1.        ,  0.99999999,  1.        ])), ('cyl.axial_local_unity', array([ 0.99999959,  0.99999964,  0.99998764,  0.9949145 ,  0.99660552])), ('cyl.axial_general_unity', array([ 0.99999958,  0.99982145,  0.91031899,  1.2161923 ,  0.93928689])), ('cyl.external_local_unity', array([ 0.85868796,  0.89067778,  0.95522241,  0.98490783,  0.99660552])), ('cyl.external_general_unity', array([ 0.92690846,  0.99999937,  1.00243   ,  1.27160385,  1.00003484])), ('sp.variable_ballast_height', array([ 9.45559567])), ('sp.variable_ballast_mass', array([ 432098.15110704])), ('sp.metacentric_height', array([ 46.5396792])), ('sp.static_stability', array([ 46.53063839])), ('sp.offset_force_ratio', array([ 1.00056272])), ('sp.heel_angle', array([ 10.0002829]))])
    '''


def conmin_optimal():
    #OrderedDict([('sp.total_cost', array([ 8.15839897]))])
    params = get_static_params()
    params['freeboard'] = 5.0
    params['fairlead'] = 7.57
    params['fairlead_offset_from_shell'] = 0.05
    params['section_height'] = np.array([ 18.99987492,  18.9998873 ,  18.99990693,  18.99990914,  18.99990425])
    params['outer_radius'] = np.array([ 6.99962345,  6.99955813,  6.99973629,  6.99978022,  6.99976883, 6.99988])
    params['wall_thickness'] = np.array([ 0.03712666,  0.02787312,  0.02712097,  0.02206188,  0.02157211, 0.03579269])
    params['scope_ratio'] = 2.40997737
    params['anchor_radius'] = 450.0
    params['mooring_diameter'] = 0.1909802
    params['stiffener_web_height']= np.array([ 0.10557588,  0.10316776,  0.09795284,  0.09743845,  0.09743956])
    params['stiffener_web_thickness'] = np.array([ 0.03599046,  0.03502903,  0.03323707,  0.03302298,  0.0330262 ])
    params['stiffener_flange_width'] = np.array([ 0.10066915,  0.10029873,  0.09894232,  0.09882406,  0.0988245 ])
    params['stiffener_flange_thickness'] = np.array([ 0.02739561,  0.02327079,  0.01406197,  0.01304515,  0.01304842])
    params['stiffener_spacing'] = np.array([ 0.40020418,  0.40036638,  0.4008825 ,  0.4009331 ,  0.40093272])
    params['permanent_ballast_height'] = 10.0

    prob = init_problem()
    prob = optimize_spar(prob)
    prob = solve_spar(prob, params)
    prob.run_once()
    print prob.driver.get_constraints()
    print prob.driver.get_desvars()
    print prob.driver.get_objectives()
    '''
OrderedDict([('sg.draft_depth_ratio', array([ 0.41284162])), ('mm.safety_factor', array([ 0.7999935])), ('mm.mooring_length_min', array([ 1.03413212])), ('mm.mooring_length_max', array([ 0.76788083])), ('sp.flange_spacing_ratio', array([ 0.25154447,  0.25051738,  0.24681128,  0.24648516,  0.2464865 ])), ('sp.web_radius_ratio', array([ 0.01508315,  0.01473899,  0.01399375,  0.01392023,  0.01392029])), ('sp.flange_compactness', array([ 4.91418164,  4.18969521,  2.56643842,  2.38370799,  2.38429482])), ('sp.web_compactness', array([ 8.20782661,  8.17503344,  8.16979471,  8.16002162,  8.160726  ])), ('sp.axial_local_unity', array([ 0.51507608,  0.46336477,  0.38573459,  0.25387818,  0.06900377])), ('sp.axial_general_unity', array([ 0.9982523 ,  0.95677559,  0.9846294 ,  0.65050248,  0.19042545])), ('sp.external_local_unity', array([ 0.43199964,  0.39057533,  0.32676712,  0.21654406,  0.05750315])), ('sp.external_general_unity', array([ 1.00397317,  0.96693407,  0.99605626,  0.66093149,  0.1917524 ])), ('sp.metacentric_height', array([ 20.55260275])), ('sp.static_stability', array([ 20.41649121])), ('sp.variable_ballast_height', array([ 28.52903672])), ('sp.variable_ballast_mass', array([ 4464694.51334896])), ('sp.offset_force_ratio', array([ 0.98529446])), ('sp.heel_angle', array([ 2.39612487]))])
    '''


def cobyla_optimal():
    #OrderedDict([('sp.total_cost', array([ 6.83851908]))])
    params = get_static_params()
    params['freeboard'] = 7.56789854
    params['fairlead'] = 9.41184644
    params['fairlead_offset_from_shell'] = 0.0471558864
    params['section_height'] = np.array([ 18.708914991,  18.784270853,  18.799716693,  18.648435942, 18.711380637])
    params['outer_radius'] = np.array([ 5.764219519,  5.657993694,  6.159558061,  6.125155506, 6.293851894,  6.606570305])
    params['wall_thickness'] = np.array([ 0.043758918  ,  0.03934623132,  0.04101795034,  0.03947006871, 0.03855182803,  0.04268526778])
    params['scope_ratio'] = 2.39202552
    params['anchor_radius'] = 442.036507
    params['mooring_diameter'] = 0.153629334
    params['stiffener_web_height']= np.array([ 0.1433863028,  0.1192863504,  0.1102913546,   0.0959098443,   0.0760210847])
    params['stiffener_web_thickness'] = np.array([ 0.0059552804,  0.0049543342,  0.004580744 ,  0.003983435 ,  0.0031573928 ])
    params['stiffener_flange_width'] = np.array([ 0.0924192057,  0.0977347306,  0.0800589589,  0.0797488027,  0.0861943184 ])
    params['stiffener_flange_thickness'] = np.array([ 0.02739561,  0.02327079,  0.01406197,  0.01304515,  0.01304842])
    params['stiffener_spacing'] = np.array([ 0.937472777,   0.913804583,   0.975992681,   0.940785141,  1.077950861])
    params['permanent_ballast_height'] = 2.1531719

    prob = init_problem()
    prob = optimize_spar(prob)
    prob = solve_spar(prob, params)
    prob.run_once()
    print prob.driver.get_constraints()
    print prob.driver.get_desvars()
    print prob.driver.get_objectives()
    '''
OrderedDict([('sg.draft_depth_ratio', array([ 0.3948845])), ('mm.safety_factor', array([ 0.79998459])), ('mm.mooring_length_min', array([ 1.03296286])), ('mm.mooring_length_max', array([ 0.76687562])), ('sp.flange_spacing_ratio', array([ 0.09858335,  0.10695364,  0.08202824,  0.08476835,  0.07996127])), ('sp.web_radius_ratio', array([ 0.02510657,  0.020188  ,  0.01795587,  0.01544565,  0.01178583])), ('sp.flange_compactness', array([ 4.34435796,  3.92278448,  4.35797943,  2.16402123,  1.        ])), ('sp.web_compactness', array([ 1.,  1.,  1.,  1.,  1.])), ('sp.axial_local_unity', array([ 0.47642979,  0.38788415,  0.28705696,  0.17106678,  0.05175507])), ('sp.axial_general_unity', array([ 0.97397846,  0.97695689,  0.98037179,  0.98931662,  0.5567153 ])), ('sp.external_local_unity', array([ 0.41057698,  0.33470586,  0.24904213,  0.14822411,  0.04496548])), ('sp.external_general_unity', array([ 1.        ,  1.        ,  1.        ,  1.        ,  0.56070475])), ('sp.metacentric_height', array([ 6.12375997])), ('sp.static_stability', array([ 5.98379185])), ('sp.variable_ballast_height', array([ 60.25812533])), ('sp.variable_ballast_mass', array([ 6785190.91932651])), ('sp.offset_force_ratio', array([ 1.0000217])), ('sp.heel_angle', array([ 9.99999845]))])
    '''
        
if __name__ == '__main__':
    example_spar()
