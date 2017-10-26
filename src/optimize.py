from openmdao.api import Problem, ScipyOptimizer
from sparAssembly import SparAssembly
from mooringAssembly import MooringAssembly
from sparGeometry import NSECTIONS
import numpy as np
import time

def optimize_mooring(params):

    # Setup the problem
    prob = Problem()
    prob.root = MooringAssembly()
    
    # Establish the optimization driver, then set design variables and constraints
    myopt={}
    prob.driver = ScipyOptimizer()
    #prob.driver.options['optimizer'] = 'SLSQP'
    prob.driver.options['optimizer'] = 'COBYLA'
    prob.driver.options['tol'] = 1e-6
    prob.driver.options['maxiter'] = 2000
    
    # DESIGN VARIABLES
    #prob.driver.add_desvar('freeboard.x',lower=0.0, upper=20.0)
    #prob.driver.add_desvar('fairlead.x',lower=0.0, upper=20.0)
    #prob.driver.add_desvar('fairlead_offset_from_shell.x',lower=0.0, upper=1.0)
    #prob.driver.add_desvar('section_height.x',lower=1e-2, upper=30.0)
    #prob.driver.add_desvar('outer_radius.x',lower=1.0, upper=20.0)
    #prob.driver.add_desvar('wall_thickness.x',lower=1e-2, upper=1.0)

    prob.driver.add_desvar('scope_ratio.x', lower=1.0, upper=5.0) #>1 means longer than water depth
    prob.driver.add_desvar('anchor_radius.x', lower=0.0, scaler=1e-2)
    prob.driver.add_desvar('mooring_diameter.x', lower=0.5, scaler=10)
    #prob.driver.add_desvar('mooring_diameter.x', lower=0.07, upper=0.5)
    # TODO: Integer design variables
    #prob.driver.add_desvar('number_of_mooring_lines.x', lower=1)
    #prob.driver.add_desvar('mooring_type.x')
    #prob.driver.add_desvar('anchor_type.x')

    
    # CONSTRAINTS
    # Ensure there is sufficient mooring line length, MAP doesn't give an error about this
    prob.driver.add_constraint('mm.mooring_length_min',lower=1.0)
    prob.driver.add_constraint('mm.mooring_length_max',upper=1.0)

    # Ensure max mooring line tension is less than X% of MBL: 60% for intact mooring, 80% for damanged
    prob.driver.add_constraint('mm.safety_factor',upper=0.8)

    # Mock restoring force need for substructure
    #prob.driver.add_constraint('mm.max_offset_restoring_force',lower=200e3)
    prob.driver.add_constraint('mm.max_offset_restoring_force',lower=2, scaler=1e-5)
    
    # OBJECTIVE FUNCTION: Minimize total cost!
    prob.driver.add_objective('mm.mooring_cost', scaler=1e-6)

    # Establish the problem
    # Note this command must be done after the constraints, design variables, and objective have been set,
    # but before the initial conditions are specified (unless we use the default initial conditions )
    # After setting the intial conditions, running setup() again will revert them back to default values
    prob.setup()

    # INITIAL CONDITIONS
    nodeOnes  = np.ones((NSECTIONS+1,))
    secOnes   = np.ones((NSECTIONS,))

    # INPUT PARAMETER SETTING
    # Parameters heading into Spar Geometry first
    prob['water_depth.x']                = params['water_depth']
    prob['freeboard.x']                  = params['freeboard']
    prob['fairlead.x']                   = params['fairlead']
    prob['section_height.x']             = (params['spar_length']/NSECTIONS) * secOnes
    prob['outer_radius.x']               = params['outer_radius'] * nodeOnes
    prob['wall_thickness.x']             = params['wall_thickness'] * nodeOnes
    prob['fairlead_offset_from_shell.x'] = params['fairlead_offset_from_shell']

    # Parameters heading into MAP Mooring second
    prob['scope_ratio.x']             = params['scope_ratio']
    prob['anchor_radius.x']           = params['anchor_radius']
    prob['mooring_diameter.x']        = params['mooring_diameter']
    prob['number_of_mooring_lines.x'] = params['number_of_mooring_lines']
    prob['mooring_type.x']            = params['mooring_type']
    prob['anchor_type.x']             = params['anchor_type']
    prob['max_offset.x']              = 0.1*params['water_depth'] # Assumption!
    prob['mooring_cost_rate.x']       = params['mooring_cost_rate']

    #Parameters heading into Spar
    prob['water_density.x']             = params['water_density']
    
    # Execute the optimization
    prob.check_setup()
    prob.pre_run_check()
    prob.check_total_derivatives()
    
    prob.run()
    #prob.run_once()

    return prob


def example_mooring():
    #tt = time.time()

    params = {}
    params['water_depth'] = 218.0
    params['freeboard'] = 13.0
    params['fairlead'] = 13.0
    params['spar_length'] = 80.0
    params['outer_radius'] = 3.0
    params['wall_thickness'] = 0.1
    params['fairlead_offset_from_shell'] = 0.5
    params['scope_ratio'] = 1.5
    params['anchor_radius'] = 150.0
    params['mooring_diameter'] = 0.2
    params['number_of_mooring_lines'] = 3
    params['mooring_type'] = 'chain'
    params['anchor_type'] = 'pile'
    params['mooring_cost_rate'] = 1.1
    params['water_density'] = 1025.0
    
    prob = optimize_mooring(params)
    print prob.driver.get_constraints()
    print prob.driver.get_desvars()
    print prob.driver.get_objectives()


        
def optimize_spar(params):

    # Setup the problem
    prob = Problem()
    prob.root = SparAssembly()
    
    # Establish the optimization driver, then set design variables and constraints
    myopt={}
    prob.driver = ScipyOptimizer()
    #prob.driver.options['optimizer'] = 'SLSQP'
    prob.driver.options['optimizer'] = 'COBYLA'
    prob.driver.options['tol'] = 1e-4
    prob.driver.options['maxiter'] = 100
    
    # DESIGN VARIABLES
    prob.driver.add_desvar('freeboard.x',lower=0.0)
    prob.driver.add_desvar('fairlead.x',lower=0.0)
    prob.driver.add_desvar('fairlead_offset_from_shell.x',lower=0.0, scaler=20)
    prob.driver.add_desvar('section_height.x',lower=1e-1)
    prob.driver.add_desvar('outer_radius.x',lower=0.5)
    prob.driver.add_desvar('wall_thickness.x',lower=1e-1, scaler=100)

    prob.driver.add_desvar('scope_ratio.x', lower=1.0, upper=5.0) #>1 means longer than water depth
    prob.driver.add_desvar('anchor_radius.x', lower=0.0, scaler=1e-2)
    prob.driver.add_desvar('mooring_diameter.x', lower=0.5, scaler=10)
    # TODO: Integer design variables
    #prob.driver.add_desvar('number_of_mooring_lines.x', lower=1)
    #prob.driver.add_desvar('mooring_type.x')
    #prob.driver.add_desvar('anchor_type.x')

    prob.driver.add_desvar('stiffener_web_height.x', lower=1, scaler=1e3)
    prob.driver.add_desvar('stiffener_web_thickness.x', lower=1, scaler=1e3)
    prob.driver.add_desvar('stiffener_flange_width.x', lower=1, scaler=1e3)
    prob.driver.add_desvar('stiffener_flange_thickness.x', lower=1, scaler=1e3)
    prob.driver.add_desvar('stiffener_spacing.x', lower=1, scaler=1e2)
    prob.driver.add_desvar('permanent_ballast_height.x', lower=0.0)
    # TODO: Boolean design variables
    #prob.driver.add_desvar('bulkhead_nodes.x')

    
    # CONSTRAINTS
    # Ensure that draft is greater than 0 (spar length>0) and that less than water depth
    prob.driver.add_constraint('sg.draft_depth_ratio',lower=0.0, upper=1.0)

    # Ensure max mooring line tension is less than X% of MBL: 60% for intact mooring, 80% for damanged
    prob.driver.add_constraint('mm.safety_factor',upper=0.8)

    # Ensure there is sufficient mooring line length, MAP doesn't give an error about this
    prob.driver.add_constraint('mm.mooring_length_min',lower=1.0)
    prob.driver.add_constraint('mm.mooring_length_max',upper=1.0)
    
    # API Bulletin 2U constraints
    prob.driver.add_constraint('sp.flange_compactness', lower=1.0)
    prob.driver.add_constraint('sp.web_compactness', lower=1.0)
    prob.driver.add_constraint('sp.axial_local_unity', upper=1.0)
    prob.driver.add_constraint('sp.axial_general_unity', upper=1.0)
    prob.driver.add_constraint('sp.external_local_unity', upper=1.0)
    prob.driver.add_constraint('sp.external_general_unity', upper=1.0)

    # Metacentric height should be positive for static stability
    prob.driver.add_constraint('sp.metacentric_height', lower=0.0)

    # Center of bouyancy should be above CG (difference should be positive)
    prob.driver.add_constraint('sp.static_stability', lower=0.0)

    # Achieving non-zero variable ballast height means the spar can be balanced
    prob.driver.add_constraint('sp.variable_ballast_height', upper=params['spar_length'])
    prob.driver.add_constraint('sp.variable_ballast_mass', lower=0.0)

    # Surge restoring force should be greater than wave-wind forces (ratio < 1)
    prob.driver.add_constraint('sp.offset_force_ratio', upper=1.0)

    # Heel angle should be less than 6deg for ordinary operation, less than 10 for extreme conditions
    prob.driver.add_constraint('sp.heel_angle', upper=10.0)

    
    # OBJECTIVE FUNCTION: Minimize total cost!
    prob.driver.add_objective('sp.total_cost', scaler=1e-10)

    # Establish the problem
    # Note this command must be done after the constraints, design variables, and objective have been set,
    # but before the initial conditions are specified (unless we use the default initial conditions )
    # After setting the intial conditions, running setup() again will revert them back to default values
    prob.setup()

    # INITIAL CONDITIONS
    nodeOnes  = np.ones((NSECTIONS+1,))
    secOnes   = np.ones((NSECTIONS,))

    # INPUT PARAMETER SETTING
    # Parameters heading into Spar Geometry first
    prob['water_depth.x']                = params['water_depth']
    prob['freeboard.x']                  = params['freeboard']
    prob['fairlead.x']                   = params['fairlead']
    prob['section_height.x']             = (params['spar_length']/NSECTIONS) * secOnes
    prob['outer_radius.x']               = params['outer_radius'] * nodeOnes
    prob['wall_thickness.x']             = params['wall_thickness'] * nodeOnes
    prob['fairlead_offset_from_shell.x'] = params['fairlead_offset_from_shell']

    # Parameters heading into MAP Mooring second
    prob['scope_ratio.x']             = params['scope_ratio']
    prob['anchor_radius.x']           = params['anchor_radius']
    prob['mooring_diameter.x']        = params['mooring_diameter']
    prob['number_of_mooring_lines.x'] = params['number_of_mooring_lines']
    prob['mooring_type.x']            = params['mooring_type']
    prob['anchor_type.x']             = params['anchor_type']
    prob['max_offset.x']              = 0.1*params['water_depth'] # Assumption!
    prob['mooring_cost_rate.x']       = params['mooring_cost_rate']

    #Parameters heading into Spar
    prob['air_density.x']               = params['air_density']
    prob['air_viscosity.x']             = params['air_viscosity']
    prob['water_density.x']             = params['water_density']
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
    
    prob['stiffener_web_height.x']       = params['stiffener_web_height'] * secOnes
    prob['stiffener_web_thickness.x']    = params['stiffener_web_thickness'] * secOnes
    prob['stiffener_flange_width.x']     = params['stiffener_flange_width'] * secOnes
    prob['stiffener_flange_thickness.x'] = params['stiffener_flange_thickness'] * secOnes
    prob['stiffener_spacing.x']          = params['stiffener_spacing'] * secOnes
    prob['bulkhead_nodes.x']             = [False] * (NSECTIONS+1)
    prob['bulkhead_nodes.x'][0]          = True
    prob['bulkhead_nodes.x'][1]          = True
    prob['permanent_ballast_height.x']   = params['permanent_ballast_height']
    
    prob['bulkhead_mass_factor.x']     = params['bulkhead_mass_factor']
    prob['ring_mass_factor.x']         = params['ring_mass_factor']
    prob['spar_mass_factor.x']         = params['spar_mass_factor']
    prob['shell_mass_factor.x']        = params['shell_mass_factor']
    prob['outfitting_mass_fraction.x'] = params['outfitting_mass_fraction']
    prob['ballast_cost_rate.x']        = params['ballast_cost_rate']
    prob['tapered_col_cost_rate.x']    = params['tapered_col_cost_rate']
    prob['outfitting_cost_rate.x']     = params['outfitting_cost_rate']
    
    prob['rna_mass.x']                = params['rna_mass']
    prob['rna_center_of_gravity.x']   = params['rna_center_of_gravity']
    prob['rna_center_of_gravity_x.x'] = params['rna_center_of_gravity_x']
    prob['rna_wind_force.x']          = params['rna_wind_force']
    prob['tower_mass.x']              = params['tower_mass']
    prob['tower_center_of_gravity.x'] = params['tower_center_of_gravity']
    prob['tower_wind_force.x']        = params['tower_wind_force']

    
    # Execute the optimization
    prob.check_setup()
    prob.pre_run_check()
    #prob.check_total_derivatives()
    
    prob.run()
    #prob.run_once()

    return prob


def example_spar():
    #tt = time.time()

    params = {}
    params['water_depth'] = 218.0
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
    params['permanent_ballast_height'] = 10.0
    params['permanent_ballast_density'] = 4492.0
    params['stiffener_web_height']= 0.1
    params['stiffener_web_thickness'] = 0.04
    params['stiffener_flange_width'] = 0.1
    params['stiffener_flange_thickness'] = 0.02
    params['stiffener_spacing'] = 0.4
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

    prob = optimize_spar(params)
    print prob.driver.get_constraints()
    print prob.driver.get_desvars()
    print prob.driver.get_objectives()

    
if __name__ == '__main__':
    #example_mooring()
    example_spar()
    
