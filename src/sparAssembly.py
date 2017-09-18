from openmdao.api import Group, Problem, IndepVarComp, ScipyOptimizer
from spar import Spar
from sparGeometry import SparGeometry, NSECTIONS
from mapMooring import MapMooring
import numpy as np

class SparAssembly(Group):

    def __init__(self):
        super(SparAssembly, self).__init__()

        # Define all input variables from all models
        self.add('water_depth',                IndepVarComp('x', 0.0))
        self.add('freeboard',                  IndepVarComp('x', 0.0))
        self.add('fairlead',                   IndepVarComp('x', 0.0))
        self.add('section_height',             IndepVarComp('x', np.zeros((NSECTIONS,))))
        self.add('outer_radius',               IndepVarComp('x', np.zeros((NSECTIONS+1,))))
        self.add('wall_thickness',             IndepVarComp('x', np.zeros((NSECTIONS+1,))))
        self.add('fairlead_offset_from_shell', IndepVarComp('x', 0.0))
        self.add('scope_ratio',                IndepVarComp('x', 0.0))
        self.add('anchor_radius',              IndepVarComp('x', 0.0))
        self.add('mooring_diameter',           IndepVarComp('x', 0.0))
        self.add('number_of_mooring_lines',    IndepVarComp('x', 0))
        self.add('mooring_type',               IndepVarComp('x', 'chain'))
        self.add('anchor_type',                IndepVarComp('x', 'pile'))
        self.add('mooring_cost_rate',          IndepVarComp('x', 0.0))
        self.add('max_offset',                 IndepVarComp('x', 0.0))
        self.add('air_density',                IndepVarComp('x', 0.0))
        self.add('air_viscosity',              IndepVarComp('x', 0.0))
        self.add('water_density',              IndepVarComp('x', 0.0))
        self.add('water_viscosity',            IndepVarComp('x', 0.0))
        self.add('wave_height',                IndepVarComp('x', 0.0))
        self.add('wave_period',                IndepVarComp('x', 0.0))
        self.add('wind_reference_speed',       IndepVarComp('x', 0.0))
        self.add('wind_reference_height',      IndepVarComp('x', 0.0))
        self.add('alpha',                      IndepVarComp('x', 0.0))
        self.add('morison_mass_coefficient',   IndepVarComp('x', 0.0))
        self.add('material_density',           IndepVarComp('x', 0.0))
        self.add('E',                          IndepVarComp('x', 0.0))
        self.add('nu',                         IndepVarComp('x', 0.0))
        self.add('yield_stress',               IndepVarComp('x', 0.0))
        self.add('permanent_ballast_density',  IndepVarComp('x', 0.0))
        self.add('stiffener_web_height',       IndepVarComp('x', np.zeros((NSECTIONS,))))
        self.add('stiffener_web_thickness',    IndepVarComp('x', np.zeros((NSECTIONS,))))
        self.add('stiffener_flange_width',     IndepVarComp('x', np.zeros((NSECTIONS,))))
        self.add('stiffener_flange_thickness', IndepVarComp('x', np.zeros((NSECTIONS,))))
        self.add('stiffener_spacing',          IndepVarComp('x', np.zeros((NSECTIONS,))))
        self.add('bulkhead_nodes',             IndepVarComp('x', [False]*(NSECTIONS+1) ))
        self.add('permanent_ballast_height',   IndepVarComp('x', 0.0))
        self.add('bulkhead_mass_factor',       IndepVarComp('x', 0.0))
        self.add('ring_mass_factor',           IndepVarComp('x', 0.0))
        self.add('shell_mass_factor',          IndepVarComp('x', 0.0))
        self.add('spar_mass_factor',           IndepVarComp('x', 0.0))
        self.add('outfitting_mass_fraction',   IndepVarComp('x', 0.0))
        self.add('ballast_cost_rate',          IndepVarComp('x', 0.0))
        self.add('tapered_col_cost_rate',      IndepVarComp('x', 0.0))
        self.add('outfitting_cost_rate',       IndepVarComp('x', 0.0))
        self.add('rna_mass',                   IndepVarComp('x', 0.0))
        self.add('rna_center_of_gravity',      IndepVarComp('x', 0.0))
        self.add('rna_center_of_gravity_x',    IndepVarComp('x', 0.0))
        self.add('tower_mass',                 IndepVarComp('x', 0.0))
        self.add('tower_center_of_gravity',    IndepVarComp('x', 0.0))
        self.add('rna_wind_force',             IndepVarComp('x', 0.0))
        self.add('tower_wind_force',           IndepVarComp('x', 0.0))

        # Run Spar Geometry component first
        self.add('sg', SparGeometry())

        # Next run MapMooring
        self.add('mm', MapMooring())

        # Run main Spar analysis
        self.add('sp', Spar())

        self.connect('water_depth.x', ['sg.water_depth', 'mm.water_depth', 'sp.water_depth'])
        self.connect('water_density.x', ['mm.water_density', 'sp.water_density'])
        self.connect('fairlead.x', ['sg.fairlead', 'mm.fairlead','sp.fairlead'])
        self.connect('freeboard.x', ['sg.freeboard', 'sp.freeboard'])
        self.connect('section_height.x', ['sg.section_height', 'sp.section_height'])
        self.connect('outer_radius.x', ['sg.outer_radius', 'sp.outer_radius'])
        self.connect('wall_thickness.x', ['sg.wall_thickness', 'sp.wall_thickness'])

        self.connect('fairlead_offset_from_shell.x', 'sg.fairlead_offset_from_shell')
        self.connect('scope_ratio.x', 'mm.scope_ratio')
        self.connect('anchor_radius.x', 'mm.anchor_radius')
        self.connect('mooring_diameter.x', 'mm.mooring_diameter')
        self.connect('number_of_mooring_lines.x', 'mm.number_of_mooring_lines')
        self.connect('mooring_type.x', 'mm.mooring_type')
        self.connect('anchor_type.x', 'mm.anchor_type')
        self.connect('max_offset.x', 'mm.max_offset')
        self.connect('mooring_cost_rate.x', 'mm.mooring_cost_rate')
        self.connect('air_density.x', 'sp.air_density')
        self.connect('air_viscosity.x', 'sp.air_viscosity')
        self.connect('water_viscosity.x', 'sp.water_viscosity')
        self.connect('wave_height.x', 'sp.wave_height')
        self.connect('wave_period.x', 'sp.wave_period')
        self.connect('wind_reference_speed.x', 'sp.wind_reference_speed')
        self.connect('wind_reference_height.x', 'sp.wind_reference_height')
        self.connect('alpha.x', 'sp.alpha')
        self.connect('morison_mass_coefficient.x', 'sp.morison_mass_coefficient')
        self.connect('material_density.x', 'sp.material_density')
        self.connect('E.x', 'sp.E')
        self.connect('nu.x', 'sp.nu')
        self.connect('yield_stress.x', 'sp.yield_stress')
        self.connect('permanent_ballast_density.x', 'sp.permanent_ballast_density')
        self.connect('stiffener_web_height.x', 'sp.stiffener_web_height')
        self.connect('stiffener_web_thickness.x', 'sp.stiffener_web_thickness')
        self.connect('stiffener_flange_width.x', 'sp.stiffener_flange_width')
        self.connect('stiffener_flange_thickness.x', 'sp.stiffener_flange_thickness')
        self.connect('stiffener_spacing.x', 'sp.stiffener_spacing')
        self.connect('bulkhead_nodes.x', 'sp.bulkhead_nodes')
        self.connect('permanent_ballast_height.x', 'sp.permanent_ballast_height')
        self.connect('bulkhead_mass_factor.x', 'sp.bulkhead_mass_factor')
        self.connect('ring_mass_factor.x', 'sp.ring_mass_factor')
        self.connect('spar_mass_factor.x', 'sp.spar_mass_factor')
        self.connect('shell_mass_factor.x', 'sp.shell_mass_factor')
        self.connect('outfitting_mass_fraction.x', 'sp.outfitting_mass_fraction')
        self.connect('ballast_cost_rate.x', 'sp.ballast_cost_rate')
        self.connect('tapered_col_cost_rate.x', 'sp.tapered_col_cost_rate')
        self.connect('outfitting_cost_rate.x', 'sp.outfitting_cost_rate')
        self.connect('rna_mass.x', 'sp.rna_mass')
        self.connect('rna_center_of_gravity.x', 'sp.rna_center_of_gravity')
        self.connect('rna_center_of_gravity_x.x', 'sp.rna_center_of_gravity_x')
        self.connect('rna_wind_force.x', 'sp.rna_wind_force')
        self.connect('tower_mass.x', 'sp.tower_mass')
        self.connect('tower_center_of_gravity.x', 'sp.tower_center_of_gravity')
        self.connect('tower_wind_force.x', 'sp.tower_wind_force')
        
        # Link outputs from one model to inputs to another
        self.connect('sg.fairlead_radius', 'mm.fairlead_radius')
        self.connect('sg.draft', 'sp.draft')
        self.connect('sg.z_nodes', 'sp.z_nodes')
        self.connect('sg.z_section', 'sp.z_section')
        self.connect('mm.mooring_mass', 'sp.mooring_mass')
        self.connect('mm.mooring_cost', 'sp.mooring_cost')
        self.connect('mm.vertical_load', 'sp.mooring_vertical_load')
        self.connect('mm.max_offset_restoring_force', 'sp.mooring_restoring_force')

        self.deriv_options['type'] = 'fd' #'cs'

        
def optimize_spar(params):

    # Setup the problem
    prob = Problem()
    prob.root = SparAssembly()
    
    # Establish the optimization driver, then set design variables and constraints
    myopt={}
    prob.driver = ScipyOptimizer() #COBYLAdriver()
    #prob.driver.options['maxiter'] = 1
    
    # DESIGN VARIABLES
    prob.driver.add_desvar('freeboard.x',lower=0.0)
    prob.driver.add_desvar('fairlead.x',lower=0.0)
    prob.driver.add_desvar('fairlead_offset_from_shell.x',lower=0.0)
    prob.driver.add_desvar('section_height.x',lower=1e-2)
    prob.driver.add_desvar('outer_radius.x',lower=1.0)
    prob.driver.add_desvar('wall_thickness.x',lower=1e-2)

    prob.driver.add_desvar('scope_ratio.x', lower=1.0) #>1 means longer than water depth
    prob.driver.add_desvar('anchor_radius.x', lower=0.0)
    prob.driver.add_desvar('mooring_diameter.x', lower=1e-2)
    # TODO: Integer design variables
    #prob.driver.add_desvar('number_of_mooring_lines.x', lower=1)
    #prob.driver.add_desvar('mooring_type.x')
    #prob.driver.add_desvar('anchor_type.x')

    prob.driver.add_desvar('stiffener_web_height.x', lower=1e-3)
    prob.driver.add_desvar('stiffener_web_thickness.x', lower=1e-3)
    prob.driver.add_desvar('stiffener_flange_width.x', lower=1e-3)
    prob.driver.add_desvar('stiffener_flange_thickness.x', lower=1e-3)
    prob.driver.add_desvar('stiffener_spacing.x', lower=1e-2)
    prob.driver.add_desvar('permanent_ballast_height.x', lower=0.0)
    # TODO: Boolean design variables
    #prob.driver.add_desvar('bulkhead_nodes.x')

    
    # CONSTRAINTS
    # Ensure that draft is greater than 0 (spar length>0) and that less than water depth
    prob.driver.add_constraint('sg.draft_depth_ratio',lower=0.0, upper=1.0)

    # Ensure max mooring line tension is less than X% of MBL: 60% for intact mooring, 80% for damanged
    prob.driver.add_constraint('mm.safety_factor',lower=0.0, upper=0.8)

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
    prob.driver.add_constraint('sp.variable_ballast_height', lower=0.0, upper=params['spar_length'])

    # Surge restoring force should be greater than wave-wind forces (ratio < 1)
    prob.driver.add_constraint('sp.offset_force_ratio', lower=0.0, upper=1.0)

    # Heel angle should be less than 6deg for ordinary operation, less than 10 for extreme conditions
    prob.driver.add_constraint('sp.heel_angle', upper=10.0)

    
    # OBJECTIVE FUNCTION: Minimize total cost!
    prob.driver.add_objective('sp.total_cost')

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
    #prob.run_once()
    prob.run()

    return prob



if __name__ == '__main__':
    params = {}
    params['water_depth'] = 218.0
    params['freeboard'] = 15.0
    params['fairlead'] = 10.0
    params['spar_length'] = 90.0
    params['outer_radius'] = 35.0
    params['wall_thickness'] = 0.1
    params['fairlead_offset_from_shell'] = 0.05
    params['scope_ratio'] = 1.5
    params['anchor_radius'] = 175.0
    params['mooring_diameter'] = 0.5
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
    params['wind_reference_height'] = 97.0
    params['alpha'] = 0.11
    params['morison_mass_coefficient'] = 2.0
    params['material_density'] = 7850.0
    params['E'] = 200e9
    params['nu'] = 0.3
    params['yield_stress'] = 3.45e8
    params['permanent_ballast_density'] = 4492.0
    params['stiffener_web_height']= 0.2
    params['stiffener_web_thickness'] = 0.05
    params['stiffener_flange_width'] = 0.15
    params['stiffener_flange_thickness'] = 0.05
    params['stiffener_spacing'] = 0.2
    params['permanent_ballast_height'] = 10.0
    params['bulkhead_mass_factor'] = 1.0
    params['ring_mass_factor'] = 1.0
    params['shell_mass_factor'] = 1.0
    params['spar_mass_factor'] = 1.05
    params['outfitting_mass_fraction'] = 0.06
    params['ballast_cost_rate'] = 100.0
    params['tapered_col_cost_rate'] = 4720.0
    params['outfitting_cost_rate'] = 6980.0
    params['rna_mass']= 3.655e5
    params['rna_center_of_gravity'] = 3.5 + 80.0
    params['rna_center_of_gravity_x'] = 5.75
    params['tower_mass'] = 3.66952e5
    params['tower_center_of_gravity'] = 35.0
    params['rna_wind_force'] = 1e4 # TODO: Better guess
    params['tower_wind_force'] = 1e4 # TODO: Better guess

    prob = optimize_spar(params)
    print prob.driver.get_constraints()
