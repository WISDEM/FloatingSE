from openmdao.api import Group, IndepVarComp
from cylinder import Cylinder, CylinderGeometry
from spar import Spar
from mapMooring import MapMooring
from towerTransition import TowerTransition
from commonse.UtilizationSupplement import GeometricConstraints
import numpy as np

class SparAssembly(Group):

    def __init__(self, nSection, nIntPts):
        super(SparAssembly, self).__init__()

        # Run Spar Geometry component first
        self.add('sg', CylinderGeometry(nSection), promotes=['water_depth'])

        # Add in transition to tower
        self.add('tt', TowerTransition(nSection+1, diamFlag=True), promotes=['tower_metric'])

        # Next run MapMooring
        self.add('mm', MapMooring(), promotes=['water_density'])

        # Next do ballast cylind
        self.add('cyl', Cylinder(nSection, nIntPts), promotes=['air_density','air_viscosity','water_viscosity','wave_height','wave_period',
                                                               'wind_reference_speed','wind_reference_height','alpha','morison_mass_coefficient',
                                                               'material_density','E','nu','yield_stress'])

        # Run main Spar analysis
        self.add('sp', Spar(nSection, nIntPts), promotes=['turbine_mass','turbine_center_of_gravity','turbine_surge_force','turbine_pitch_moment'])

        # Manufacturing and Welding constraints
        self.add('gc', GeometricConstraints(nSection+1, diamFlag=True), promotes=['min_taper','min_d_to_t'])

        
        # Define all input variables from all models
        # SparGeometry
        self.add('freeboard',                  IndepVarComp('freeboard', 0.0), promotes=['*'])
        self.add('fairlead',                   IndepVarComp('fairlead', 0.0), promotes=['*'])
        self.add('spar_section_height',        IndepVarComp('spar_section_height', np.zeros((nSection,))), promotes=['*'])
        self.add('spar_outer_diameter',        IndepVarComp('spar_outer_diameter', np.zeros((nSection+1,))), promotes=['*'])
        self.add('spar_wall_thickness',        IndepVarComp('spar_wall_thickness', np.zeros((nSection+1,))), promotes=['*'])
        self.add('fairlead_offset_from_shell', IndepVarComp('fairlead_offset_from_shell', 0.0), promotes=['*'])
        #self.add('tower_diameter',             IndepVarComp('tower_diameter', np.zeros((nSection+1,))), promotes=['*'])
        #self.add('water_depth',                IndepVarComp('water_depth', 0.0), promotes=['*'])

        # Turbine
        #self.add('turbine_mass',               IndepVarComp('turbine_mass', 0.0), promotes=['*'])
        #self.add('turbine_center_of_gravity',  IndepVarComp('turbine_center_of_gravity', np.zeros((3,))), promotes=['*'])
        #self.add('turbine_surge_force',        IndepVarComp('turbine_surge_force', 0.0), promotes=['*'])
        #self.add('turbine_pitch_moment',       IndepVarComp('turbine_pitch_moment', 0.0), promotes=['*'])

        # Mooring
        #self.add('water_density',              IndepVarComp('water_density', 0.0), promotes=['*'])
        self.add('scope_ratio',                IndepVarComp('scope_ratio', 0.0), promotes=['*'])
        self.add('anchor_radius',              IndepVarComp('anchor_radius', 0.0), promotes=['*'])
        self.add('mooring_diameter',           IndepVarComp('mooring_diameter', 0.0), promotes=['*'])
        self.add('number_of_mooring_lines',    IndepVarComp('number_of_mooring_lines', 0, pass_by_obj=True), promotes=['*'])
        self.add('mooring_type',               IndepVarComp('mooring_type', 'chain', pass_by_obj=True), promotes=['*'])
        self.add('anchor_type',                IndepVarComp('anchor_type', 'SUCTIONPILE', pass_by_obj=True), promotes=['*'])
        self.add('drag_embedment_extra_length',IndepVarComp('drag_embedment_extra_length', 0.0), promotes=['*'])
        self.add('mooring_max_offset',         IndepVarComp('mooring_max_offset', 0.0), promotes=['*'])
        self.add('mooring_cost_rate',          IndepVarComp('mooring_cost_rate', 0.0), promotes=['*'])

        # Cylinder
        #self.add('air_density',                IndepVarComp('air_density', 0.0), promotes=['*'])
        #self.add('air_viscosity',              IndepVarComp('air_viscosity', 0.0), promotes=['*'])
        #self.add('water_viscosity',            IndepVarComp('water_viscosity', 0.0), promotes=['*'])
        #self.add('wave_height',                IndepVarComp('wave_height', 0.0), promotes=['*'])
        #self.add('wave_period',                IndepVarComp('wave_period', 0.0), promotes=['*'])
        #self.add('wind_reference_speed',       IndepVarComp('wind_reference_speed', 0.0), promotes=['*'])
        #self.add('wind_reference_height',      IndepVarComp('wind_reference_height', 0.0), promotes=['*'])
        #self.add('alpha',                      IndepVarComp('alpha', 0.0), promotes=['*'])
        #self.add('morison_mass_coefficient',   IndepVarComp('morison_mass_coefficient', 0.0), promotes=['*'])
        #self.add('material_density',           IndepVarComp('material_density', 0.0), promotes=['*'])
        #self.add('E',                          IndepVarComp('E', 0.0), promotes=['*'])
        #self.add('nu',                         IndepVarComp('nu', 0.0), promotes=['*'])
        #self.add('yield_stress',               IndepVarComp('yield_stress', 0.0), promotes=['*'])
        self.add('permanent_ballast_density',  IndepVarComp('permanent_ballast_density', 0.0), promotes=['*'])
        
        self.add('stiffener_web_height',       IndepVarComp('stiffener_web_height', np.zeros((nSection,))), promotes=['*'])
        self.add('stiffener_web_thickness',    IndepVarComp('stiffener_web_thickness', np.zeros((nSection,))), promotes=['*'])
        self.add('stiffener_flange_width',     IndepVarComp('stiffener_flange_width', np.zeros((nSection,))), promotes=['*'])
        self.add('stiffener_flange_thickness', IndepVarComp('stiffener_flange_thickness', np.zeros((nSection,))), promotes=['*'])
        self.add('stiffener_spacing',          IndepVarComp('stiffener_spacing', np.zeros((nSection,))), promotes=['*'])
        
        self.add('bulkhead_nodes',             IndepVarComp('bulkhead_nodes', [False]*(nSection+1), pass_by_obj=True ), promotes=['*'])
        self.add('permanent_ballast_height',   IndepVarComp('permanent_ballast_height', 0.0), promotes=['*'])
        self.add('bulkhead_mass_factor',       IndepVarComp('bulkhead_mass_factor', 0.0), promotes=['*'])
        self.add('ring_mass_factor',           IndepVarComp('ring_mass_factor', 0.0), promotes=['*'])
        self.add('shell_mass_factor',          IndepVarComp('shell_mass_factor', 0.0), promotes=['*'])
        self.add('spar_mass_factor',           IndepVarComp('spar_mass_factor', 0.0), promotes=['*'])
        self.add('outfitting_mass_fraction',   IndepVarComp('outfitting_mass_fraction', 0.0), promotes=['*'])
        self.add('ballast_cost_rate',          IndepVarComp('ballast_cost_rate', 0.0), promotes=['*'])
        self.add('tapered_col_cost_rate',      IndepVarComp('tapered_col_cost_rate', 0.0), promotes=['*'])
        self.add('outfitting_cost_rate',       IndepVarComp('outfitting_cost_rate', 0.0), promotes=['*'])

        # Design constraints
        #self.add('min_taper_ratio',            IndepVarComp('min_taper_ratio', 0.0), promotes=['*'])
        #self.add('min_diameter_thickness_ratio', IndepVarComp('min_diameter_thickness_ratio', 0.0), promotes=['*'])

        # Connect all input variables from all models
        #self.connect('water_depth', ['sg.water_depth', 'mm.water_depth', 'cyl.water_depth'])
        self.connect('water_depth', ['mm.water_depth', 'cyl.water_depth'])
        self.connect('freeboard', ['sg.freeboard', 'sp.base_freeboard'])
        self.connect('fairlead', ['sg.fairlead', 'mm.fairlead','sp.fairlead'])
        self.connect('spar_section_height', ['sg.section_height', 'cyl.section_height'])
        self.connect('spar_outer_diameter', ['sg.outer_diameter', 'cyl.outer_diameter', 'gc.d', 'tt.base_metric'])
        self.connect('spar_wall_thickness', ['sg.wall_thickness', 'cyl.wall_thickness', 'gc.t'])
        self.connect('fairlead_offset_from_shell', 'sg.fairlead_offset_from_shell')
        #self.connect('tower_diameter', 'tt.tower_metric')
        
        #self.connect('turbine_mass', ['cyl.stack_mass_in', 'sp.turbine_mass'])
        self.connect('turbine_mass', 'cyl.stack_mass_in')
        #self.connect('turbine_center_of_gravity', 'sp.turbine_center_of_gravity')
        #self.connect('turbine_surge_force', 'sp.turbine_surge_force')
        #self.connect('turbine_pitch_moment', 'sp.turbine_pitch_moment')

        #self.connect('water_density', ['mm.water_density', 'cyl.water_density', 'sp.water_density'])
        self.connect('water_density', ['cyl.water_density', 'sp.water_density'])
        self.connect('scope_ratio', 'mm.scope_ratio')
        self.connect('anchor_radius', 'mm.anchor_radius')
        self.connect('mooring_diameter', 'mm.mooring_diameter')
        self.connect('number_of_mooring_lines', 'mm.number_of_mooring_lines')
        self.connect('mooring_type', 'mm.mooring_type')
        self.connect('anchor_type', 'mm.anchor_type')
        self.connect('drag_embedment_extra_length', 'mm.drag_embedment_extra_length')
        self.connect('mooring_max_offset', 'mm.max_offset')
        self.connect('mooring_cost_rate', 'mm.mooring_cost_rate')
        
        #self.connect('air_density', 'cyl.air_density')
        #self.connect('air_viscosity', 'cyl.air_viscosity')
        #self.connect('water_viscosity', 'cyl.water_viscosity')
        #self.connect('wave_height', 'cyl.wave_height')
        #self.connect('wave_period', 'cyl.wave_period')
        #self.connect('wind_reference_speed', 'cyl.wind_reference_speed')
        #self.connect('wind_reference_height', 'cyl.wind_reference_height')
        #self.connect('alpha', 'cyl.alpha')
        #self.connect('morison_mass_coefficient', 'cyl.morison_mass_coefficient')
        #self.connect('material_density', 'cyl.material_density')
        #self.connect('E', 'cyl.E')
        #self.connect('nu', 'cyl.nu')
        #self.connect('yield_stress', 'cyl.yield_stress')
        self.connect('permanent_ballast_density', 'cyl.permanent_ballast_density')
        self.connect('stiffener_web_height', 'cyl.stiffener_web_height')
        self.connect('stiffener_web_thickness', 'cyl.stiffener_web_thickness')
        self.connect('stiffener_flange_width', 'cyl.stiffener_flange_width')
        self.connect('stiffener_flange_thickness', 'cyl.stiffener_flange_thickness')
        self.connect('stiffener_spacing', 'cyl.stiffener_spacing')
        self.connect('bulkhead_nodes', 'cyl.bulkhead_nodes')
        self.connect('permanent_ballast_height', 'cyl.permanent_ballast_height')
        self.connect('bulkhead_mass_factor', 'cyl.bulkhead_mass_factor')
        self.connect('ring_mass_factor', 'cyl.ring_mass_factor')
        self.connect('shell_mass_factor', 'cyl.shell_mass_factor')
        self.connect('spar_mass_factor', 'cyl.spar_mass_factor')
        self.connect('outfitting_mass_fraction', 'cyl.outfitting_mass_fraction')
        self.connect('ballast_cost_rate', 'cyl.ballast_cost_rate')
        self.connect('tapered_col_cost_rate', 'cyl.tapered_col_cost_rate')
        self.connect('outfitting_cost_rate', 'cyl.outfitting_cost_rate')

        #self.connect('min_taper_ratio', 'gc.min_taper')
        #self.connect('min_diameter_thickness_ratio', 'gc.min_d_to_t')
        
        # Link outputs from one model to inputs to another
        self.connect('sg.fairlead_radius', 'mm.fairlead_radius')
        self.connect('sg.z_nodes', 'cyl.z_nodes')
        self.connect('sg.z_section', 'cyl.z_section')
        
        self.connect('mm.mooring_mass', 'sp.mooring_mass')
        self.connect('mm.mooring_effective_mass', 'sp.mooring_effective_mass')
        self.connect('mm.mooring_cost', 'sp.mooring_cost')
        self.connect('mm.max_offset_restoring_force', 'sp.mooring_surge_restoring_force')
        
        self.connect('cyl.z_center_of_gravity', 'sp.base_cylinder_center_of_gravity')
        self.connect('cyl.z_center_of_buoyancy', 'sp.base_cylinder_center_of_buoyancy')
        self.connect('cyl.Iwater', 'sp.base_cylinder_Iwaterplane')
        self.connect('cyl.displaced_volume', 'sp.base_cylinder_displaced_volume')
        self.connect('cyl.total_mass', 'sp.base_cylinder_mass')
        self.connect('cyl.total_cost', 'sp.base_cylinder_cost')
        self.connect('cyl.variable_ballast_interp_mass', 'sp.water_ballast_mass_vector')
        self.connect('cyl.variable_ballast_interp_zpts', 'sp.water_ballast_zpts_vector')
        self.connect('cyl.surge_force_vector', 'sp.base_cylinder_surge_force')
        self.connect('cyl.surge_force_points', 'sp.base_cylinder_force_points')

         # Use complex number finite differences
        typeStr = 'fd'
        formStr = 'central'
        stepVal = 1e-5
        stepStr = 'relative'
        
        self.deriv_options['type'] = typeStr
        self.deriv_options['form'] = formStr
        self.deriv_options['step_size'] = stepVal
        self.deriv_options['step_calc'] = stepStr

        self.gc.deriv_options['type'] = typeStr
        self.gc.deriv_options['form'] = formStr
        self.gc.deriv_options['step_size'] = stepVal
        self.gc.deriv_options['step_calc'] = stepStr

