from openmdao.api import Group, IndepVarComp, DirectSolver, ScipyGMRES, Newton, NLGaussSeidel, Brent, RunOnce
from cylinder import Cylinder, CylinderGeometry
from semi import Semi, SemiGeometry
from semiPontoon import SemiPontoon
from mapMooring import MapMooring
from towerTransition import TowerTransition
from commonse.UtilizationSupplement import GeometricConstraints
import numpy as np

class SemiAssembly(Group):

    def __init__(self, nSection, nIntPts):
        super(SemiAssembly, self).__init__()

        # Run Cylinder Geometry components first
        self.add('geomBase', CylinderGeometry(nSection), promotes=['water_depth'])
        self.add('geomBall', CylinderGeometry(nSection))

        # Run Semi Geometry for interfaces
        self.add('sg', SemiGeometry(nSection))
        
        # Add in transition to tower
        self.add('tt', TowerTransition(nSection+1, diamFlag=True), promotes=['tower_metric'])

        # Next run MapMooring
        self.add('mm', MapMooring(), promotes=['water_density'])

        # Next do base and ballast cylinders
        # Ballast cylinders are replicated from same design in the components
        self.add('base', Cylinder(nSection, nIntPts), promotes=['air_density','air_viscosity','water_viscosity','wave_height','wave_period',
                                                               'wind_reference_speed','wind_reference_height','alpha','morison_mass_coefficient',
                                                               'material_density','E','nu','yield_stress'])
        self.add('ball', Cylinder(nSection, nIntPts))

        # Add in the connecting truss
        self.add('pon', SemiPontoon(nSection), promotes=['G'])
        
        # Run main Semi analysis
        self.add('sm', Semi(nSection, nIntPts), promotes=['turbine_mass','turbine_center_of_gravity','turbine_surge_force','turbine_pitch_moment',
                                                          'total_cost','total_mass'])

        # Manufacturing and Welding constraints
        self.add('gcBase', GeometricConstraints(nSection+1, diamFlag=True), promotes=['min_taper','min_d_to_t'])
        self.add('gcBall', GeometricConstraints(nSection+1, diamFlag=True))

        # Define all input variables from all models
        # SemiGeometry
        self.add('radius_to_ballast_cylinder', IndepVarComp('radius_to_ballast_cylinder', 0.0), promotes=['*'])
        
        self.add('fairlead',                   IndepVarComp('fairlead', 0.0), promotes=['*'])
        self.add('fairlead_offset_from_shell', IndepVarComp('fairlead_offset_from_shell', 0.0), promotes=['*'])
        #self.add('water_depth',                IndepVarComp('water_depth', 0.0), promotes=['*'])
        #self.add('tower_diameter',             IndepVarComp('tower_diameter', np.zeros((nSection+1,))), promotes=['*'])

        self.add('freeboard_base',             IndepVarComp('freeboard_base', 0.0), promotes=['*'])
        self.add('section_height_base',        IndepVarComp('section_height_base', np.zeros((nSection,))), promotes=['*'])
        self.add('outer_diameter_base',        IndepVarComp('outer_diameter_base', np.zeros((nSection+1,))), promotes=['*'])
        self.add('wall_thickness_base',        IndepVarComp('wall_thickness_base', np.zeros((nSection+1,))), promotes=['*'])

        self.add('freeboard_ballast',          IndepVarComp('freeboard_ballast', 0.0), promotes=['*'])
        self.add('section_height_ballast',     IndepVarComp('section_height_ballast', np.zeros((nSection,))), promotes=['*'])
        self.add('outer_diameter_ballast',     IndepVarComp('outer_diameter_ballast', np.zeros((nSection+1,))), promotes=['*'])
        self.add('wall_thickness_ballast',     IndepVarComp('wall_thickness_ballast', np.zeros((nSection+1,))), promotes=['*'])

        # Turbine
        #self.add('turbine_mass',               IndepVarComp('turbine_mass', 0.0), promotes=['*'])
        #self.add('dummy_mass',                 IndepVarComp('dummy_mass', 0.0), promotes=['*'])
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
        
        self.add('stiffener_web_height_base',       IndepVarComp('stiffener_web_height_base', np.zeros((nSection,))), promotes=['*'])
        self.add('stiffener_web_thickness_base',    IndepVarComp('stiffener_web_thickness_base', np.zeros((nSection,))), promotes=['*'])
        self.add('stiffener_flange_width_base',     IndepVarComp('stiffener_flange_width_base', np.zeros((nSection,))), promotes=['*'])
        self.add('stiffener_flange_thickness_base', IndepVarComp('stiffener_flange_thickness_base', np.zeros((nSection,))), promotes=['*'])
        self.add('stiffener_spacing_base',          IndepVarComp('stiffener_spacing_base', np.zeros((nSection,))), promotes=['*'])
        self.add('bulkhead_nodes_base',             IndepVarComp('bulkhead_nodes_base', [False]*(nSection+1), pass_by_obj=True ), promotes=['*'])
        self.add('permanent_ballast_height_base',   IndepVarComp('permanent_ballast_height_base', 0.0), promotes=['*'])

        self.add('stiffener_web_height_ballast',       IndepVarComp('stiffener_web_height_ballast', np.zeros((nSection,))), promotes=['*'])
        self.add('stiffener_web_thickness_ballast',    IndepVarComp('stiffener_web_thickness_ballast', np.zeros((nSection,))), promotes=['*'])
        self.add('stiffener_flange_width_ballast',     IndepVarComp('stiffener_flange_width_ballast', np.zeros((nSection,))), promotes=['*'])
        self.add('stiffener_flange_thickness_ballast', IndepVarComp('stiffener_flange_thickness_ballast', np.zeros((nSection,))), promotes=['*'])
        self.add('stiffener_spacing_ballast',          IndepVarComp('stiffener_spacing_ballast', np.zeros((nSection,))), promotes=['*'])
        self.add('bulkhead_nodes_ballast',             IndepVarComp('bulkhead_nodes_ballast', [False]*(nSection+1), pass_by_obj=True ), promotes=['*'])
        self.add('permanent_ballast_height_ballast',   IndepVarComp('permanent_ballast_height_ballast', 0.0), promotes=['*'])

        self.add('bulkhead_mass_factor',       IndepVarComp('bulkhead_mass_factor', 0.0), promotes=['*'])
        self.add('ring_mass_factor',           IndepVarComp('ring_mass_factor', 0.0), promotes=['*'])
        self.add('shell_mass_factor',          IndepVarComp('shell_mass_factor', 0.0), promotes=['*'])
        self.add('spar_mass_factor',           IndepVarComp('spar_mass_factor', 0.0), promotes=['*'])
        self.add('outfitting_mass_fraction',   IndepVarComp('outfitting_mass_fraction', 0.0), promotes=['*'])
        self.add('ballast_cost_rate',          IndepVarComp('ballast_cost_rate', 0.0), promotes=['*'])
        self.add('tapered_col_cost_rate',      IndepVarComp('tapered_col_cost_rate', 0.0), promotes=['*'])
        self.add('outfitting_cost_rate',       IndepVarComp('outfitting_cost_rate', 0.0), promotes=['*'])

        # Pontoons
        #self.add('G',                          IndepVarComp('G', 0.0), promotes=['*'])
        self.add('number_of_ballast_columns',  IndepVarComp('number_of_ballast_columns', 0, pass_by_obj=True), promotes=['*'])
        self.add('pontoon_outer_diameter',       IndepVarComp('pontoon_outer_diameter', 0.0), promotes=['*'])
        self.add('pontoon_inner_diameter',       IndepVarComp('pontoon_inner_diameter', 0.0), promotes=['*'])
        self.add('cross_attachment_pontoons',  IndepVarComp('cross_attachment_pontoons', True, pass_by_obj=True), promotes=['*'])
        self.add('lower_attachment_pontoons',  IndepVarComp('lower_attachment_pontoons', True, pass_by_obj=True), promotes=['*'])
        self.add('upper_attachment_pontoons',  IndepVarComp('upper_attachment_pontoons', True, pass_by_obj=True), promotes=['*'])
        self.add('lower_ring_pontoons',        IndepVarComp('lower_ring_pontoons', True, pass_by_obj=True), promotes=['*'])
        self.add('upper_ring_pontoons',        IndepVarComp('upper_ring_pontoons', True, pass_by_obj=True), promotes=['*'])
        self.add('pontoon_cost_rate',          IndepVarComp('pontoon_cost_rate', 0.0), promotes=['*'])
        self.add('base_connection_ratio_min', IndepVarComp('base_connection_ratio_min', 0.0), promotes=['*'])
        self.add('ballast_connection_ratio_min', IndepVarComp('ballast_connection_ratio_min', 0.0), promotes=['*'])

        # Design constraints
        #self.add('min_taper_ratio',            IndepVarComp('min_taper_ratio', 0.0), promotes=['*'])
        #self.add('min_diameter_thickness_ratio', IndepVarComp('min_diameter_thickness_ratio', 0.0), promotes=['*'])

        # Connect all input variables from all models
        #self.connect('water_depth', ['geomBase.water_depth', 'geomBall.water_depth', 'mm.water_depth', 'base.water_depth', 'ball.water_depth'])
        self.connect('water_depth', ['geomBall.water_depth', 'mm.water_depth', 'base.water_depth', 'ball.water_depth'])
        self.connect('radius_to_ballast_cylinder', ['sg.radius_to_ballast_cylinder', 'pon.radius_to_ballast_cylinder', 'sm.radius_to_ballast_cylinder'])

        self.connect('freeboard_base', ['geomBase.freeboard', 'sm.base_freeboard'])
        self.connect('section_height_base', ['geomBase.section_height', 'base.section_height'])
        self.connect('outer_diameter_base', ['geomBase.outer_diameter', 'base.outer_diameter', 'sg.base_outer_diameter',
                                             'pon.base_outer_diameter', 'gcBase.d', 'tt.base_metric'])
        self.connect('wall_thickness_base', ['geomBase.wall_thickness', 'base.wall_thickness', 'pon.base_wall_thickness', 'gcBase.t'])

        #self.connect('turbine_mass', ['base.stack_mass_in', 'sm.turbine_mass', 'pon.turbine_mass'])
        self.connect('turbine_mass', ['base.stack_mass_in', 'pon.turbine_mass'])
        #self.connect('dummy_mass', 'ball.stack_mass_in')
        #self.connect('turbine_center_of_gravity', 'sm.turbine_center_of_gravity')
        #self.connect('turbine_surge_force', ['pon.turbine_surge_force', 'sm.turbine_surge_force'])
        self.connect('turbine_surge_force', 'pon.turbine_surge_force')
        #self.connect('turbine_pitch_moment', ['pon.turbine_pitch_moment', 'sm.turbine_pitch_moment'])
        self.connect('turbine_pitch_moment', 'pon.turbine_pitch_moment')

        self.connect('freeboard_ballast', 'geomBall.freeboard')
        self.connect('section_height_ballast', ['geomBall.section_height', 'ball.section_height'])
        self.connect('outer_diameter_ballast', ['geomBall.outer_diameter', 'ball.outer_diameter', 'sg.ballast_outer_diameter', 'pon.ballast_outer_diameter', 'gcBall.d'])
        self.connect('wall_thickness_ballast', ['geomBall.wall_thickness', 'ball.wall_thickness', 'pon.ballast_wall_thickness', 'gcBall.t'])

        self.connect('fairlead', ['geomBase.fairlead', 'geomBall.fairlead', 'sg.fairlead', 'mm.fairlead','sm.fairlead'])
        self.connect('fairlead_offset_from_shell', ['geomBase.fairlead_offset_from_shell', 'geomBall.fairlead_offset_from_shell', 'sg.fairlead_offset_from_shell'])
        self.connect('tower_metric', 'pon.tower_diameter')

        #self.connect('water_density', ['mm.water_density', 'base.water_density', 'ball.water_density', 'pon.water_density', 'sm.water_density'])
        self.connect('water_density', ['base.water_density', 'ball.water_density', 'pon.water_density', 'sm.water_density'])
        self.connect('scope_ratio', 'mm.scope_ratio')
        self.connect('anchor_radius', 'mm.anchor_radius')
        self.connect('mooring_diameter', 'mm.mooring_diameter')
        self.connect('number_of_mooring_lines', 'mm.number_of_mooring_lines')
        self.connect('mooring_type', 'mm.mooring_type')
        self.connect('anchor_type', 'mm.anchor_type')
        self.connect('drag_embedment_extra_length', 'mm.drag_embedment_extra_length')
        self.connect('mooring_max_offset', 'mm.max_offset')
        self.connect('mooring_cost_rate', 'mm.mooring_cost_rate')
        
        #self.connect('air_density', ['base.air_density', 'ball.air_density'])
        self.connect('air_density', 'ball.air_density')
        #self.connect('air_viscosity', ['base.air_viscosity', 'ball.air_viscosity'])
        self.connect('air_viscosity', 'ball.air_viscosity')
        #self.connect('water_viscosity', ['base.water_viscosity', 'ball.water_viscosity'])
        self.connect('water_viscosity', 'ball.water_viscosity')
        #self.connect('wave_height', ['base.wave_height', 'ball.wave_height'])
        self.connect('wave_height', 'ball.wave_height')
        #self.connect('wave_period', ['base.wave_period', 'ball.wave_period'])
        self.connect('wave_period', 'ball.wave_period')
        #self.connect('wind_reference_speed', ['base.wind_reference_speed', 'ball.wind_reference_speed'])
        self.connect('wind_reference_speed', 'ball.wind_reference_speed')
        #self.connect('wind_reference_height', ['base.wind_reference_height', 'ball.wind_reference_height'])
        self.connect('wind_reference_height', 'ball.wind_reference_height')
        #self.connect('alpha', ['base.alpha', 'ball.alpha'])
        self.connect('alpha', 'ball.alpha')
        #self.connect('morison_mass_coefficient', ['base.morison_mass_coefficient', 'ball.morison_mass_coefficient'])
        self.connect('morison_mass_coefficient', 'ball.morison_mass_coefficient')
        #self.connect('material_density', ['base.material_density', 'ball.material_density', 'pon.material_density'])
        self.connect('material_density', ['ball.material_density', 'pon.material_density'])
        #self.connect('E', ['base.E', 'ball.E', 'pon.E'])
        self.connect('E', ['ball.E', 'pon.E'])
        #self.connect('G', 'pon.G')
        #self.connect('nu', ['base.nu', 'ball.nu'])
        self.connect('nu', 'ball.nu')
        #self.connect('yield_stress', ['base.yield_stress', 'ball.yield_stress', 'pon.yield_stress'])
        self.connect('yield_stress', ['ball.yield_stress', 'pon.yield_stress'])
        self.connect('permanent_ballast_density', ['base.permanent_ballast_density', 'ball.permanent_ballast_density'])
        
        self.connect('stiffener_web_height_base', 'base.stiffener_web_height')
        self.connect('stiffener_web_thickness_base', 'base.stiffener_web_thickness')
        self.connect('stiffener_flange_width_base', 'base.stiffener_flange_width')
        self.connect('stiffener_flange_thickness_base', 'base.stiffener_flange_thickness')
        self.connect('stiffener_spacing_base', 'base.stiffener_spacing')
        self.connect('bulkhead_nodes_base', 'base.bulkhead_nodes')
        self.connect('permanent_ballast_height_base', 'base.permanent_ballast_height')
        
        self.connect('stiffener_web_height_ballast', 'ball.stiffener_web_height')
        self.connect('stiffener_web_thickness_ballast', 'ball.stiffener_web_thickness')
        self.connect('stiffener_flange_width_ballast', 'ball.stiffener_flange_width')
        self.connect('stiffener_flange_thickness_ballast', 'ball.stiffener_flange_thickness')
        self.connect('stiffener_spacing_ballast', 'ball.stiffener_spacing')
        self.connect('bulkhead_nodes_ballast', 'ball.bulkhead_nodes')
        self.connect('permanent_ballast_height_ballast', 'ball.permanent_ballast_height')
        
        self.connect('bulkhead_mass_factor', ['base.bulkhead_mass_factor', 'ball.bulkhead_mass_factor'])
        self.connect('ring_mass_factor', ['base.ring_mass_factor', 'ball.ring_mass_factor'])
        self.connect('shell_mass_factor', ['base.shell_mass_factor', 'ball.shell_mass_factor'])
        self.connect('spar_mass_factor', ['base.spar_mass_factor', 'ball.spar_mass_factor'])
        self.connect('outfitting_mass_fraction', ['base.outfitting_mass_fraction', 'ball.outfitting_mass_fraction'])
        self.connect('ballast_cost_rate', ['base.ballast_cost_rate', 'ball.ballast_cost_rate'])
        self.connect('tapered_col_cost_rate', ['base.tapered_col_cost_rate', 'ball.tapered_col_cost_rate'])
        self.connect('outfitting_cost_rate', ['base.outfitting_cost_rate', 'ball.outfitting_cost_rate'])

        self.connect('pontoon_outer_diameter', 'pon.pontoon_outer_diameter')
        self.connect('pontoon_inner_diameter', 'pon.pontoon_inner_diameter')
        self.connect('cross_attachment_pontoons', 'pon.cross_attachment_pontoons')
        self.connect('lower_attachment_pontoons', 'pon.lower_attachment_pontoons')
        self.connect('upper_attachment_pontoons', 'pon.upper_attachment_pontoons')
        self.connect('lower_ring_pontoons', 'pon.lower_ring_pontoons')
        self.connect('upper_ring_pontoons', 'pon.upper_ring_pontoons')
        self.connect('pontoon_cost_rate', 'pon.pontoon_cost_rate')

        self.connect('number_of_ballast_columns', ['pon.number_of_ballast_cylinders', 'sm.number_of_ballast_cylinders'])

        #self.connect('min_taper_ratio', ['gcBase.min_taper', 'gcBall.min_taper'])
        self.connect('min_taper', 'gcBall.min_taper')
        #self.connect('min_diameter_thickness_ratio', ['gcBase.min_d_to_t', 'gcBall.min_d_to_t'])
        self.connect('min_d_to_t', 'gcBall.min_d_to_t')
        
        # Link outputs from one model to inputs to another
        self.connect('geomBase.fairlead_radius', 'mm.fairlead_radius')
        self.connect('geomBase.z_nodes', ['base.z_nodes', 'pon.base_z_nodes'])
        self.connect('geomBase.z_section', 'base.z_section')
        self.connect('geomBall.z_nodes', ['ball.z_nodes', 'sg.ballast_z_nodes', 'pon.ballast_z_nodes'])
        self.connect('geomBall.z_section', 'ball.z_section')
        
        self.connect('mm.mooring_mass', 'sm.mooring_mass')
        self.connect('mm.mooring_effective_mass', 'sm.mooring_effective_mass')
        self.connect('mm.mooring_cost', 'sm.mooring_cost')
        self.connect('mm.max_offset_restoring_force', 'sm.mooring_surge_restoring_force')
        
        self.connect('base.z_center_of_gravity', 'sm.base_cylinder_center_of_gravity')
        self.connect('base.z_center_of_buoyancy', 'sm.base_cylinder_center_of_buoyancy')
        self.connect('base.Iwater', 'sm.base_cylinder_Iwaterplane')
        self.connect('base.displaced_volume', ['pon.base_cylinder_displaced_volume', 'sm.base_cylinder_displaced_volume'])
        self.connect('base.total_mass', ['pon.base_cylinder_mass', 'sm.base_cylinder_mass'])
        self.connect('base.total_cost', 'sm.base_cylinder_cost')
        self.connect('base.variable_ballast_interp_mass', 'sm.water_ballast_mass_vector')
        self.connect('base.variable_ballast_interp_zpts', 'sm.water_ballast_zpts_vector')
        self.connect('base.surge_force_vector', 'sm.base_cylinder_surge_force')
        self.connect('base.surge_force_points', 'sm.base_cylinder_force_points')

        self.connect('ball.z_center_of_gravity', 'sm.ballast_cylinder_center_of_gravity')
        self.connect('ball.z_center_of_buoyancy', 'sm.ballast_cylinder_center_of_buoyancy')
        self.connect('ball.Iwater', 'sm.ballast_cylinder_Iwaterplane')
        self.connect('ball.Awater', 'sm.ballast_cylinder_Awaterplane')
        self.connect('ball.displaced_volume', ['pon.ballast_cylinder_displaced_volume', 'sm.ballast_cylinder_displaced_volume'])
        self.connect('ball.total_mass', ['pon.ballast_cylinder_mass', 'sm.ballast_cylinder_mass'])
        self.connect('ball.total_cost', 'sm.ballast_cylinder_cost')
        self.connect('ball.surge_force_vector', 'sm.ballast_cylinder_surge_force')
        self.connect('ball.surge_force_points', 'sm.ballast_cylinder_force_points')

        self.connect('pon.pontoon_mass', 'sm.pontoon_mass')
        self.connect('pon.pontoon_cost', 'sm.pontoon_cost')
        self.connect('pon.pontoon_buoyancy', 'sm.pontoon_buoyancy')
        self.connect('pon.pontoon_center_of_buoyancy', 'sm.pontoon_center_of_buoyancy')
        self.connect('pon.pontoon_center_of_gravity', 'sm.pontoon_center_of_gravity')
        self.connect('base_connection_ratio_min', 'pon.base_connection_ratio_min')
        self.connect('ballast_connection_ratio_min', 'pon.ballast_connection_ratio_min')

         # Use complex number finite differences
        typeStr = 'fd'
        formStr = 'central'
        stepVal = 1e-5
        stepStr = 'relative'
        
        self.deriv_options['type'] = typeStr
        self.deriv_options['form'] = formStr
        self.deriv_options['step_size'] = stepVal
        self.deriv_options['step_calc'] = stepStr

        self.gcBase.deriv_options['type'] = typeStr
        self.gcBase.deriv_options['form'] = formStr
        self.gcBase.deriv_options['step_size'] = stepVal
        self.gcBase.deriv_options['step_calc'] = stepStr

        self.gcBall.deriv_options['type'] = typeStr
        self.gcBall.deriv_options['form'] = formStr
        self.gcBall.deriv_options['step_size'] = stepVal
        self.gcBall.deriv_options['step_calc'] = stepStr

