from openmdao.api import Group, IndepVarComp, DirectSolver, ScipyGMRES, Newton, NLGaussSeidel, Brent, RunOnce
from column import Column, ColumnGeometry
from substructure import Semi, SemiGeometry, SubstructureDiscretization
from semiPontoon import SemiPontoon
from mapMooring import MapMooring
from towerTransition import TowerTransition
import numpy as np
        

class SemiAssembly(Group):

    def __init__(self, nSection):
        super(SemiAssembly, self).__init__()

        #self.add('geomsys', SubstructureDiscretization(nSection), promotes=['z_system'])
        nFull = 5*nSection+1

        # Next do base and ballast columns
        # Ballast columns are replicated from same design in the components
        self.add('base', Column(nSection, nFull), promotes=['water_depth','water_density','material_density','E','nu','yield_stress','z0',
                                                              'Uref','zref','shearExp','beta','yaw','Uc','hmax','T','cd_usr','cm','loading',
                                                              'min_taper','min_d_to_t'])
        self.add('ball', Column(nSection, nFull), promotes=['water_depth','water_density','material_density','E','nu','yield_stress','z0',
                                                              'Uref','zref','shearExp','beta','yaw','Uc','hmax','T','cd_usr','cm','loading',
                                                              'min_taper','min_d_to_t'])

        # Run Semi Geometry for interfaces
        self.add('sg', SemiGeometry(nFull))
        
        # Add in transition to tower
        self.add('tt', TowerTransition(nSection+1, diamFlag=True), promotes=['tower_metric'])

        # Next run MapMooring
        self.add('mm', MapMooring(), promotes=['water_density','water_depth'])

        # Add in the connecting truss
        self.add('pon', SemiPontoon(nFull), promotes=['water_density','turbine_mass','turbine_force','turbine_moment','turbine_I_base',
                                                      'material_density','E','G','yield_stress'])
        
        # Run main Semi analysis
        self.add('sm', Semi(nFull), promotes=['water_density','turbine_mass','turbine_center_of_gravity','turbine_force','turbine_moment',
                                              'total_cost','total_mass'])

        # Define all input variables from all models
        # SemiGeometry
        self.add('radius_to_ballast_column', IndepVarComp('radius_to_ballast_column', 0.0), promotes=['*'])
        
        self.add('fairlead',                   IndepVarComp('fairlead', 0.0), promotes=['*'])
        self.add('fairlead_offset_from_shell', IndepVarComp('fairlead_offset_from_shell', 0.0), promotes=['*'])

        self.add('freeboard_base',             IndepVarComp('freeboard_base', 0.0), promotes=['*'])
        self.add('section_height_base',        IndepVarComp('section_height_base', np.zeros((nSection,))), promotes=['*'])
        self.add('outer_diameter_base',        IndepVarComp('outer_diameter_base', np.zeros((nSection+1,))), promotes=['*'])
        self.add('wall_thickness_base',        IndepVarComp('wall_thickness_base', np.zeros((nSection+1,))), promotes=['*'])

        self.add('freeboard_ballast',          IndepVarComp('freeboard_ballast', 0.0), promotes=['*'])
        self.add('section_height_ballast',     IndepVarComp('section_height_ballast', np.zeros((nSection,))), promotes=['*'])
        self.add('outer_diameter_ballast',     IndepVarComp('outer_diameter_ballast', np.zeros((nSection+1,))), promotes=['*'])
        self.add('wall_thickness_ballast',     IndepVarComp('wall_thickness_ballast', np.zeros((nSection+1,))), promotes=['*'])

        # Mooring
        self.add('scope_ratio',                IndepVarComp('scope_ratio', 0.0), promotes=['*'])
        self.add('anchor_radius',              IndepVarComp('anchor_radius', 0.0), promotes=['*'])
        self.add('mooring_diameter',           IndepVarComp('mooring_diameter', 0.0), promotes=['*'])
        self.add('number_of_mooring_lines',    IndepVarComp('number_of_mooring_lines', 0, pass_by_obj=True), promotes=['*'])
        self.add('mooring_type',               IndepVarComp('mooring_type', 'chain', pass_by_obj=True), promotes=['*'])
        self.add('anchor_type',                IndepVarComp('anchor_type', 'SUCTIONPILE', pass_by_obj=True), promotes=['*'])
        self.add('drag_embedment_extra_length',IndepVarComp('drag_embedment_extra_length', 0.0), promotes=['*'])
        self.add('mooring_max_offset',         IndepVarComp('mooring_max_offset', 0.0), promotes=['*'])
        self.add('mooring_max_heel',           IndepVarComp('mooring_max_heel', 0.0), promotes=['*'])
        self.add('mooring_cost_rate',          IndepVarComp('mooring_cost_rate', 0.0), promotes=['*'])

        # Column
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
        self.add('pontoon_outer_diameter',     IndepVarComp('pontoon_outer_diameter', 0.0), promotes=['*'])
        self.add('pontoon_wall_thickness',     IndepVarComp('pontoon_wall_thickness', 0.0), promotes=['*'])
        self.add('outer_cross_pontoons',       IndepVarComp('outer_cross_pontoons', True, pass_by_obj=True), promotes=['*'])
        self.add('cross_attachment_pontoons',  IndepVarComp('cross_attachment_pontoons', True, pass_by_obj=True), promotes=['*'])
        self.add('lower_attachment_pontoons',  IndepVarComp('lower_attachment_pontoons', True, pass_by_obj=True), promotes=['*'])
        self.add('upper_attachment_pontoons',  IndepVarComp('upper_attachment_pontoons', True, pass_by_obj=True), promotes=['*'])
        self.add('lower_ring_pontoons',        IndepVarComp('lower_ring_pontoons', True, pass_by_obj=True), promotes=['*'])
        self.add('upper_ring_pontoons',        IndepVarComp('upper_ring_pontoons', True, pass_by_obj=True), promotes=['*'])
        self.add('pontoon_cost_rate',          IndepVarComp('pontoon_cost_rate', 0.0), promotes=['*'])
        self.add('connection_ratio_max',       IndepVarComp('connection_ratio_max', 0.0), promotes=['*'])
        self.add('base_pontoon_attach_lower',  IndepVarComp('base_pontoon_attach_lower', 0.0), promotes=['*'])
        self.add('base_pontoon_attach_upper',  IndepVarComp('base_pontoon_attach_upper', 0.0), promotes=['*'])

        # Connect all input variables from all models
        self.connect('radius_to_ballast_column', ['sg.radius_to_ballast_column', 'pon.radius_to_ballast_column', 'sm.radius_to_ballast_column'])

        self.connect('freeboard_base', ['base.freeboard', 'sm.base_freeboard'])
        self.connect('section_height_base', 'base.section_height')
        self.connect('outer_diameter_base', ['base.diameter','tt.base_metric'])
        self.connect('wall_thickness_base', 'base.wall_thickness')

        self.connect('turbine_mass', 'base.stack_mass_in')
        #self.connect('dummy_mass', 'ball.stack_mass_in')

        self.connect('freeboard_ballast', 'ball.freeboard')
        self.connect('section_height_ballast', 'ball.section_height')
        self.connect('outer_diameter_ballast', 'ball.diameter')
        self.connect('wall_thickness_ballast', 'ball.wall_thickness')

        self.connect('fairlead', ['base.fairlead','ball.fairlead','sg.fairlead','mm.fairlead','sm.fairlead','pon.fairlead'])
        self.connect('fairlead_offset_from_shell', 'sg.fairlead_offset_from_shell')

        self.connect('scope_ratio', 'mm.scope_ratio')
        self.connect('anchor_radius', 'mm.anchor_radius')
        self.connect('mooring_diameter', 'mm.mooring_diameter')
        self.connect('number_of_mooring_lines', 'mm.number_of_mooring_lines')
        self.connect('mooring_type', 'mm.mooring_type')
        self.connect('anchor_type', 'mm.anchor_type')
        self.connect('drag_embedment_extra_length', 'mm.drag_embedment_extra_length')
        self.connect('mooring_max_offset', 'mm.max_offset')
        self.connect('mooring_max_heel', ['mm.max_heel', 'sm.max_heel'])
        self.connect('mooring_cost_rate', 'mm.mooring_cost_rate')

        # To do: connect these to independent variables
        self.connect('base.windLoads.rho','ball.windLoads.rho')
        self.connect('base.windLoads.mu','ball.windLoads.mu')
        self.connect('base.waveLoads.mu','ball.waveLoads.mu')

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
        self.connect('shell_mass_factor', ['base.cyl_mass.outfitting_factor', 'ball.cyl_mass.outfitting_factor'])
        self.connect('spar_mass_factor', ['base.spar_mass_factor', 'ball.spar_mass_factor'])
        self.connect('outfitting_mass_fraction', ['base.outfitting_mass_fraction', 'ball.outfitting_mass_fraction'])
        self.connect('ballast_cost_rate', ['base.ballast_cost_rate', 'ball.ballast_cost_rate'])
        self.connect('tapered_col_cost_rate', ['base.tapered_col_cost_rate', 'ball.tapered_col_cost_rate'])
        self.connect('outfitting_cost_rate', ['base.outfitting_cost_rate', 'ball.outfitting_cost_rate'])

        self.connect('pontoon_outer_diameter', 'pon.pontoon_outer_diameter')
        self.connect('pontoon_wall_thickness', 'pon.pontoon_wall_thickness')
        self.connect('outer_cross_pontoons', 'pon.outer_cross_pontoons')
        self.connect('cross_attachment_pontoons', 'pon.cross_attachment_pontoons')
        self.connect('lower_attachment_pontoons', 'pon.lower_attachment_pontoons')
        self.connect('upper_attachment_pontoons', 'pon.upper_attachment_pontoons')
        self.connect('lower_ring_pontoons', 'pon.lower_ring_pontoons')
        self.connect('upper_ring_pontoons', 'pon.upper_ring_pontoons')
        self.connect('pontoon_cost_rate', 'pon.pontoon_cost_rate')
        self.connect('connection_ratio_max', 'pon.connection_ratio_max')
        self.connect('base_pontoon_attach_lower', 'pon.base_pontoon_attach_lower')
        self.connect('base_pontoon_attach_upper', 'pon.base_pontoon_attach_upper')

        self.connect('number_of_ballast_columns', ['pon.number_of_ballast_columns', 'sm.number_of_ballast_columns'])

        # Link outputs from one model to inputs to another
        self.connect('sg.fairlead_radius', ['mm.fairlead_radius', 'sm.fairlead_radius'])

        self.connect('base.z_full', 'pon.base_z_nodes')
        self.connect('base.d_full', ['pon.base_outer_diameter', 'sg.base_outer_diameter'])
        self.connect('base.t_full', 'pon.base_wall_thickness')

        self.connect('ball.z_full', ['sg.ballast_z_nodes', 'pon.ballast_z_nodes'])
        self.connect('ball.d_full', ['pon.ballast_outer_diameter', 'sg.ballast_outer_diameter'])
        self.connect('ball.t_full', 'pon.ballast_wall_thickness')

        self.connect('mm.mooring_mass', 'sm.mooring_mass')
        self.connect('mm.mooring_effective_mass', 'sm.mooring_effective_mass')
        self.connect('mm.mooring_cost', 'sm.mooring_cost')
        self.connect('mm.max_offset_restoring_force', 'sm.mooring_surge_restoring_force')
        self.connect('mm.max_heel_restoring_force', 'sm.mooring_pitch_restoring_force')
        
        self.connect('base.z_center_of_gravity', 'sm.base_column_center_of_gravity')
        self.connect('base.z_center_of_buoyancy', 'sm.base_column_center_of_buoyancy')
        self.connect('base.Iwater', 'sm.base_column_Iwaterplane')
        self.connect('base.displaced_volume', ['pon.base_column_displaced_volume', 'sm.base_column_displaced_volume'])
        self.connect('base.total_mass', ['pon.base_column_mass', 'sm.base_column_mass'])
        self.connect('base.total_cost', 'sm.base_column_cost')
        self.connect('base.variable_ballast_interp_mass', 'sm.water_ballast_mass_vector')
        self.connect('base.variable_ballast_interp_zpts', 'sm.water_ballast_zpts_vector')
        self.connect('base.Px', 'sm.base_column_surge_force')
        self.connect('base.z_full', 'sm.base_column_force_points')

        self.connect('ball.z_center_of_gravity', 'sm.ballast_column_center_of_gravity')
        self.connect('ball.z_center_of_buoyancy', 'sm.ballast_column_center_of_buoyancy')
        self.connect('ball.Iwater', 'sm.ballast_column_Iwaterplane')
        self.connect('ball.Awater', 'sm.ballast_column_Awaterplane')
        self.connect('ball.displaced_volume', ['pon.ballast_column_displaced_volume', 'sm.ballast_column_displaced_volume'])
        self.connect('ball.total_mass', ['pon.ballast_column_mass', 'sm.ballast_column_mass'])
        self.connect('ball.total_cost', 'sm.ballast_column_cost')
        self.connect('ball.Px', 'sm.ballast_column_surge_force')
        self.connect('ball.z_full', 'sm.ballast_column_force_points')

        self.connect('pon.pontoon_mass', 'sm.pontoon_mass')
        self.connect('pon.pontoon_cost', 'sm.pontoon_cost')
        self.connect('pon.pontoon_buoyancy', 'sm.pontoon_buoyancy')
        self.connect('pon.pontoon_center_of_buoyancy', 'sm.pontoon_center_of_buoyancy')
        self.connect('pon.pontoon_center_of_gravity', 'sm.pontoon_center_of_gravity')

         # Use complex number finite differences
        typeStr = 'fd'
        formStr = 'central'
        stepVal = 1e-5
        stepStr = 'relative'
        
        self.deriv_options['type'] = typeStr
        self.deriv_options['form'] = formStr
        self.deriv_options['step_size'] = stepVal
        self.deriv_options['step_calc'] = stepStr

