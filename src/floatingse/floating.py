from openmdao.api import Group, IndepVarComp
from column import Column, ColumnGeometry
from substructure import Semi, SemiGeometry
from floating_loading import FloatingLoading
from mapMooring import MapMooring
from towerTransition import TowerTransition
from towerse.tower import TowerLeanSE
import numpy as np
        

class FloatingSE(Group):

    def __init__(self, nSection):
        super(FloatingSE, self).__init__()

        #self.add('geomsys', SubstructureDiscretization(nSection), promotes=['z_system'])
        nFull = 5*nSection+1

        self.add('tow', TowerLeanSE(nSection+1,nFull), promotes=['material_density','hub_height','tower_section_height', 'tower_outer_diameter', 'tower_wall_thickness', 'tower_outfitting_factor', 'tower_buckling_length','min_taper','min_d_to_t','rna_mass','rna_cg','rna_I','tower_mass'])
        
        # Next do base and ballast columns
        # Ballast columns are replicated from same design in the components
        self.add('base', Column(nSection, nFull), promotes=['water_depth','water_density','material_density','E','nu','yield_stress','z0',
                                                              'Uref','zref','shearExp','beta','yaw','Uc','hmax','T','cd_usr','cm','loading',
                                                              'min_taper','min_d_to_t'])
        self.add('ball', Column(nSection, nFull), promotes=['water_depth','water_density','material_density','E','nu','yield_stress','z0',
                                                              'Uref','zref','shearExp','beta','yaw','Uc','hmax','T','cd_usr','cm','loading',
                                                              'min_taper','min_d_to_t'])

        # Add in the connecting truss
        self.add('pon', FloatingLoading(nSection, nFull), promotes=['water_density','material_density','E','G','yield_stress','z0','beta','Uref','zref','shearExp','beta','cd_usr','pontoon_outer_diameter', 'pontoon_wall_thickness', 'outer_cross_pontoons', 'cross_attachment_pontoons', 'lower_attachment_pontoons', 'upper_attachment_pontoons', 'lower_ring_pontoons', 'upper_ring_pontoons', 'pontoon_cost_rate', 'connection_ratio_max', 'base_pontoon_attach_lower', 'base_pontoon_attach_upper','gamma_b','gamma_f','gamma_fatigue','gamma_m','gamma_n','rna_I','rna_cg','rna_force','rna_moment','rna_mass'])


        # Run Semi Geometry for interfaces
        self.add('sg', SemiGeometry(nFull))
        
        # Add in transition to tower
        self.add('tt', TowerTransition(nSection+1, diamFlag=True), promotes=['tower_metric'])

        # Next run MapMooring
        self.add('mm', MapMooring(), promotes=['water_density','water_depth'])
        
        # Run main Semi analysis
        self.add('sm', Semi(nFull), promotes=['water_density','total_cost','total_mass'])

        # Define all input variables from all models
        # SemiGeometry
        self.add('radius_to_ballast_column', IndepVarComp('radius_to_ballast_column', 0.0), promotes=['*'])
        
        self.add('fairlead',                   IndepVarComp('fairlead', 0.0), promotes=['*'])
        self.add('fairlead_offset_from_shell', IndepVarComp('fairlead_offset_from_shell', 0.0), promotes=['*'])

        self.add('base_freeboard',             IndepVarComp('base_freeboard', 0.0), promotes=['*'])
        self.add('base_section_height',        IndepVarComp('base_section_height', np.zeros((nSection,))), promotes=['*'])
        self.add('base_outer_diameter',        IndepVarComp('base_outer_diameter', np.zeros((nSection+1,))), promotes=['*'])
        self.add('base_wall_thickness',        IndepVarComp('base_wall_thickness', np.zeros((nSection+1,))), promotes=['*'])

        self.add('ballast_freeboard',          IndepVarComp('ballast_freeboard', 0.0), promotes=['*'])
        self.add('ballast_section_height',     IndepVarComp('ballast_section_height', np.zeros((nSection,))), promotes=['*'])
        self.add('ballast_outer_diameter',     IndepVarComp('ballast_outer_diameter', np.zeros((nSection+1,))), promotes=['*'])
        self.add('ballast_wall_thickness',     IndepVarComp('ballast_wall_thickness', np.zeros((nSection+1,))), promotes=['*'])

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
        
        self.add('base_stiffener_web_height',       IndepVarComp('base_stiffener_web_height', np.zeros((nSection,))), promotes=['*'])
        self.add('base_stiffener_web_thickness',    IndepVarComp('base_stiffener_web_thickness', np.zeros((nSection,))), promotes=['*'])
        self.add('base_stiffener_flange_width',     IndepVarComp('base_stiffener_flange_width', np.zeros((nSection,))), promotes=['*'])
        self.add('base_stiffener_flange_thickness', IndepVarComp('base_stiffener_flange_thickness', np.zeros((nSection,))), promotes=['*'])
        self.add('base_stiffener_spacing',          IndepVarComp('base_stiffener_spacing', np.zeros((nSection,))), promotes=['*'])
        self.add('base_bulkhead_nodes',             IndepVarComp('base_bulkhead_nodes', [False]*(nSection+1), pass_by_obj=True ), promotes=['*'])
        self.add('base_permanent_ballast_height',   IndepVarComp('base_permanent_ballast_height', 0.0), promotes=['*'])

        self.add('ballast_stiffener_web_height',       IndepVarComp('ballast_stiffener_web_height', np.zeros((nSection,))), promotes=['*'])
        self.add('ballast_stiffener_web_thickness',    IndepVarComp('ballast_stiffener_web_thickness', np.zeros((nSection,))), promotes=['*'])
        self.add('ballast_stiffener_flange_width',     IndepVarComp('ballast_stiffener_flange_width', np.zeros((nSection,))), promotes=['*'])
        self.add('ballast_stiffener_flange_thickness', IndepVarComp('ballast_stiffener_flange_thickness', np.zeros((nSection,))), promotes=['*'])
        self.add('ballast_stiffener_spacing',          IndepVarComp('ballast_stiffener_spacing', np.zeros((nSection,))), promotes=['*'])
        self.add('ballast_bulkhead_nodes',             IndepVarComp('ballast_bulkhead_nodes', [False]*(nSection+1), pass_by_obj=True ), promotes=['*'])
        self.add('ballast_permanent_ballast_height',   IndepVarComp('ballast_permanent_ballast_height', 0.0), promotes=['*'])

        self.add('bulkhead_mass_factor',       IndepVarComp('bulkhead_mass_factor', 0.0), promotes=['*'])
        self.add('ring_mass_factor',           IndepVarComp('ring_mass_factor', 0.0), promotes=['*'])
        self.add('shell_mass_factor',          IndepVarComp('shell_mass_factor', 0.0), promotes=['*'])
        self.add('spar_mass_factor',           IndepVarComp('spar_mass_factor', 0.0), promotes=['*'])
        self.add('outfitting_mass_fraction',   IndepVarComp('outfitting_mass_fraction', 0.0), promotes=['*'])
        self.add('ballast_cost_rate',          IndepVarComp('ballast_cost_rate', 0.0), promotes=['*'])
        self.add('tapered_col_cost_rate',      IndepVarComp('tapered_col_cost_rate', 0.0), promotes=['*'])
        self.add('outfitting_cost_rate',       IndepVarComp('outfitting_cost_rate', 0.0), promotes=['*'])
        self.add('loading',                    IndepVarComp('loading', val='hydrostatic', pass_by_obj=True), promotes=['*'])
        
        # Pontoons
        #self.add('G',                          IndepVarComp('G', 0.0), promotes=['*'])
        self.add('number_of_ballast_columns',  IndepVarComp('number_of_ballast_columns', 0, pass_by_obj=True), promotes=['*'])

        # Connect all input variables from all models
        self.connect('radius_to_ballast_column', ['sg.radius_to_ballast_column', 'pon.radius_to_ballast_column', 'sm.radius_to_ballast_column'])

        self.connect('base_freeboard', ['base.freeboard', 'sm.base_freeboard'])
        self.connect('base_section_height', 'base.section_height')
        self.connect('base_outer_diameter', ['base.diameter','tt.base_metric'])
        self.connect('base_wall_thickness', 'base.wall_thickness')

        self.connect('tow.d_full', 'pon.windLoads.d')
        self.connect('tow.t_full', 'pon.tower_t_full')
        self.connect('tow.z_full', 'pon.wind.z')
        self.connect('tower_outer_diameter','tower_metric')
        self.connect('tow.cm.mass','pon.tower_mass')
        self.connect('tower_buckling_length','pon.tower_buckling_length')
        self.connect('tow.turbine_mass','base.stack_mass_in')
        self.connect('tow.tower_center_of_mass','pon.tower_center_of_mass')
        
        self.connect('ballast_freeboard', 'ball.freeboard')
        self.connect('ballast_section_height', 'ball.section_height')
        self.connect('ballast_outer_diameter', 'ball.diameter')
        self.connect('ballast_wall_thickness', 'ball.wall_thickness')

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
        self.connect('base.windLoads.rho',['ball.windLoads.rho','pon.windLoads.rho'])
        self.connect('base.windLoads.mu',['ball.windLoads.mu','pon.windLoads.mu'])
        self.connect('base.waveLoads.mu','ball.waveLoads.mu')

        self.connect('permanent_ballast_density', ['base.permanent_ballast_density', 'ball.permanent_ballast_density'])
        
        self.connect('base_stiffener_web_height', 'base.stiffener_web_height')
        self.connect('base_stiffener_web_thickness', 'base.stiffener_web_thickness')
        self.connect('base_stiffener_flange_width', 'base.stiffener_flange_width')
        self.connect('base_stiffener_flange_thickness', 'base.stiffener_flange_thickness')
        self.connect('base_stiffener_spacing', 'base.stiffener_spacing')
        self.connect('base_bulkhead_nodes', 'base.bulkhead_nodes')
        self.connect('base_permanent_ballast_height', 'base.permanent_ballast_height')
        
        self.connect('ballast_stiffener_web_height', 'ball.stiffener_web_height')
        self.connect('ballast_stiffener_web_thickness', 'ball.stiffener_web_thickness')
        self.connect('ballast_stiffener_flange_width', 'ball.stiffener_flange_width')
        self.connect('ballast_stiffener_flange_thickness', 'ball.stiffener_flange_thickness')
        self.connect('ballast_stiffener_spacing', 'ball.stiffener_spacing')
        self.connect('ballast_bulkhead_nodes', 'ball.bulkhead_nodes')
        self.connect('ballast_permanent_ballast_height', 'ball.permanent_ballast_height')
        
        self.connect('bulkhead_mass_factor', ['base.bulkhead_mass_factor', 'ball.bulkhead_mass_factor'])
        self.connect('ring_mass_factor', ['base.ring_mass_factor', 'ball.ring_mass_factor'])
        self.connect('shell_mass_factor', ['base.cyl_mass.outfitting_factor', 'ball.cyl_mass.outfitting_factor'])
        self.connect('spar_mass_factor', ['base.spar_mass_factor', 'ball.spar_mass_factor'])
        self.connect('outfitting_mass_fraction', ['base.outfitting_mass_fraction', 'ball.outfitting_mass_fraction'])
        self.connect('ballast_cost_rate', ['base.ballast_cost_rate', 'ball.ballast_cost_rate'])
        self.connect('tapered_col_cost_rate', ['base.tapered_col_cost_rate', 'ball.tapered_col_cost_rate'])
        self.connect('outfitting_cost_rate', ['base.outfitting_cost_rate', 'ball.outfitting_cost_rate'])

        self.connect('number_of_ballast_columns', ['pon.number_of_ballast_columns', 'sm.number_of_ballast_columns'])

        # Link outputs from one model to inputs to another
        self.connect('sg.fairlead_radius', ['mm.fairlead_radius', 'sm.fairlead_radius'])

        self.connect('base.z_full', 'pon.base_z_full')
        self.connect('base.d_full', ['pon.base_d_full', 'sg.base_outer_diameter'])
        self.connect('base.t_full', 'pon.base_t_full')

        self.connect('ball.z_full', ['sg.ballast_z_nodes', 'pon.ballast_z_full'])
        self.connect('ball.d_full', ['pon.ballast_d_full', 'sg.ballast_outer_diameter'])
        self.connect('ball.t_full', 'pon.ballast_t_full')

        self.connect('mm.mooring_mass', 'sm.mooring_mass')
        self.connect('mm.mooring_effective_mass', 'sm.mooring_effective_mass')
        self.connect('mm.mooring_cost', 'sm.mooring_cost')
        self.connect('mm.max_offset_restoring_force', 'sm.mooring_surge_restoring_force')
        self.connect('mm.max_heel_restoring_force', 'sm.mooring_pitch_restoring_force')
        
        self.connect('base.z_center_of_mass', 'pon.base_column_center_of_mass')
        self.connect('base.z_center_of_buoyancy', 'pon.base_column_center_of_buoyancy')
        self.connect('base.Iwater', 'sm.base_column_Iwaterplane')
        self.connect('base.displaced_volume', 'pon.base_column_displaced_volume')
        self.connect('base.total_mass', 'pon.base_column_mass')
        self.connect('base.total_cost', 'sm.base_column_cost')
        self.connect('base.variable_ballast_interp_mass', 'sm.water_ballast_mass_vector')
        self.connect('base.variable_ballast_interp_zpts', 'sm.water_ballast_zpts_vector')
        self.connect('base.Px', 'pon.base_column_Px')
        self.connect('base.Py', 'pon.base_column_Py')
        self.connect('base.Pz', 'pon.base_column_Pz')
        self.connect('base.qdyn', 'pon.base_column_qdyn')

        self.connect('ball.z_center_of_mass', 'pon.ballast_column_center_of_mass')
        self.connect('ball.z_center_of_buoyancy', 'pon.ballast_column_center_of_buoyancy')
        self.connect('ball.Iwater', 'sm.ballast_column_Iwaterplane')
        self.connect('ball.Awater', 'sm.ballast_column_Awaterplane')
        self.connect('ball.displaced_volume', 'pon.ballast_column_displaced_volume')
        self.connect('ball.total_mass', 'pon.ballast_column_mass')
        self.connect('ball.total_cost', 'sm.ballast_column_cost')
        self.connect('ball.Px', 'pon.ballast_column_Px')
        self.connect('ball.Py', 'pon.ballast_column_Py')
        self.connect('ball.Pz', 'pon.ballast_column_Pz')
        self.connect('ball.qdyn', 'pon.ballast_column_qdyn')

        self.connect('pon.structural_mass', 'sm.structural_mass')
        self.connect('pon.center_of_mass', 'sm.structure_center_of_mass')
        self.connect('pon.total_displacement', 'sm.total_displacement')
        self.connect('pon.total_force', 'sm.total_force')
        self.connect('pon.total_moment', 'sm.total_moment')
        self.connect('pon.pontoon_cost', 'sm.pontoon_cost')

         # Use complex number finite differences
        typeStr = 'fd'
        formStr = 'central'
        stepVal = 1e-5
        stepStr = 'relative'
        
        self.deriv_options['type'] = typeStr
        self.deriv_options['form'] = formStr
        self.deriv_options['step_size'] = stepVal
        self.deriv_options['step_calc'] = stepStr

