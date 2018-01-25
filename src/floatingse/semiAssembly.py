from openmdao.api import Group, IndepVarComp, DirectSolver, ScipyGMRES, Newton, NLGaussSeidel, Brent, RunOnce
from cylinder import Cylinder
from semi import Semi
from semiPontoon import SemiPontoon
from semiGeometry import SemiGeometry
from mapMooring import MapMooring, Anchor
from turbine import Turbine
import numpy as np

class SemiAssembly(Group):

    def __init__(self, nSection):
        super(SemiAssembly, self).__init__()

        # Run Spar Geometry component first
        self.add('sg', SemiGeometry(nSection))

        # Run Turbine setup second
        self.add('turb', Turbine())

        # Next run MapMooring
        self.add('mm', MapMooring())

        # Next do base and ballast cylinders
        # Ballast cylinders are replicated from same design in the components
        self.add('base', Cylinder(nSection))
        self.add('ball', Cylinder(nSection))

        # Add in the connecting truss
        self.add('pon', SemiPontoon(nSection))
        
        # Run main Semi analysis
        self.add('sm', Semi(nSection))

        # Define all input variables from all models
        # SemiGeometry
        self.add('radius_to_ballast_cylinder', IndepVarComp('x', 0.0))
        
        self.add('water_depth',                IndepVarComp('x', 0.0))
        self.add('fairlead',                   IndepVarComp('x', 0.0))
        self.add('fairlead_offset_from_shell', IndepVarComp('x', 0.0))
        self.add('tower_base_radius',          IndepVarComp('x', 0.0))

        self.add('freeboard_base',             IndepVarComp('x', 0.0))
        self.add('section_height_base',        IndepVarComp('x', np.zeros((nSection,))))
        self.add('outer_radius_base',          IndepVarComp('x', np.zeros((nSection+1,))))
        self.add('wall_thickness_base',        IndepVarComp('x', np.zeros((nSection+1,))))

        self.add('freeboard_ballast',          IndepVarComp('x', 0.0))
        self.add('section_height_ballast',     IndepVarComp('x', np.zeros((nSection,))))
        self.add('outer_radius_ballast',       IndepVarComp('x', np.zeros((nSection+1,))))
        self.add('wall_thickness_ballast',     IndepVarComp('x', np.zeros((nSection+1,))))

        # Turbine
        self.add('rna_mass',                   IndepVarComp('x', 0.0))
        self.add('rna_center_of_gravity',      IndepVarComp('x', 0.0))
        self.add('rna_center_of_gravity_x',    IndepVarComp('x', 0.0))
        self.add('rna_wind_force',             IndepVarComp('x', 0.0))
        self.add('tower_mass',                 IndepVarComp('x', 0.0))
        self.add('tower_center_of_gravity',    IndepVarComp('x', 0.0))
        self.add('tower_wind_force',           IndepVarComp('x', 0.0))

        # Mooring
        self.add('water_density',              IndepVarComp('x', 0.0))
        self.add('scope_ratio',                IndepVarComp('x', 0.0))
        self.add('anchor_radius',              IndepVarComp('x', 0.0))
        self.add('mooring_diameter',           IndepVarComp('x', 0.0))
        self.add('number_of_mooring_lines',    IndepVarComp('x', 0, pass_by_obj=True))
        self.add('mooring_type',               IndepVarComp('x', 'chain', pass_by_obj=True))
        self.add('anchor_type',                IndepVarComp('x', Anchor['SUCTIONPILE'], pass_by_obj=True))
        self.add('drag_embedment_extra_length',IndepVarComp('x', 0.0))
        self.add('max_offset',                 IndepVarComp('x', 0.0))
        self.add('mooring_cost_rate',          IndepVarComp('x', 0.0))

        # Cylinder
        self.add('air_density',                IndepVarComp('x', 0.0))
        self.add('air_viscosity',              IndepVarComp('x', 0.0))
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
        
        self.add('stiffener_web_height_base',       IndepVarComp('x', np.zeros((nSection,))))
        self.add('stiffener_web_thickness_base',    IndepVarComp('x', np.zeros((nSection,))))
        self.add('stiffener_flange_width_base',     IndepVarComp('x', np.zeros((nSection,))))
        self.add('stiffener_flange_thickness_base', IndepVarComp('x', np.zeros((nSection,))))
        self.add('stiffener_spacing_base',          IndepVarComp('x', np.zeros((nSection,))))
        self.add('bulkhead_nodes_base',             IndepVarComp('x', [False]*(nSection+1), pass_by_obj=True ))
        self.add('permanent_ballast_height_base',   IndepVarComp('x', 0.0))

        self.add('stiffener_web_height_ballast',       IndepVarComp('x', np.zeros((nSection,))))
        self.add('stiffener_web_thickness_ballast',    IndepVarComp('x', np.zeros((nSection,))))
        self.add('stiffener_flange_width_ballast',     IndepVarComp('x', np.zeros((nSection,))))
        self.add('stiffener_flange_thickness_ballast', IndepVarComp('x', np.zeros((nSection,))))
        self.add('stiffener_spacing_ballast',          IndepVarComp('x', np.zeros((nSection,))))
        self.add('bulkhead_nodes_ballast',             IndepVarComp('x', [False]*(nSection+1), pass_by_obj=True ))
        self.add('permanent_ballast_height_ballast',   IndepVarComp('x', 0.0))

        self.add('bulkhead_mass_factor',       IndepVarComp('x', 0.0))
        self.add('ring_mass_factor',           IndepVarComp('x', 0.0))
        self.add('shell_mass_factor',          IndepVarComp('x', 0.0))
        self.add('spar_mass_factor',           IndepVarComp('x', 0.0))
        self.add('outfitting_mass_fraction',   IndepVarComp('x', 0.0))
        self.add('ballast_cost_rate',          IndepVarComp('x', 0.0))
        self.add('tapered_col_cost_rate',      IndepVarComp('x', 0.0))
        self.add('outfitting_cost_rate',       IndepVarComp('x', 0.0))

        # Pontoons
        self.add('G',                          IndepVarComp('x', 0.0))
        self.add('number_of_ballast_columns',  IndepVarComp('x', 0, pass_by_obj=True))
        self.add('outer_pontoon_radius',       IndepVarComp('x', 0.0))
        self.add('inner_pontoon_radius',       IndepVarComp('x', 0.0))
        self.add('cross_attachment_pontoons',  IndepVarComp('x', True, pass_by_obj=True))
        self.add('lower_attachment_pontoons',  IndepVarComp('x', True, pass_by_obj=True))
        self.add('upper_attachment_pontoons',  IndepVarComp('x', True, pass_by_obj=True))
        self.add('lower_ring_pontoons',        IndepVarComp('x', True, pass_by_obj=True))
        self.add('upper_ring_pontoons',        IndepVarComp('x', True, pass_by_obj=True))
        self.add('pontoon_cost_rate',          IndepVarComp('x', 0.0))

        # Connect all input variables from all models
        self.connect('water_depth.x', ['sg.water_depth', 'mm.water_depth', 'base.water_depth', 'ball.water_depth'])
        self.connect('radius_to_ballast_cylinder.x', ['sg.radius_to_ballast_cylinder', 'pon.radius_to_ballast_cylinder', 'sm.radius_to_ballast_cylinder'])

        self.connect('freeboard_base.x', ['sg.base_freeboard', 'turb.freeboard'])
        self.connect('section_height_base.x', ['sg.base_section_height', 'base.section_height'])
        self.connect('outer_radius_base.x', ['sg.base_outer_radius', 'base.outer_radius', 'pon.base_outer_radius', ])
        self.connect('wall_thickness_base.x', ['sg.base_wall_thickness', 'base.wall_thickness', 'pon.base_wall_thickness', ])

        self.connect('freeboard_ballast.x', 'sg.ballast_freeboard')
        self.connect('section_height_ballast.x', ['sg.ballast_section_height', 'ball.section_height'])
        self.connect('outer_radius_ballast.x', ['sg.ballast_outer_radius', 'ball.outer_radius', 'pon.ballast_outer_radius'])
        self.connect('wall_thickness_ballast.x', ['sg.ballast_wall_thickness', 'ball.wall_thickness', 'pon.ballast_wall_thickness'])

        self.connect('fairlead.x', ['sg.fairlead', 'mm.fairlead','sm.fairlead'])
        self.connect('fairlead_offset_from_shell.x', 'sg.fairlead_offset_from_shell')
        self.connect('tower_base_radius.x', ['sg.tower_base_radius', 'pon.tower_base_radius'])

        self.connect('rna_mass.x', ['turb.rna_mass', 'pon.rna_mass'])
        self.connect('rna_center_of_gravity.x', 'turb.rna_center_of_gravity')
        self.connect('rna_center_of_gravity_x.x', ['turb.rna_center_of_gravity_x', 'pon.rna_center_of_gravity_x'])
        self.connect('rna_wind_force.x', 'turb.rna_wind_force')
        self.connect('tower_mass.x', ['turb.tower_mass', 'pon.tower_mass'])
        self.connect('tower_center_of_gravity.x', 'turb.tower_center_of_gravity')
        self.connect('tower_wind_force.x', 'turb.tower_wind_force')

        self.connect('water_density.x', ['mm.water_density', 'base.water_density', 'ball.water_density', 'pon.water_density', 'sm.water_density'])
        self.connect('scope_ratio.x', 'mm.scope_ratio')
        self.connect('anchor_radius.x', 'mm.anchor_radius')
        self.connect('mooring_diameter.x', 'mm.mooring_diameter')
        self.connect('number_of_mooring_lines.x', 'mm.number_of_mooring_lines')
        self.connect('mooring_type.x', 'mm.mooring_type')
        self.connect('anchor_type.x', 'mm.anchor_type')
        self.connect('drag_embedment_extra_length.x', 'mm.drag_embedment_extra_length')
        self.connect('max_offset.x', 'mm.max_offset')
        self.connect('mooring_cost_rate.x', 'mm.mooring_cost_rate')
        
        self.connect('air_density.x', ['base.air_density', 'ball.air_density'])
        self.connect('air_viscosity.x', ['base.air_viscosity', 'ball.air_viscosity'])
        self.connect('water_viscosity.x', ['base.water_viscosity', 'ball.water_viscosity'])
        self.connect('wave_height.x', ['base.wave_height', 'ball.wave_height'])
        self.connect('wave_period.x', ['base.wave_period', 'ball.wave_period'])
        self.connect('wind_reference_speed.x', ['base.wind_reference_speed', 'ball.wind_reference_speed'])
        self.connect('wind_reference_height.x', ['base.wind_reference_height', 'ball.wind_reference_height'])
        self.connect('alpha.x', ['base.alpha', 'ball.alpha'])
        self.connect('morison_mass_coefficient.x', ['base.morison_mass_coefficient', 'ball.morison_mass_coefficient'])
        self.connect('material_density.x', ['base.material_density', 'ball.material_density', 'pon.material_density'])
        self.connect('E.x', ['base.E', 'ball.E', 'pon.E'])
        self.connect('G.x', 'pon.G')
        self.connect('nu.x', ['base.nu', 'ball.nu'])
        self.connect('yield_stress.x', ['base.yield_stress', 'ball.yield_stress', 'pon.yield_stress'])
        self.connect('permanent_ballast_density.x', ['base.permanent_ballast_density', 'ball.permanent_ballast_density'])
        
        self.connect('stiffener_web_height_base.x', 'base.stiffener_web_height')
        self.connect('stiffener_web_thickness_base.x', 'base.stiffener_web_thickness')
        self.connect('stiffener_flange_width_base.x', 'base.stiffener_flange_width')
        self.connect('stiffener_flange_thickness_base.x', 'base.stiffener_flange_thickness')
        self.connect('stiffener_spacing_base.x', 'base.stiffener_spacing')
        self.connect('bulkhead_nodes_base.x', 'base.bulkhead_nodes')
        self.connect('permanent_ballast_height_base.x', 'base.permanent_ballast_height')
        
        self.connect('stiffener_web_height_ballast.x', 'ball.stiffener_web_height')
        self.connect('stiffener_web_thickness_ballast.x', 'ball.stiffener_web_thickness')
        self.connect('stiffener_flange_width_ballast.x', 'ball.stiffener_flange_width')
        self.connect('stiffener_flange_thickness_ballast.x', 'ball.stiffener_flange_thickness')
        self.connect('stiffener_spacing_ballast.x', 'ball.stiffener_spacing')
        self.connect('bulkhead_nodes_ballast.x', 'ball.bulkhead_nodes')
        self.connect('permanent_ballast_height_ballast.x', 'ball.permanent_ballast_height')
        
        self.connect('bulkhead_mass_factor.x', ['base.bulkhead_mass_factor', 'ball.bulkhead_mass_factor'])
        self.connect('ring_mass_factor.x', ['base.ring_mass_factor', 'ball.ring_mass_factor'])
        self.connect('shell_mass_factor.x', ['base.shell_mass_factor', 'ball.shell_mass_factor'])
        self.connect('spar_mass_factor.x', ['base.spar_mass_factor', 'ball.spar_mass_factor'])
        self.connect('outfitting_mass_fraction.x', ['base.outfitting_mass_fraction', 'ball.outfitting_mass_fraction'])
        self.connect('ballast_cost_rate.x', ['base.ballast_cost_rate', 'ball.ballast_cost_rate'])
        self.connect('tapered_col_cost_rate.x', ['base.tapered_col_cost_rate', 'ball.tapered_col_cost_rate'])
        self.connect('outfitting_cost_rate.x', ['base.outfitting_cost_rate', 'ball.outfitting_cost_rate'])

        self.connect('outer_pontoon_radius.x', 'pon.outer_pontoon_radius')
        self.connect('inner_pontoon_radius.x', 'pon.inner_pontoon_radius')
        self.connect('cross_attachment_pontoons.x', 'pon.cross_attachment_pontoons')
        self.connect('lower_attachment_pontoons.x', 'pon.lower_attachment_pontoons')
        self.connect('upper_attachment_pontoons.x', 'pon.upper_attachment_pontoons')
        self.connect('lower_ring_pontoons.x', 'pon.lower_ring_pontoons')
        self.connect('upper_ring_pontoons.x', 'pon.upper_ring_pontoons')
        self.connect('pontoon_cost_rate.x', 'pon.pontoon_cost_rate')

        self.connect('number_of_ballast_columns.x', ['pon.number_of_ballast_cylinders', 'sm.number_of_ballast_cylinders'])

        # Link outputs from one model to inputs to another
        self.connect('sg.fairlead_radius', 'mm.fairlead_radius')
        self.connect('sg.base_z_nodes', ['base.z_nodes', 'pon.base_z_nodes'])
        self.connect('sg.base_z_section', 'base.z_section')
        self.connect('sg.ballast_z_nodes', ['ball.z_nodes', 'pon.ballast_z_nodes'])
        self.connect('sg.ballast_z_section', 'ball.z_section')
        
        self.connect('turb.total_mass', ['base.stack_mass_in', 'sm.turbine_mass'])
        self.connect('turb.z_center_of_gravity', 'sm.turbine_center_of_gravity')
        self.connect('turb.surge_force', ['pon.turbine_surge_force', 'sm.turbine_surge_force'])
        self.connect('turb.force_points', ['pon.turbine_force_points', 'sm.turbine_force_points'])
        self.connect('turb.pitch_moment', 'sm.turbine_pitch_moment')
        
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

         # Use complex number finite differences
        typeStr = 'fd'
        formStr = 'central'
        stepVal = 1e-5
        stepStr = 'relative'
        
        self.deriv_options['type'] = typeStr
        self.deriv_options['form'] = formStr
        self.deriv_options['step_size'] = stepVal
        self.deriv_options['step_calc'] = stepStr

        self.sg.deriv_options['type'] = typeStr
        self.sg.deriv_options['form'] = formStr
        self.sg.deriv_options['step_size'] = stepVal
        self.sg.deriv_options['step_calc'] = stepStr

        self.mm.deriv_options['type'] = typeStr
        self.mm.deriv_options['form'] = formStr
        self.mm.deriv_options['step_size'] = stepVal
        self.mm.deriv_options['step_calc'] = stepStr

        self.base.deriv_options['type'] = typeStr
        self.base.deriv_options['form'] = formStr
        self.base.deriv_options['step_size'] = stepVal
        self.base.deriv_options['step_calc'] = stepStr

        self.ball.deriv_options['type'] = typeStr
        self.ball.deriv_options['form'] = formStr
        self.ball.deriv_options['step_size'] = stepVal
        self.ball.deriv_options['step_calc'] = stepStr

        self.turb.deriv_options['type'] = typeStr
        self.turb.deriv_options['form'] = formStr
        self.turb.deriv_options['step_size'] = stepVal
        self.turb.deriv_options['step_calc'] = stepStr

        self.pon.deriv_options['type'] = typeStr
        self.pon.deriv_options['form'] = formStr
        self.pon.deriv_options['step_size'] = stepVal
        self.pon.deriv_options['step_calc'] = stepStr

        self.sm.deriv_options['type'] = typeStr
        self.sm.deriv_options['form'] = formStr
        self.sm.deriv_options['step_size'] = stepVal
        self.sm.deriv_options['step_calc'] = stepStr

