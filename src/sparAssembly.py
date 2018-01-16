from openmdao.api import Group, IndepVarComp
from cylinder import Cylinder
from spar import Spar
from sparGeometry import SparGeometry
from floatingInstance import NSECTIONS
from mapMooring import MapMooring
from turbine import Turbine
import numpy as np

class SparAssembly(Group):

    def __init__(self):
        super(SparAssembly, self).__init__()

        # Run Spar Geometry component first
        self.add('sg', SparGeometry())

        # Run Turbine setup second
        self.add('turb', Turbine())

        # Next run MapMooring
        self.add('mm', MapMooring())

        # Next do ballast cylind
        self.add('cyl', Cylinder())

        # Run main Spar analysis
        self.add('sp', Spar())

        # Define all input variables from all models
        # SparGeometry
        self.add('water_depth',                IndepVarComp('x', 0.0))
        self.add('freeboard',                  IndepVarComp('x', 0.0))
        self.add('fairlead',                   IndepVarComp('x', 0.0))
        self.add('section_height',             IndepVarComp('x', np.zeros((NSECTIONS,))))
        self.add('outer_radius',               IndepVarComp('x', np.zeros((NSECTIONS+1,))))
        self.add('wall_thickness',             IndepVarComp('x', np.zeros((NSECTIONS+1,))))
        self.add('fairlead_offset_from_shell', IndepVarComp('x', 0.0))
        self.add('tower_base_radius',          IndepVarComp('x', 0.0))

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
        self.add('anchor_type',                IndepVarComp('x', 'pile', pass_by_obj=True))
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
        
        self.add('stiffener_web_height',       IndepVarComp('x', np.zeros((NSECTIONS,))))
        self.add('stiffener_web_thickness',    IndepVarComp('x', np.zeros((NSECTIONS,))))
        self.add('stiffener_flange_width',     IndepVarComp('x', np.zeros((NSECTIONS,))))
        self.add('stiffener_flange_thickness', IndepVarComp('x', np.zeros((NSECTIONS,))))
        self.add('stiffener_spacing',          IndepVarComp('x', np.zeros((NSECTIONS,))))
        
        self.add('bulkhead_nodes',             IndepVarComp('x', [False]*(NSECTIONS+1), pass_by_obj=True ))
        self.add('permanent_ballast_height',   IndepVarComp('x', 0.0))
        self.add('bulkhead_mass_factor',       IndepVarComp('x', 0.0))
        self.add('ring_mass_factor',           IndepVarComp('x', 0.0))
        self.add('shell_mass_factor',          IndepVarComp('x', 0.0))
        self.add('spar_mass_factor',           IndepVarComp('x', 0.0))
        self.add('outfitting_mass_fraction',   IndepVarComp('x', 0.0))
        self.add('ballast_cost_rate',          IndepVarComp('x', 0.0))
        self.add('tapered_col_cost_rate',      IndepVarComp('x', 0.0))
        self.add('outfitting_cost_rate',       IndepVarComp('x', 0.0))

        # Connect all input variables from all models
        self.connect('water_depth.x', ['sg.water_depth', 'mm.water_depth', 'cyl.water_depth'])
        self.connect('freeboard.x', ['sg.freeboard', 'turb.freeboard'])
        self.connect('fairlead.x', ['sg.fairlead', 'mm.fairlead','sp.fairlead'])
        self.connect('section_height.x', ['sg.section_height', 'cyl.section_height'])
        self.connect('outer_radius.x', ['sg.outer_radius', 'cyl.outer_radius'])
        self.connect('wall_thickness.x', ['sg.wall_thickness', 'cyl.wall_thickness'])
        self.connect('fairlead_offset_from_shell.x', 'sg.fairlead_offset_from_shell')
        self.connect('tower_base_radius.x', 'sg.tower_base_radius')

        self.connect('rna_mass.x', 'turb.rna_mass')
        self.connect('rna_center_of_gravity.x', 'turb.rna_center_of_gravity')
        self.connect('rna_center_of_gravity_x.x', 'turb.rna_center_of_gravity_x')
        self.connect('rna_wind_force.x', 'turb.rna_wind_force')
        self.connect('tower_mass.x', 'turb.tower_mass')
        self.connect('tower_center_of_gravity.x', 'turb.tower_center_of_gravity')
        self.connect('tower_wind_force.x', 'turb.tower_wind_force')

        self.connect('water_density.x', ['mm.water_density', 'cyl.water_density', 'sp.water_density'])
        self.connect('scope_ratio.x', 'mm.scope_ratio')
        self.connect('anchor_radius.x', 'mm.anchor_radius')
        self.connect('mooring_diameter.x', 'mm.mooring_diameter')
        self.connect('number_of_mooring_lines.x', 'mm.number_of_mooring_lines')
        self.connect('mooring_type.x', 'mm.mooring_type')
        self.connect('anchor_type.x', 'mm.anchor_type')
        self.connect('max_offset.x', 'mm.max_offset')
        self.connect('mooring_cost_rate.x', 'mm.mooring_cost_rate')
        
        self.connect('air_density.x', 'cyl.air_density')
        self.connect('air_viscosity.x', 'cyl.air_viscosity')
        self.connect('water_viscosity.x', 'cyl.water_viscosity')
        self.connect('wave_height.x', 'cyl.wave_height')
        self.connect('wave_period.x', 'cyl.wave_period')
        self.connect('wind_reference_speed.x', 'cyl.wind_reference_speed')
        self.connect('wind_reference_height.x', 'cyl.wind_reference_height')
        self.connect('alpha.x', 'cyl.alpha')
        self.connect('morison_mass_coefficient.x', 'cyl.morison_mass_coefficient')
        self.connect('material_density.x', 'cyl.material_density')
        self.connect('E.x', 'cyl.E')
        self.connect('nu.x', 'cyl.nu')
        self.connect('yield_stress.x', 'cyl.yield_stress')
        self.connect('permanent_ballast_density.x', 'cyl.permanent_ballast_density')
        self.connect('stiffener_web_height.x', 'cyl.stiffener_web_height')
        self.connect('stiffener_web_thickness.x', 'cyl.stiffener_web_thickness')
        self.connect('stiffener_flange_width.x', 'cyl.stiffener_flange_width')
        self.connect('stiffener_flange_thickness.x', 'cyl.stiffener_flange_thickness')
        self.connect('stiffener_spacing.x', 'cyl.stiffener_spacing')
        self.connect('bulkhead_nodes.x', 'cyl.bulkhead_nodes')
        self.connect('permanent_ballast_height.x', 'cyl.permanent_ballast_height')
        self.connect('bulkhead_mass_factor.x', 'cyl.bulkhead_mass_factor')
        self.connect('ring_mass_factor.x', 'cyl.ring_mass_factor')
        self.connect('shell_mass_factor.x', 'cyl.shell_mass_factor')
        self.connect('spar_mass_factor.x', 'cyl.spar_mass_factor')
        self.connect('outfitting_mass_fraction.x', 'cyl.outfitting_mass_fraction')
        self.connect('ballast_cost_rate.x', 'cyl.ballast_cost_rate')
        self.connect('tapered_col_cost_rate.x', 'cyl.tapered_col_cost_rate')
        self.connect('outfitting_cost_rate.x', 'cyl.outfitting_cost_rate')
        
        # Link outputs from one model to inputs to another
        self.connect('sg.fairlead_radius', 'mm.fairlead_radius')
        self.connect('sg.z_nodes', 'cyl.z_nodes')
        self.connect('sg.z_section', 'cyl.z_section')
        
        self.connect('turb.total_mass', ['cyl.stack_mass_in', 'sp.turbine_mass'])
        self.connect('turb.z_center_of_gravity', 'sp.turbine_center_of_gravity')
        self.connect('turb.surge_force', 'sp.turbine_surge_force')
        self.connect('turb.force_points', 'sp.turbine_force_points')
        self.connect('turb.pitch_moment', 'sp.turbine_pitch_moment')
        
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
        formStr = 'forward' #'central'
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

        self.cyl.deriv_options['type'] = typeStr
        self.cyl.deriv_options['form'] = formStr
        self.cyl.deriv_options['step_size'] = stepVal
        self.cyl.deriv_options['step_calc'] = stepStr

        self.turb.deriv_options['type'] = typeStr
        self.turb.deriv_options['form'] = formStr
        self.turb.deriv_options['step_size'] = stepVal
        self.turb.deriv_options['step_calc'] = stepStr

        self.sp.deriv_options['type'] = typeStr
        self.sp.deriv_options['form'] = formStr
        self.sp.deriv_options['step_size'] = stepVal
        self.sp.deriv_options['step_calc'] = stepStr

