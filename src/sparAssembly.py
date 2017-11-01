from openmdao.api import Group, IndepVarComp
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
        self.add('number_of_mooring_lines',    IndepVarComp('x', 0, pass_by_obj=True))
        self.add('mooring_type',               IndepVarComp('x', 'chain', pass_by_obj=True))
        self.add('anchor_type',                IndepVarComp('x', 'pile', pass_by_obj=True))
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

        # Use complex number finite differences
        self.deriv_options['type'] = 'fd'
        #self.deriv_options['type'] = 'cs'
        self.deriv_options['form'] = 'central'
        self.deriv_options['step_size'] = 1e-4
        self.deriv_options['step_calc'] = 'relative'

