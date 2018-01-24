from openmdao.api import Group, IndepVarComp
from sparGeometry import SparGeometry
from floatingInstance import NSECTIONS
from mapMooring import MapMooring, Anchor
import numpy as np

class MooringAssembly(Group):

    def __init__(self):
        super(MooringAssembly, self).__init__()

         # Define all input variables from all models
        self.add('water_density',              IndepVarComp('x', 0.0))
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
        self.add('anchor_type',                IndepVarComp('x', Anchor['SUCTIONPILE'], pass_by_obj=True))
        self.add('mooring_cost_rate',          IndepVarComp('x', 0.0))
        self.add('max_offset',                 IndepVarComp('x', 0.0))

        # Run Spar Geometry component first
        self.add('sg', SparGeometry())

        # Next run MapMooring
        self.add('mm', MapMooring())
        
        self.connect('water_depth.x', ['sg.water_depth', 'mm.water_depth'])
        self.connect('water_density.x', 'mm.water_density')
        self.connect('fairlead.x', ['sg.fairlead', 'mm.fairlead'])
        self.connect('freeboard.x', 'sg.freeboard')
        self.connect('section_height.x', 'sg.section_height')
        self.connect('outer_radius.x', 'sg.outer_radius')
        self.connect('wall_thickness.x', 'sg.wall_thickness')

        self.connect('fairlead_offset_from_shell.x', 'sg.fairlead_offset_from_shell')
        self.connect('scope_ratio.x', 'mm.scope_ratio')
        self.connect('anchor_radius.x', 'mm.anchor_radius')
        self.connect('mooring_diameter.x', 'mm.mooring_diameter')
        self.connect('number_of_mooring_lines.x', 'mm.number_of_mooring_lines')
        self.connect('mooring_type.x', 'mm.mooring_type')
        self.connect('anchor_type.x', 'mm.anchor_type')
        self.connect('max_offset.x', 'mm.max_offset')
        self.connect('mooring_cost_rate.x', 'mm.mooring_cost_rate')

        # Link outputs from one model to inputs to another
        self.connect('sg.fairlead_radius', 'mm.fairlead_radius')

        # Use complex number finite differences
        self.deriv_options['type'] = 'fd'
        #self.deriv_options['type'] = 'cs'
        self.deriv_options['form'] = 'central'
        self.deriv_options['step_size'] = 1e-4
        self.deriv_options['step_calc'] = 'relative'

