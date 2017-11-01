#!/usr/bin/env python
# encoding: utf-8

import sys
import os
# just to temporarily change PYTHONPATH without installing
sys.path.append(os.path.expanduser('~') + '/Dropbox/NREL/NREL_WISDEM/src/twister/rotoraero')
from openmdao.main.api import Component, Assembly, convert_units
from openmdao.main.datatypes.api import Float, Array, Enum, Str, Int, Bool
from openmdao.lib.drivers.api import COBYLAdriver,SLSQPdriver
from sparAssemblyWithMAP import sparAssemblyCalculation
import numpy as np
import time
#from utils import filtered_stiffeners_table
from utils import sys_print

def example_OC3():
    """Calculation with properties based mostly on the OC3."""
    example = sparAssemblyCalculation()
    example.tower_base_outer_diameter = 6.5
    example.tower_top_outer_diameter = 3.87
    example.tower_length = 77.6
    example.example_turbine_size = '3MW' #not sure if this is correct
    example.RNA_center_of_gravity_y = 1.75
    example.wall_thickness = [.057, .056, .042, .046, .052]
    example.rotor_diameter = 126.
    # example.cut_out_speed
    example.air_density = 1.198
    example.wind_reference_speed = 11.
    example.wind_reference_height = 89.350
    example.gust_factor = 1.0
    example.alpha = .11
    example.RNA_center_of_gravity_x = 1.9
    example.tower_mass = 249718.0
    example.RNA_mass = 347460.
    example.stiffener_index = 259
    example.number_of_sections = 5
    example.bulk_head = ['N', 'T', 'N', 'B', 'B']
    example.number_of_rings = [3, 2, 10, 19, 32]
    example.neutral_axis = .21 #not sure if this number is correct
    # example.straight_col_cost
    # example.tapered_col_cost
    # example.outfitting_cost
    # example.ballast_cost
    example.gravity = 9.806
    example.load_condition = 'N'
    example.significant_wave_height = 8.
    example.significant_wave_period = 10.
    example.material_density = 7850
    example.E = 200.
    example.nu = .3
    example.yield_stress = 345.
    example.shell_mass_factor = 1
    example.bulkhead_mass_factor = 1.25
    # example.ring_mass_factor
    example.outfitting_factor = .06
    example.spar_mass_factor = 1.04
    example.permanent_ballast_height = 0.
    example.fixed_ballast_height = 10.
    example.permanent_ballast_density = 4000.
    example.fixed_ballast_density = 4492.48
    # example.offset_amplification_factor
    example.water_density = 1025.
    example.spar_elevations = [10.0, -4.0, -12.0, -42., -71., -120.]
    example.spar_outer_diameter = [6.5, 6.5, 9.4, 9.4, 9.4]
    example.water_depth = 320.
    example.fairlead_depth = 70.
    example.scope_ratio = 3.609
    example.pretension_percent = 11.173 #map doesnt use
    example.mooring_diameter = .09
    example.number_of_mooring_lines = 3
    example.mooring_type = 'CHAIN'
    example.anchor_type = 'PILE'
    example.fairlead_offset_from_shell = .5
    example.user_MBL = 8158000.
    example.user_WML = 71.186
    example.user_AE_storm = 384243000/.006
    example.user_MCPL = 0.
    example.user_anchor_cost = 0.
    example.misc_cost_factor = 10
    example.number_of_discretizations = 20 #map doesnt use
    example.spar.stiffener_curve_fit = False #not sure if this is correct
    example.run()
    print '-------------OC3---------------'
    sys_print(example)

if __name__ == "__main__":
    example_OC3()