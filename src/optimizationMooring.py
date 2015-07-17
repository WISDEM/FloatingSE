from openmdao.main.api import Component, Assembly, convert_units
from openmdao.main.datatypes.api import Float, Array, Enum, Str, Int, Bool
from openmdao.lib.drivers.api import COBYLAdriver, SLSQPdriver
from mooring import Mooring
#from spar_discrete import spar_discrete
import time
import numpy as np
from mooring_utils import ref_table, fairlead_anchor_table

class optimizationMooring(Assembly):
    # variables 
    def configure(self):


def example_218WD_3MW():
	test=Mooring()
	test.number_of_mooring_lines = 3
	test.water_depth = 218.
	test.mooring_type = 'CHAIN'
	test.anchor_type = 'PILE'
	test.fairlead_offset_from_shell = 0.5
	test.outer_diameter_base = 9.0
	test.start_elevation = [13., 7., -5., -20.]
	test.end_elevation = [7., -5., -20., -67.]
	test.shell_buoyancy = [0.000,144905.961,688303.315,3064761.078]
	test.shell_mass = [40321.556,88041.563,137796.144,518693.048]
	test.bulkhead_mass = [0.000,10730.836,0.000,24417.970]
    test.ring_mass = [1245.878,5444.950,6829.259,28747.490]
	test.KCG = [76.819,68.649,53.513,22.445]
	test.KCB = 
	test.water_density
	test.outer_diameter_top 
    test.run()


if __name__ == "__main__":
    example_218WD_3MW()
    #example_218WD_6MW()
    #example_218WD_10MW()