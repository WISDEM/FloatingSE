from openmdao.main.api import Component, Assembly,convert_units
from openmdao.lib.datatypes.api import Float, Array, Str, Int, Bool
from openmdao.lib.drivers.api import SLSQPdriver
import numpy as np
from scipy.optimize import fmin, minimize
from sympy.solvers import solve
from sympy import Symbol
import math
from spar_utils import filtered_stiffeners_table
pi=np.pi

class Mooring(Component):
	#design variables
	fairlead_depth = Float(iotype='in',units='m',desc = 'fairlead depth')
	scope_ratio = Float(iotype='in',units='m',desc = 'scope to fairlead height ratio')
	pretension_percent = Float(iotype='in',desc='Pre-Tension Percentage of MBL (match PreTension)')
	# inputs 
	water_depth = Float(iotype='in',units='m',desc='water depth')
	
	def __init__(self):
        super(Mooring,self).__init__()
    def execute(self):
    	FH = self.water_depth-self.fairlead_depth
		S = FH*self.scope_ratio
		PTEN = MBL*self.pretension_percent/100.
		







 
	