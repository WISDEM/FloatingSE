from openmdao.api import Component
import numpy as np

class TowerTransition(Component):
    """
    OpenMDAO Component class for coupling between substructure and turbine tower
    """

    def __init__(self, nNodes, diamFlag=True):
        super(TowerTransition,self).__init__()

        self.diamFlag = diamFlag
        
        # Design variables
        self.add_param('tower_metric', val=np.zeros((nNodes,)), units='m', desc='water depth')
        self.add_param('base_metric', val=np.zeros((nNodes,)), units='m', desc='outer radius of tower at base')

        # Output constraints
        self.add_output('transition_buffer', val=0.0, units='m', desc='Buffer between substructure base and tower base')
        
        # Derivatives
        self.deriv_options['type'] = 'fd'
        self.deriv_options['form'] = 'central'
        self.deriv_options['step_calc'] = 'relative'
        self.deriv_options['step_size'] = 1e-5
        
    def solve_nonlinear(self, params, unknowns, resids):
        r_tower = params['tower_metric']
        r_base  = params['base_metric']

        if self.diamFlag:
            # Inputs were diameters and not radii, so we need to convert to radii
            r_tower *= 0.5
            r_base *= 0.5

        # Constrain spar top to be at least greater than tower base
        unknowns['transition_buffer'] = r_base[-1] - r_tower[0]
    
