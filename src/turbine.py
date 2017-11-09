from openmdao.api import Component
import numpy as np
from constants import gravity as g

class Turbine(Component):
    
    def __init__(self):
        super(Turbine,self).__init__()

        # Geometry inputs
        self.add_param('freeboard', val=25.0, units='m', desc='Length of spar above water line')
        
        # Inputs from TowerSE or user
        self.add_param('rna_mass', val=1e5, units='kg', desc='Mass of rotor nacelle assembly')
        self.add_param('rna_center_of_gravity', val=1.0, units='m', desc='Center of gravity along y-axis measured from tower base')
        self.add_param('rna_center_of_gravity_x', val=1.0, units='m', desc='Center of gravity along x-axis measured from tower centerline')
        self.add_param('rna_wind_force', val=0.0, units='kg*m/s**2', desc='Sum of drag and rotor thrust on rotor nacelle assembly')
        self.add_param('tower_mass', val=1e5, units='kg', desc='Mass of tower')
        self.add_param('tower_center_of_gravity', val=1.0, units='m', desc='Center of gravity along y-axis measured from tower base')
        self.add_param('tower_wind_force', val=0.0, units='kg*m/s**2', desc='Wind drag on tower')

        # Outputs
        self.add_output('total_mass', val=0.0, units='kg', desc='Total mass of tower + RNA')
        self.add_output('z_center_of_gravity', val=0.0, units='m', desc='z-position CofG of tower + RNA')
        self.add_output('surge_force', val=np.zeros((2,)), units='N', desc='Net forces in surge direction')
        self.add_output('force_points', val=np.zeros((2,)), units='m', desc='Net forces in surge direction')
        self.add_output('pitch_moment', val=0.0, units='N*m', desc='Net pitching moment')

 
    def solve_nonlinear(self, params, unknowns, resids):
        """Computes turbine mass (tower+rna).

        INPUTS:
        ----------
        params : dictionary of input parameters

        OUTPUTS:
        -------
        m_tubine : float, mass of turbine
        z_cg     : z-position of turbine center of gravity (z=0 at waterline)
        """
        # Unpack variables
        m_tower   = params['tower_mass']
        m_rna     = params['rna_mass']
        tower_cg  = params['tower_center_of_gravity']
        rna_cg    = params['rna_center_of_gravity'] # From base of tower z-direction
        rna_cg_x  = params['rna_center_of_gravity_x'] # x-direction from centerline
        Ftower    = params['tower_wind_force']
        Frna      = params['rna_wind_force'] # Drag or thrust?
        freeboard = params['freeboard'] # From cylinder geometry

        m_total = m_tower + m_rna
        unknowns['total_mass']          = m_total
        unknowns['z_center_of_gravity'] = freeboard + (m_tower*tower_cg + m_rna*rna_cg) / m_total

        unknowns['pitch_moment'] = - m_rna*g*rna_cg_x
        unknowns['surge_force']  = np.array([Ftower, Frna])
        unknowns['force_points'] = np.array([tower_cg+freeboard, rna_cg+freeboard])
        
