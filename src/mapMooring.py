from openmdao.api import Component
import numpy as np
import os
from mapapi import *

from constants import gravity

FINPUTSTR = os.path.abspath('input.map')


class MapMooring(Component):
    """
    OpenMDAO Component class for mooring system attached to sub-structure of floating offshore wind turbines.
    Should be tightly coupled with Spar class for full system representation.
    """
    '''The InputMAP class takes everything from FloatingSE and puts it in the
    correct format for MAP++. MAP++ then outputs a linearlized stiffness matrix
    that FloatingSE uses in its optimization analysis.'''


    def __init__(self):
        super(MapMooring,self).__init__()
    
        # Variables local to the class and not OpenMDAO

        # Environment
        self.add_param('water_density', val=1025, units='kg/m**3', desc='density of water')
        self.add_param('water_depth', val=0.0, units='m', desc='water depth')

        # Material properties

        # Design variables
        self.add_param('fairlead', val=1.0, units='m', desc='Depth below water for mooring line attachment')
        self.add_param('scope_ratio', val=1.5, units='m',desc = 'scope to fairlead height ratio')
        self.add_param('mooring_diameter', val=0.09, units='m',desc='diameter of mooring chain')
        self.add_param('number_of_mooring_lines' val=3, desc='number of mooring lines')
        self.add_param('mooring_type', val='CHAIN', desc='CHAIN, STRAND, IWRC, or FIBER')
        self.add_param('anchor_type', val='PILE', desc='PILE or DRAG')
        self.add_param('spar_outer_radius', val=0.0, units='m',desc='spar outer radius at point of fairlead attachment')
        self.add_param('fairlead_offset_from_shell', .5, units='m',desc='fairlead offset from shell')
        self.add_param('max_offset' val=0.0, units='m',desc='X offsets in discretization')

        # Cost rates
        self.add_param('mooring_cost_rate', 0.0,units='USD/m',desc='user defined mooring cost per length')
        self.add_param('anchor_cost_rate', 0.0,units='USD',desc='user defined cost per anchor')
        self.add_param('misc_cost_factor', 10.0,desc='miscellaneous cost factor in percent')

        # Outputs
        self.add_output('total_mass', units='kg',desc='total mass of mooring')
        self.add_output('total_cost', units='USD',desc='total cost for anchor + legs + miscellaneous costs')
        self.add_output('vertical_load', units='N',desc='mooring vertical load in all mooring lines')
        self.add_output('max_offset_restoring_force', units='N',desc='sume of forces in x direction')
        self.add_output('safety_factor', units='m',desc='range of damaged mooring')

        
    def solve_nonlinear(self, params, unknowns, resids):

        # Set geometry profile and other characteristics based on regressions
        self.set_geometry(params)

        # Write MAP input file and analyze the system at every angle
        self.runMAP(params, unknowns)

        self.gather_ouputs(params, unknowns)

        self.compute_costs(params, unknowns)


    def set_geometry(self, params):
        '''Minimun breaking load (MBL), wet mass per length (WML), element axial
        stiffness (ae_storm and AE_drift), area, and MCPL are calcuated here
        with MOORING DIAMETER and LINE TYPE'''

        # Unpack variables
        Dmooring       = params['mooring_diameter']
        lineType       = params['mooring_type']
        fairleadOffset = params['fairlead_offset_from_shell']
        scopeRatio     = params['scope_ratio']
        fairleadDepth  = params['fairlead']
        waterDepth     = params['water_depth']
        
        self.R_fairlead = (sparOuterDiameter/2) + fairleadOffset
        self.scope      = (waterDepth - fairleadDepth) * scopeRatio
        
        if lineType == 'CHAIN':
            self.min_break_load      = (27600.*(Dmooring**2)*(44.-80.*Dmooring))*1e3
            self.wet_mass_per_length = 18070.*(Dmooring**2)
            self.axial_stiffness     = (1.3788*Dmooring**2 - 4.93*Dmooring**3)*1e11
            self.area                = 2.64*(Dmooring**2)
            self.cost_per_length     = 0.58*1e-3*self.min_break_load/self.gravity - 87.6

        elif lineType == 'STRAND':
            self.min_break_load      = (937600*Dmooring**2-1408.3*Dmooring)*1e3
            self.wet_mass_per_length = 4110*(Dmooring**2)
            self.axial_stiffness     = 9.28*(Dmooring**2)*1e10
            self.area                = 0.58*(Dmooring**2)
            self.cost_per_length     = 0.42059603*1e-3*self.min_break_load/self.gravity + 109.5

        elif lineType == 'IWRC':
            self.min_break_load      = 648000*(Dmooring**2)*1e3
            self.wet_mass_per_length = 3670*(Dmooring**2)
            self.axial_stiffness     = 6.01*(Dmooring**2)*1e10
            self.area                = 0.54*(Dmooring**2)
            self.cost_per_length     = 0.33*1e-3*self.min_break_load/self.gravity + 139.5

        elif lineType == 'FIBER': 
            self.min_break_load      = (274700*(Dmooring**2) + 7953.9*Dmooring-879.24)*1e3
            self.wet_mass_per_length = 160.9*(Dmooring**2) + 5.522*Dmooring-0.04798
            self.axial_stiffness     = (10120*(Dmooring**2) + 320.7*Dmooring-35.47)*1e6
            #self.AE_drift           = (5156*(Dmooring**2) + 142.7*Dmooring-16)*1e6
            self.area                = 0.25*pi * (Dmooring**2)
            self.cost_per_length     = 0.53676471*1e-3*self.min_break_load/self.gravity
        else:
            raise ValueError('Available line types are: chain strand iwrc fiber')

            
    def write_line_dictionary(self, params, cable_sea_friction_coefficient=0.65):
        '''Writes the forth line of the input.map file. This is where 
        LINE_TYPE, DIAMETER, AIR_MASS_DENSITY, ELEMENT_AXIAL_STIFFNESS, and
        CABLE_SEA_FRICTION_COEFFICIENT is inputted. 
        CABLE_SEA_FRICTION_COEFFICIENT defaults to 1 when none is given.'''
        # Unpack variables
        rhoWater = params['water_density']
        
        air_mass_density = self.wet_mass_per_length + (rhoWater*self.area)
        self.finput.write('---------------------- LINE DICTIONARY ---------------------------------------')
        self.finput.write('LineType  Diam      MassDenInAir   EA            CB   CIntDamp  Ca   Cdn    Cdt')
        self.finput.write('(-)       (m)       (kg/m)        (N)           (-)   (Pa-s)    (-)  (-)    (-)')
        self.finput.write('%s   %.5f   %.5f   %.5f   %.5f   1.0E8   0.6   -1.0   0.05\n' %
                          (params['mooring_type'], params['mooring_diameter'], air_mass_density,
                           self.axial_stiffness, cable_sea_friction_coefficient) )

        
    def write_node_properties_header(self):
        '''Writes the node properties header:'''
        self.finput.write('---------------------- NODE PROPERTIES ---------------------------------------')
        self.finput.write('Node  Type       X       Y       Z      M     B     FX      FY      FZ')
        self.finput.write('(-)   (-)       (m)     (m)     (m)    (kg)  (mˆ3)  (N)     (N)     (N)')


    def write_node_properties(self, number, node_type, x_pos, y_pos, z_pos, point_mass=0, displaced_volume=0,
                              x_force=None, y_force=None, z_force=None):
        '''Writes the input information for a node based on NODE_TYPE. X_FORCE, 
        Y_FORCE_APP, Z_FORCE_APP defaults to '#' if none is given.'''

        nodeStr = node_type.lower()

        if not nodeStr in ['fix', 'connect', 'vessel']:
            raise ValueError('%s is not a valid node type for node %d' % (node_type, number))

        if nodeStr == 'connect':
            try:
                x_force = float(x_force)
                y_force = float(y_force)
                z_force = float(z_force)
            except:
                raise ValueError('%s must have numerical force applied values.' % node_type)

        forceStr = '#   #   #' 
        if nodeStr == 'connect':
            forceStr = '%.5f   %.5f   %.5f\n' % (x_force, y_force, z_force)
            posStr   = '#%.5f   #%.5f   #%.5f   ' % (x_pos, y_pos, z_pos)
        elif nodeStr == 'fix':
            posStr   = '%.5f   %.5f   depth   ' % (x_pos, y_pos)
        elif nodeStr == 'vessel':
            posStr   = '%.5f   %.5f   %.5f   ' % (x_pos, y_pos, z_pos)
            
        self.finput.write('%d   ' % number)
        self.finput.write('%s   ' % node_type)
        self.finput.write(posStr)
        self.finput.write('%.5f   %.5f   ' % (point_mass, displaced_volume) )
        self.finput.write(forceStr)
        self.finput.write('\n')


    def write_line_properties(self, params, line_number=1, anchor_node=1, fairlead_node=2, flags=' '):
        '''Writes the input information for the line properties. This explains
        what node number is the ANCHOR and what node number is the FAIRLEAD, 
        as well as the UNSTRETCHED_LENGTH between the two nodes.'''
        self.finput.write('---------------------- LINE PROPERTIES ---------------------------------------')
        self.finput.write('Line    LineType  UnstrLen  NodeAnch  NodeFair  Flags')
        self.finput.write('(-)      (-)       (m)       (-)       (-)       (-)')
        self.finput.write('%d   %s   %.5f   %d   %d   %s\n' %
                          (line_number, params['mooring_type'], self.scope, anchor_node, fairlead_node, flags) )

        
    def write_solver_options(self, params):
        '''Writes the solver options at the end of the input file, as well as 
        takes the self.NUMBER_OF_MOORING_LINES and places them evenly within 360
        degrees. For NUMBER_OF_MOORING_LINES = 3:
        '''

        # Unpack variables
        nlines = params['number_of_mooring_lines']
        
        self.finput.write('---------------------- SOLVER OPTIONS-----------------------------------------')
        self.finput.write('Option\n')
        self.finput.write('(-)\n')
        self.finput.write('help\n')
        self.finput.write(' integration_dt 0\n')
        self.finput.write(' kb_default 3.0e6\n')
        self.finput.write(' cb_default 3.0e5\n')
        self.finput.write(' wave_kinematics \n')
        self.finput.write('inner_ftol 1e-6\n')
        self.finput.write('inner_gtol 1e-6\n')
        self.finput.write('inner_xtol 1e-6\n')
        self.finput.write('outer_tol 1e-4\n')
        self.finput.write(' pg_cooked 10000 1\n')
        self.finput.write(' outer_fd \n')
        self.finput.write(' outer_bd \n')
        self.finput.write(' outer_cd\n')
        self.finput.write(' inner_max_its 100\n')
        self.finput.write(' outer_max_its 500\n')
        self.finput.write('repeat ')
        n = 360.0/nlines
        degree = n
        while degree + n <= 360:
            self.finput.write('%d ' % degree)
            degree += n
        self.finput.write('\n')
        self.finput.write(' krylov_accelerator 3\n')
        self.finput.write(' ref_position 0.0 0.0 0.0\n')

        
    def write_input_file(self, params):
        # Unpack variables
        fairleadDepth = params['fairlead']

        # Open the map input file
        self.finput = open(FINPUTSTR, 'wb')

        # Write the "Line Dictionary" section
        self.write_line_dictionary(params)

        # Write the "Node Properties" section
        self.write_node_properties_header()
        #firgure out how to find out the anchor radius radius
        self.write_node_properties(1, "FIX", 853.87, 0, None)
        self.write_node_properties(2, "VESSEL", self.R_fairlead, 0, -fairleadDepth)

        # Write the "Line Properties" section
        self.write_line_properties(params)

        # Write the "Solve Options" section
        self.write_solver_options(params)

        # Close the input file
        self.finput.close()
        
        
    def runMAP(self, params, unknowns):
        # Unpack variables
        rhoWater      = params['water_density']
        waterDepth    = params['water_depth']
        fairleadDepth = params['fairlead']
        Dmooring      = params['mooring_diameter']
        nlines        = params['number_of_mooring_lines']

        # Write the mooring system input file for this design
        self.write_input_file(params)

        # Initiate MAP++ for this design
        mymap = MapAPI( )
        mymap.map_set_sea_depth(waterDepth)
        mymap.map_set_gravity(gravity)
        mymap.map_set_sea_density(rhoWater)
        mymap.read_file(FINPUTSTR)
        mymap.init( )

        # Stress limits TODO: MAKE CONSTRAINTS
        intact_mooring  = self.min_break_load*.60
        damaged_mooring = self.min_break_load*.80

        # Get the vertical load on the spar
        Fz1 = 0.0
        Fz2 = 0.0
        for k in xrange(nlines):
            _,_,fz = mymap.get_fairlead_force_3d(k)
            _,V    = mymap.get_fairlead_force_2d(k)
            Fz1 += fz
            Fz2 += V
        print Fz1, Fz2
        unknowns['vertical_load'] = np.maximum(Fz1, Fz2)
        
        # Get angles by which to find the weakest line
        dangle  = 2.0
        angles  = np.deg2rad( np.arange(0.0, 360.0, dangle) )
        nangles = len(angles)

        # Finite difference epsilon for stiffness linearization
        epsilon = 1e-3

        # Get restoring force at weakest line at maximum allowable offset
        # TODO: Remember global worst- will this always be along one of the mooring line angles?
        max_tension = 0.0
        F_max_tension = None
        T = np.zeros((nlines,))
        for a in angles:
            idir = np.array([np.cos(angle), np.sin(angle)])
            surge = offset * idir[0]
            sway  = offset * idir[1]

            F     = 0.0
            mymap.displace_vessel(surge, sway, 0, 0, 0, 0)
            mymap.update_states(0.0, 0)
            for k in xrange(nlines):
                fx,fy,_ = mymap.get_fairlead_force_3d(k)
                H,V     = mymap.get_fairlead_force_2d(k)
                T[k]    = np.hypot(H, V)
                F      += np.dot([fx, fy], idir) #hypot(fx, fy)

            tempMax = T.max()
            if tempMax > max_tension:
                max_tension   = tempMax
                F_max_tension = F

        # Store the weakest restoring force when the vessel is offset the maximum amount
        unknowns['max_offset_restoring_force'] = F_max_tension
        unknowns['safety_factor'] = max_tension / self.min_break_load
        # TODO: Ensure global maximum is less than X% of MBL
        # TODO: Return force along weakest to compare to current force in spar.py
        mymap.end()

    def compute_cost(self, params, unknowns):
        nlines = params['number_of_mooring_lines']
        legs_total = nlines * self.cost_per_length * self.scope
        if self.anchor_type =='DRAG':
            each_anchor = 1e-3 * self.min_break_load / gravity / 20*2000
        elif self.anchor_type == 'PILE':
            each_anchor = 150000.* np.sqrt(1e-3*self.min_break_load/gravity/1250.)
        anchor_total = each_anchor*nlines
        unknowns['mooring_total_cost'] = (legs_total + anchor_total) * (1 + 1e-2*self.misc_cost_factor)
        unknowns['mooring_mass'] = (self.wet_mass_per_length + 0.25*np.pi*Dmooring**2*rhoWater)*self.scope*nlines
