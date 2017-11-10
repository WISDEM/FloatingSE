from openmdao.api import Component
import numpy as np
import os
import sys
from mapapi import *

from constants import gravity

FINPUTSTR = os.path.abspath('input.map')
        
class MapMooring(Component):
    """
    OpenMDAO Component class for mooring system attached to sub-structure of floating offshore wind turbines.
    Should be tightly coupled with Spar class for full system representation.
    """

    def __init__(self):
        super(MapMooring,self).__init__()
    
        # Variables local to the class and not OpenMDAO
        self.scope               = None
        self.min_break_load      = None
        self.wet_mass_per_length = None
        self.axial_stiffness     = None
        self.area                = None
        self.cost_per_length     = None

        # Environment
        self.add_param('water_density', val=1025.0, units='kg/m**3', desc='density of water')
        self.add_param('water_depth', val=0.0, units='m', desc='water depth')

        # Material properties

        # Inputs from SparGeometry
        self.add_param('fairlead_radius', val=0.0, units='m', desc='Outer spar radius at fairlead depth (point of mooring attachment)')
        
        # Design variables
        self.add_param('fairlead', val=1.0, units='m', desc='Depth below water for mooring line attachment')
        self.add_param('scope_ratio', val=1.5, units='m',desc='total mooring line length (scope) to fairlead depth (fairlead anchor to sea floor)')
        self.add_param('anchor_radius', val=1.0, units='m', desc='radius from center of spar to mooring anchor point')
        self.add_param('mooring_diameter', val=0.2, units='m',desc='diameter of mooring line')

        # User inputs (could be design variables)
        self.add_param('number_of_mooring_lines', val=3, desc='number of mooring lines', pass_by_obj=True)
        self.add_param('mooring_type', val='CHAIN', desc='chain, nylon, polyester, fiber, or iwrc', pass_by_obj=True)
        self.add_param('anchor_type', val='PILE', desc='PILE or DRAG', pass_by_obj=True)
        self.add_param('max_offset', val=0.0, units='m',desc='X offsets in discretization')

        # Cost rates
        self.add_param('mooring_cost_rate', val=1.1, desc='miscellaneous cost factor in percent')

        # Outputs
        self.add_output('mooring_effective_mass', val=0.0, units='kg',desc='total effective mass of mooring based on vertical loading')
        self.add_output('mooring_mass', val=0.0, units='kg',desc='total mass of mooring')
        self.add_output('mooring_cost', val=0.0, units='USD',desc='total cost for anchor + legs + miscellaneous costs')
        self.add_output('vertical_load', val=0.0, units='kg*m/s**2',desc='mooring vertical load in all mooring lines')
        self.add_output('max_offset_restoring_force', val=0.0, units='kg*m/s**2',desc='sume of forces in x direction')

        # Output constriants
        self.add_output('safety_factor', val=0.0, units='m',desc='range of damaged mooring')
        self.add_output('mooring_length_min', val=0.0, desc='mooring line length ratio to nodal distance')
        self.add_output('mooring_length_max', val=0.0, desc='mooring line length ratio to nodal distance')
        
    def solve_nonlinear(self, params, unknowns, resids):
        """Sets mooring line properties then writes MAP input file and executes MAP.
        
        INPUTS:
        ----------
        params   : dictionary of input parameters
        unknowns : dictionary of output parameters
        
        OUTPUTS  : none (all unknown dictionary values set)
        """

        # Set characteristics based on regressions / empirical data
        self.set_properties(params)

        # Set geometry profile
        self.set_geometry(params, unknowns)

        # Write MAP input file and analyze the system at every angle
        self.runMAP(params, unknowns)

        # Compute costs for the system
        self.compute_cost(params, unknowns)


    def set_properties(self, params):
        """Sets mooring line properties: Minimum Breaking Load, Mass per Length, Axial Stiffness, Cross-Sectional Area, Cost-per-Length.
        
        INPUTS:
        ----------
        params   : dictionary of input parameters
        
        OUTPUTS  : Parameters are class variables and are set internally

        References:
        https://daim.idi.ntnu.no/masteroppgaver/015/15116/masteroppgave.pdf
        http://offshoremechanics.asmedigitalcollection.asme.org/article.aspx?articleid=2543338
        https://www.orcina.com/SoftwareProducts/OrcaFlex/Documentation/Help/Content/html/
        Chain.htm
        Chain,AxialandBendingStiffness.htm
        Chain,MechanicalProperties.htm
        RopeWire.htm
        RopeWire,MinimumBreakingLoads.htm
        RopeWire,Massperunitlength.htm
        RopeWire,AxialandBendingStiffness.htm
        """
        
        # Unpack variables
        Dmooring       = params['mooring_diameter']
        lineType       = params['mooring_type'].upper()

        # Set parameters based on regressions for different mooring line type
        Dmooring2 = Dmooring**2

        # TODO: Costs per unit length are not synced with new input sources
        if lineType == 'CHAIN':
            #self.min_break_load      = 2.74e7  * Dmooring2 * (44.0 - 80.0*Dmooring)
            # Use a linear fit to the other fit becuase it is poorly conditioned for optimization
            self.min_break_load      = 1e3*np.maximum(1.0, -5445.2957034820683+176972.68498888266*Dmooring)
            self.wet_mass_per_length = 19.9e3  * Dmooring2
            self.axial_stiffness     = 8.54e10 * Dmooring2
            self.area                = 2.0 * 0.25 * np.pi * Dmooring2
            self.cost_per_length     = 3.415e4  * Dmooring2 #0.58*1e-3*self.min_break_load/gravity - 87.6

        elif lineType == 'NYLON':
            self.min_break_load      = 139357e3 * Dmooring2
            self.wet_mass_per_length = 0.6476e3 * Dmooring2
            self.axial_stiffness     = 1.18e8   * Dmooring2
            self.area                = 0.25 * np.pi * Dmooring2
            self.cost_per_length     = 3.415e4  * Dmooring2 #0.42059603*1e-3*self.min_break_load/gravity + 109.5

        elif lineType == 'POLYESTER':
            self.min_break_load      = 170466e3 * Dmooring2
            self.wet_mass_per_length = 0.7978e3 * Dmooring2
            self.axial_stiffness     = 1.09e9   * Dmooring2
            self.area                = 0.25 * np.pi * Dmooring2
            self.cost_per_length     = 3.415e4  * Dmooring2 #0.42059603*1e-3*self.min_break_load/gravity + 109.5

        elif lineType == 'FIBER': 
            self.min_break_load      = 584175e3 * Dmooring2
            self.wet_mass_per_length = 3.6109e3 * Dmooring2
            self.axial_stiffness     = 3.67e10  * Dmooring2
            self.area                = 0.455 * 0.25 * np.pi * Dmooring2
            self.cost_per_length     = 2.0 * 6.32e4  * Dmooring2 #0.53676471*1e-3*self.min_break_load/gravity

        elif lineType == 'IWRC':
            self.min_break_load      = 633358e3 * Dmooring2
            self.wet_mass_per_length = 3.9897e3 * Dmooring2
            self.axial_stiffness     = 4.04e10  * Dmooring2
            self.area                = 0.455 * 0.25 * np.pi * Dmooring2
            self.cost_per_length     = 6.32e4  * Dmooring2 #0.33*1e-3*self.min_break_load/gravity + 139.5

        else:
            raise ValueError('Available line types are: chain nylon polyester fiber iwrc')

            
    def set_geometry(self, params, unknowns):
        # Unpack variables
        fairleadDepth = params['fairlead']
        R_fairlead    = params['fairlead_radius']
        R_anchor      = params['anchor_radius']
        waterDepth    = params['water_depth']
        scopeRatio    = params['scope_ratio']

        # Define total length of mooring line
        self.scope = scopeRatio * (waterDepth - fairleadDepth)

        # Check that there is enough mooring line between anchor point and attachment point
        vessel_node = np.array([R_fairlead, -fairleadDepth])
        anchor_node = np.array([R_anchor, -waterDepth])
        node_dist   = np.sqrt( np.sum( (anchor_node - vessel_node)**2) )

        # Create constraint that there is at least enough line to cover this distance
        unknowns['mooring_length_min'] = self.scope / node_dist
        unknowns['mooring_length_max'] = self.scope / (R_anchor + waterDepth - fairleadDepth)
        
    
    def write_line_dictionary(self, params, cable_sea_friction_coefficient=0.65):
        """Writes LINE DICTIONARY section of input.map file
        
        INPUTS:
        ----------
        params   : dictionary of input parameters
        cable_sea_friction_coefficient : coefficient of friction with sea floor (defaults to 0.65)
        
        OUTPUTS  : none
        """
        # Unpack variables
        rhoWater = params['water_density']
        lineType = params['mooring_type'].lower()
        Dmooring = params['mooring_diameter']
        
        air_mass_density = self.wet_mass_per_length + (rhoWater*self.area)
        self.finput.write('---------------------- LINE DICTIONARY ---------------------------------------\n')
        self.finput.write('LineType  Diam      MassDenInAir   EA            CB   CIntDamp  Ca   Cdn    Cdt\n')
        self.finput.write('(-)       (m)       (kg/m)        (N)           (-)   (Pa-s)    (-)  (-)    (-)\n')
        self.finput.write('%s   %.5f   %.5f   %.5f   %.5f   1.0E8   0.6   -1.0   0.05\n' %
                          (lineType, Dmooring, air_mass_density, self.axial_stiffness, cable_sea_friction_coefficient) )

        
    def write_node_properties_header(self):
        """Writes NODE PROPERTIES section header of input.map file
        
        INPUTS: none 
        ----------
        
        OUTPUTS  : none
        """
        self.finput.write('---------------------- NODE PROPERTIES ---------------------------------------\n')
        # Doesn't like some weird character here somewhere
        #self.finput.write('Node  Type       X       Y       Z      M     B     FX      FY      FZ\n')
        #self.finput.write('(-)   (-)       (m)     (m)     (m)    (kg)  (m^3)  (N)     (N)     (N)\n')
        self.finput.write('Node Type X     Y    Z   M     V FX FY FZ\n')
        self.finput.write('(-)  (-) (m)   (m)  (m) (kg) (m^3) (kN) (kN) (kN)\n')



    def write_node_properties(self, number, node_type, x_pos, y_pos, z_pos,
                              point_mass=0, displaced_volume=0,
                              x_force=None, y_force=None, z_force=None):
        """Writes NODE PROPERTIES data of input.map file.  Nodes are connections between mooring lines and bridles, vessels, and anchors
        
        INPUTS:
        ----------
        number           : The node number listing in the input file
        node_type        : fix / vessel / connect (fix=anchor)
        x_, y_, z_pos    : position of node in coordinate system (separate inputs)
        point_mass       : see MAP reference (defaults to 0)
        displaced_volume : see MAP reference (defaults to 0)
        x_, y_, z_force  : see MAP reference (defaults to None)
        
        OUTPUTS  : none
        """
        # Ensure this connection is something MAP understands
        nodeStr = node_type.lower()
        if not nodeStr in ['fix', 'connect', 'vessel']:
            raise ValueError('%s is not a valid node type for node %d' % (node_type, number))

        # If this is a node between two lines have to specify connection details
        if nodeStr == 'connect':
            try:
                x_force = float(x_force)
                y_force = float(y_force)
                z_force = float(z_force)
            except:
                raise ValueError('%s must have numerical force applied values.' % node_type)

        # Set location strings
        forceStr = '#   #   #' 
        if nodeStr == 'connect':
            forceStr = '%.5f   %.5f   %.5f\n' % (x_force, y_force, z_force)
            posStr   = '#%.5f   #%.5f   #%.5f   ' % (x_pos, y_pos, z_pos)
        elif nodeStr == 'fix':
            posStr   = '%.5f   %.5f   depth   ' % (x_pos, y_pos)
        elif nodeStr == 'vessel':
            posStr   = '%.5f   %.5f   %.5f   ' % (x_pos, y_pos, z_pos)

        # Write the connection line
        self.finput.write('%d   ' % number)
        self.finput.write('%s   ' % node_type)
        self.finput.write(posStr)
        self.finput.write('%.5f   %.5f   ' % (point_mass, displaced_volume) )
        self.finput.write(forceStr)
        self.finput.write('\n')


    def write_line_properties(self, params, line_number=1, anchor_node=1, fairlead_node=2, flags=' '):
        """Writes LINE PROPERTIES section of input.map file that connects multiple nodes
        
        INPUTS:
        ----------
        params        : dictionary of input parameters- only 'mooring_type' is used
        line_number   : Line ID number (defaults to 1)
        anchor_node   : Node number corresponding to anchor (defaults to 1)
        fairlead_node : Node number corresponding to fairlead (vessel) node (defaults to 2)
        flags         : see MAP reference (defaults to empty string ' ')
        
        OUTPUTS  : none
        """
        self.finput.write('---------------------- LINE PROPERTIES ---------------------------------------\n')
        self.finput.write('Line    LineType  UnstrLen  NodeAnch  NodeFair  Flags\n')
        self.finput.write('(-)      (-)       (m)       (-)       (-)       (-)\n')
        self.finput.write('%d   %s   %.5f   %d   %d   %s\n' %
                          (line_number, params['mooring_type'], self.scope, anchor_node, fairlead_node, flags) )

        
    def write_solver_options(self, params):
        """Writes SOLVER OPTIONS section of input.map file, 
        which includes repeating node/line arrangement in even angular spacing around structure.
        
        INPUTS:
        ----------
        params        : dictionary of input parameters- only 'number_of_mooring_lines' is used
        
        OUTPUTS  : none
        """
        # Unpack variables
        nlines = params['number_of_mooring_lines']
        
        self.finput.write('---------------------- SOLVER OPTIONS-----------------------------------------\n')
        self.finput.write('Option\n')
        self.finput.write('(-)\n')
        self.finput.write('help\n')
        self.finput.write(' integration_dt 0\n')
        self.finput.write(' kb_default 3.0e6\n')
        self.finput.write(' cb_default 3.0e5\n')
        self.finput.write(' wave_kinematics \n')
        self.finput.write('inner_ftol 1e-5\n')
        self.finput.write('inner_gtol 1e-5\n')
        self.finput.write('inner_xtol 1e-5\n')
        self.finput.write('outer_tol 1e-3\n')
        self.finput.write(' pg_cooked 10000 1\n')
        self.finput.write(' outer_fd \n')
        self.finput.write(' outer_bd \n')
        self.finput.write(' outer_cd\n')
        self.finput.write(' inner_max_its 200\n')
        self.finput.write(' outer_max_its 600\n')
        # Repeat the details for the one mooring line multiple times
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
        """Writes SOLVER OPTIONS section of input.map file, 
        which includes repeating node/line arrangement in even angular spacing around structure.
        
        INPUTS:
        ----------
        params        : dictionary of input parameters- only 'number_of_mooring_lines' is used
        
        OUTPUTS  : none
        """
        # Unpack variables
        fairleadDepth = params['fairlead']
        R_fairlead    = params['fairlead_radius']
        R_anchor      = params['anchor_radius']

        # Open the map input file
        self.finput = open(FINPUTSTR, 'wb')

        # Write the "Line Dictionary" section
        self.write_line_dictionary(params)

        # Write the "Node Properties" section
        self.write_node_properties_header()
        # One end on sea floor the other at fairlead
        self.write_node_properties(1, "FIX", R_anchor, 0, None)
        self.write_node_properties(2, "VESSEL", R_fairlead, 0, -fairleadDepth)

        # Write the "Line Properties" section
        self.write_line_properties(params)

        # Write the "Solve Options" section
        self.write_solver_options(params)

        # Close the input file
        self.finput.close()
        
        
    def runMAP(self, params, unknowns):
        """Writes MAP input file, executes, and then queries MAP to find 
        maximum loading and displacement from vessel displacement around all 360 degrees
        
        INPUTS:
        ----------
        params   : dictionary of input parameters
        unknowns : dictionary of output parameters
        
        OUTPUTS  : none (multiple unknown dictionary values set)
        """
        # Unpack variables
        rhoWater      = params['water_density']
        waterDepth    = params['water_depth']
        fairleadDepth = params['fairlead']
        Dmooring      = params['mooring_diameter']
        nlines        = params['number_of_mooring_lines']
        offset        = params['max_offset']

        # Write the mooring system input file for this design
        self.write_input_file(params)

        # Initiate MAP++ for this design
        mymap = MapAPI( )
        #mymap.ierr = 0
        mymap.map_set_sea_depth(waterDepth)
        mymap.map_set_gravity(gravity)
        mymap.map_set_sea_density(rhoWater)
        mymap.read_file(FINPUTSTR)
        mymap.init( )

        # Get the vertical load on the spar
        Fz = 0.0
        for k in xrange(nlines):
            _,_,fz = mymap.get_fairlead_force_3d(k)
            Fz += fz
        unknowns['vertical_load'] = Fz
        unknowns['mooring_effective_mass'] = Fz / gravity
        
        # Get angles by which to find the weakest line
        dangle  = 2.0
        angles  = np.deg2rad( np.arange(0.0, 360.0, dangle) )
        nangles = len(angles)

        # Get restoring force at weakest line at maximum allowable offset
        # Will global minimum always be along mooring angle?
        max_tension   = 0.0
        max_angle     = None
        F_max_tension = None
        T = np.zeros((nlines,))
        F = np.zeros((nlines,))
        # Loop around all angles to find weakest point
        for a in angles:
            # Unit vector and offset in x-y components
            idir  = np.array([np.cos(a), np.sin(a)])
            surge = offset * idir[0]
            sway  = offset * idir[1]

            # Get restoring force of offset at this angle
            mymap.displace_vessel(surge, sway, 0, 0, 0, 0) # 0s for z, angles
            mymap.update_states(0.0, 0)
            for k in xrange(nlines):
                # Force in x-y-z coordinates
                fx,fy,fz = mymap.get_fairlead_force_3d(k)
                T[k]     = np.sqrt(fx*fx + fy*fy + fz*fz)
                # Total restoring force
                F[k]     = np.dot([fx, fy], idir)
                
            # Check if this is the weakest direction (highest tension)
            tempMax = T.max()
            if tempMax > max_tension:
                max_tension   = tempMax
                F_max_tension = F.sum()
                max_angle     = a
                
        # Store the weakest restoring force when the vessel is offset the maximum amount
        unknowns['max_offset_restoring_force'] = F_max_tension
        unknowns['safety_factor'] = max_tension / self.min_break_load

        mymap.end()

        
    def compute_cost(self, params, unknowns):
        """Computes cost, based on mass scaling, of mooring system.
        
        INPUTS:
        ----------
        params   : dictionary of input parameters
        unknowns : dictionary of output parameters
        
        OUTPUTS  : none (mooring_cost/mass unknown dictionary values set)
        """
        # Unpack variables
        nlines        = params['number_of_mooring_lines']
        rhoWater      = params['water_density']
        Dmooring      = params['mooring_diameter']
        anchorType    = params['anchor_type'].upper()
        costFact      = params['mooring_cost_rate']
        
        # Cost of all of the mooring lines
        legs_total = nlines * self.cost_per_length * self.scope

        # Cost of anchors
        anchor_rate = 1e-3 * self.min_break_load / gravity / 20*2000
        #if anchorType =='DRAG':
        #    anchor_rate = 1e-3 * self.min_break_load / gravity / 20*2000
        #elif anchorType  == 'PILE':
        #    anchor_rate = 150000.* np.sqrt(1e-3*self.min_break_load/gravity/1250.)
        #else:
        #    raise ValueError('Anchor Type must be DRAG or PILE')
        anchor_total = anchor_rate*nlines

        # Total summations
        unknowns['mooring_cost'] = costFact*(legs_total + anchor_total)
        unknowns['mooring_mass'] = (self.wet_mass_per_length + rhoWater*0.25*np.pi*Dmooring**2)*self.scope*nlines
        