from openmdao.api import Component, Group, IndepVarComp
import numpy as np
import pyframe3dd.frame3dd as frame3dd
from commonse.utilities import nodal2sectional

from commonse import gravity, eps, Tube
import commonse.UtilizationSupplement as util
from commonse.WindWaveDrag import AeroHydroLoads, CylinderWindDrag, CylinderWaveDrag
from commonse.environment import WaveBase, PowerWind
from commonse.vertical_cylinder import CylinderDiscretization, CylinderMass

def find_nearest(array,value):
    return (np.abs(array-value)).argmin() 



class FloatingFrame(Component):
    """
    OpenMDAO Component class for semisubmersible pontoon / truss structure for floating offshore wind turbines.
    Should be tightly coupled with Semi and Mooring classes for full system representation.
    """

    def __init__(self, nFull):
        super(FloatingFrame,self).__init__()

        # Environment
        self.add_param('water_density', val=0.0, units='kg/m**3', desc='density of water')

        # Material properties
        self.add_param('material_density', val=0., units='kg/m**3', desc='density of material')
        self.add_param('E', val=0.0, units='Pa', desc='Modulus of elasticity (Youngs) of material')
        self.add_param('G', val=0.0, units='Pa', desc='Shear modulus of material')
        self.add_param('yield_stress', val=0.0, units='Pa', desc='yield stress of material')

        # Base column
        self.add_param('base_z_full', val=np.zeros((nFull,)), units='m', desc='z-coordinates of section nodes (length = nsection+1)')
        self.add_param('base_d_full', val=np.zeros((nFull,)), units='m', desc='outer radius at each section node bottom to top (length = nsection + 1)')
        self.add_param('base_t_full', val=np.zeros((nFull,)), units='m', desc='shell wall thickness at each section node bottom to top (length = nsection + 1)')
        self.add_param('base_column_mass', val=np.zeros((nFull-1,)), units='kg', desc='mass of base column by section')
        self.add_param('base_column_displaced_volume', val=np.zeros((nFull-1,)), units='m**3', desc='column volume of water displaced by section')
        self.add_param('base_column_center_of_buoyancy', val=0.0, units='m', desc='z-position of center of column buoyancy force')
        self.add_param('base_column_center_of_mass', val=0.0, units='m', desc='z-position of center of column mass')
        self.add_param('base_column_Px', np.zeros(nFull), units='N/m', desc='force per unit length in x-direction on base')
        self.add_param('base_column_Py', np.zeros(nFull), units='N/m', desc='force per unit length in y-direction on base')
        self.add_param('base_column_Pz', np.zeros(nFull), units='N/m', desc='force per unit length in z-direction on base')
        self.add_param('base_column_qdyn', np.zeros(nFull), units='N/m**2', desc='dynamic pressure on base')

        self.add_param('base_pontoon_attach_upper', val=0.0, units='m', desc='z-value of upper truss attachment on base column')
        self.add_param('base_pontoon_attach_lower', val=0.0, units='m', desc='z-value of lower truss attachment on base column')

        # Ballast columns
        self.add_param('auxiliary_z_full', val=np.zeros((nFull,)), units='m', desc='z-coordinates of section nodes (length = nsection+1)')
        self.add_param('auxiliary_d_full', val=np.zeros((nFull,)), units='m', desc='outer radius at each section node bottom to top (length = nsection + 1)')
        self.add_param('auxiliary_t_full', val=np.zeros((nFull,)), units='m', desc='shell wall thickness at each section node bottom to top (length = nsection + 1)')
        self.add_param('auxiliary_column_mass', val=np.zeros((nFull-1,)), units='kg', desc='mass of ballast column by section')
        self.add_param('auxiliary_column_displaced_volume', val=np.zeros((nFull-1,)), units='m**3', desc='column volume of water displaced by section')
        self.add_param('auxiliary_column_center_of_buoyancy', val=0.0, units='m', desc='z-position of center of column buoyancy force')
        self.add_param('auxiliary_column_center_of_mass', val=0.0, units='m', desc='z-position of center of column mass')
        self.add_param('auxiliary_column_Px', np.zeros(nFull), units='N/m', desc='force per unit length in x-direction on ballast')
        self.add_param('auxiliary_column_Py', np.zeros(nFull), units='N/m', desc='force per unit length in y-direction on ballast')
        self.add_param('auxiliary_column_Pz', np.zeros(nFull), units='N/m', desc='force per unit length in z-direction on ballast')
        self.add_param('auxiliary_column_qdyn', np.zeros(nFull), units='N/m**2', desc='dynamic pressure on ballast')

        self.add_param('fairlead', val=0.0, units='m', desc='Depth below water for mooring line attachment')

        # Tower
        self.add_param('tower_z_full', val=np.zeros((nFull,)), units='m', desc='z-coordinates of section nodes (length = nsection+1)')
        self.add_param('tower_d_full', val=np.zeros((nFull,)), units='m', desc='outer radius at each section node bottom to top (length = nsection + 1)')
        self.add_param('tower_t_full', val=np.zeros((nFull,)), units='m', desc='shell wall thickness at each section node bottom to top (length = nsection + 1)')
        self.add_param('tower_mass', val=np.zeros((nFull-1,)), units='kg', desc='mass of tower column by section')
        self.add_param('tower_buckling_length', 0.0, units='m', desc='buckling length')
        self.add_param('tower_center_of_mass', val=0.0, units='m', desc='z-position of center of tower mass')
        self.add_param('tower_Px', np.zeros(nFull), units='N/m', desc='force per unit length in x-direction on tower')
        self.add_param('tower_Py', np.zeros(nFull), units='N/m', desc='force per unit length in y-direction on tower')
        self.add_param('tower_Pz', np.zeros(nFull), units='N/m', desc='force per unit length in z-direction on tower')
        self.add_param('tower_qdyn', np.zeros(nFull), units='N/m**2', desc='dynamic pressure on tower')
        
        # Semi geometry
        self.add_param('radius_to_auxiliary_column', val=0.0, units='m',desc='Distance from base column centerpoint to ballast column centerpoint')
        self.add_param('number_of_auxiliary_columns', val=3, desc='Number of ballast columns evenly spaced around base column', pass_by_obj=True)

        # Pontoon properties
        self.add_param('pontoon_outer_diameter', val=0.0, units='m',desc='Outer radius of tubular pontoon that connects ballast or base columns')
        self.add_param('pontoon_wall_thickness', val=0.0, units='m',desc='Inner radius of tubular pontoon that connects ballast or base columns')
        self.add_param('cross_attachment_pontoons', val=True, desc='Inclusion of pontoons that connect the bottom of the central base to the tops of the outer ballast columns', pass_by_obj=True)
        self.add_param('lower_attachment_pontoons', val=True, desc='Inclusion of pontoons that connect the central base to the outer ballast columns at their bottoms', pass_by_obj=True)
        self.add_param('upper_attachment_pontoons', val=True, desc='Inclusion of pontoons that connect the central base to the outer ballast columns at their tops', pass_by_obj=True)
        self.add_param('lower_ring_pontoons', val=True, desc='Inclusion of pontoons that ring around outer ballast columns at their bottoms', pass_by_obj=True)
        self.add_param('upper_ring_pontoons', val=True, desc='Inclusion of pontoons that ring around outer ballast columns at their tops', pass_by_obj=True)
        self.add_param('outer_cross_pontoons', val=True, desc='Inclusion of pontoons that ring around outer ballast columns at their tops', pass_by_obj=True)
        
        # Turbine parameters
        self.add_param('rna_mass', val=0.0, units='kg', desc='mass of tower')
        self.add_param('rna_cg', val=np.zeros(3), units='m', desc='Location of RNA center of mass relative to tower top')
        self.add_param('rna_force', val=np.zeros(3), units='N', desc='Force in xyz-direction on turbine')
        self.add_param('rna_moment', val=np.zeros(3), units='N*m', desc='Moments about turbine base')
        self.add_param('rna_I', val=np.zeros(6), units='kg*m**2', desc='Moments about turbine base')

        # safety factors
        self.add_param('gamma_f', 0.0, desc='safety factor on loads')
        self.add_param('gamma_m', 0.0, desc='safety factor on materials')
        self.add_param('gamma_n', 0.0, desc='safety factor on consequence of failure')
        self.add_param('gamma_b', 0.0, desc='buckling safety factor')
        self.add_param('gamma_fatigue', 0.0, desc='total safety factor for fatigue')

        # Manufacturing
        self.add_param('connection_ratio_max', val=0.0, desc='Maximum ratio of pontoon outer diameter to base/ballast outer diameter')
        
        # Costing
        self.add_param('pontoon_cost_rate', val=6.250, units='USD/kg', desc='Finished cost rate of truss components')
        
        # Outputs
        self.add_output('pontoon_cost', val=0.0, units='USD', desc='Cost of pontoon elements and connecting truss')
        self.add_output('pontoon_mass', val=0.0, units='kg', desc='Mass of pontoon elements and connecting truss')
        self.add_output('pontoon_displacement', val=0.0, units='m**3', desc='Buoyancy force of submerged pontoon elements')
        self.add_output('pontoon_center_of_buoyancy', val=0.0, units='m', desc='z-position of center of pontoon buoyancy force')
        self.add_output('pontoon_center_of_mass', val=0.0, units='m', desc='z-position of center of pontoon mass')

        self.add_output('pontoon_stress', val=np.zeros((60,)), desc='Utilization (<1) of von Mises stress by yield stress and safety factor for all pontoon elements')
        self.add_output('tower_stress', np.zeros(nFull-1), desc='Von Mises stress utilization along tower at specified locations.  incudes safety factor.')
        self.add_output('tower_shell_buckling', np.zeros(nFull-1), desc='Shell buckling constraint.  Should be < 1 for feasibility.  Includes safety factors')
        self.add_output('tower_global_buckling', np.zeros(nFull-1), desc='Global buckling constraint.  Should be < 1 for feasibility.  Includes safety factors')
        self.add_output('top_deflection', 0.0, units='m', desc='Deflection of tower top in yaw-aligned +x direction')

        self.add_output('plot_matrix', val=np.array([]), desc='Ratio of shear stress to yield stress for all pontoon elements', pass_by_obj=True)
        self.add_output('base_connection_ratio', val=np.zeros((nFull,)), desc='Ratio of pontoon outer diameter to base outer diameter')
        self.add_output('auxiliary_connection_ratio', val=np.zeros((nFull,)), desc='Ratio of pontoon outer diameter to base outer diameter')
        self.add_output('pontoon_base_attach_upper', val=0.0, desc='Fractional distance along base column for upper truss attachment')
        self.add_output('pontoon_base_attach_lower', val=0.0, desc='Fractional distance along base column for lower truss attachment')

        self.add_output('f1', 0.0, units='Hz', desc='First natural frequency')
        self.add_output('f2', 0.0, units='Hz', desc='Second natural frequency')
        self.add_output('substructure_mass', val=0.0, units='kg', desc='Mass of substructure elements and connecting truss')
        self.add_output('structural_mass', val=0.0, units='kg', desc='Mass of whole turbine except for mooring lines')
        self.add_output('total_displacement', val=0.0, units='m**3', desc='Total volume of water displaced by floating turbine (except for mooring lines)')
        self.add_output('z_center_of_buoyancy', val=0.0, units='m', desc='z-position of center of buoyancy of whole turbine')
        self.add_output('substructure_center_of_mass', val=np.zeros(3), units='m', desc='xyz-position of center of gravity of substructure only')
        self.add_output('center_of_mass', val=np.zeros(3), units='m', desc='xyz-position of center of gravity of whole turbine')
        self.add_output('total_force', val=np.zeros(3), units='N', desc='Net forces on turbine')
        self.add_output('total_moment', val=np.zeros(3), units='N*m', desc='Moments on whole turbine')
        
        # Derivatives
        self.deriv_options['type'] = 'fd'
        self.deriv_options['form'] = 'central'
        self.deriv_options['check_form'] = 'central'
        self.deriv_options['step_calc'] = 'relative'
        self.deriv_options['step_size'] = 1e-5
         
    def solve_nonlinear(self, params, unknowns, resids):
        # Unpack variables
        crossAttachFlag = params['cross_attachment_pontoons']
        lowerAttachFlag = params['lower_attachment_pontoons']
        upperAttachFlag = params['upper_attachment_pontoons']
        lowerRingFlag   = params['lower_ring_pontoons']
        upperRingFlag   = params['upper_ring_pontoons']
        outerCrossFlag  = params['outer_cross_pontoons']
        
        R_semi         = params['radius_to_auxiliary_column']
        R_od_pontoon   = 0.5*params['pontoon_outer_diameter']
        R_od_base      = 0.5*params['base_d_full']
        R_od_ballast   = 0.5*params['auxiliary_d_full']
        R_od_tower     = 0.5*params['tower_d_full']

        t_wall_base    = params['base_t_full']
        t_wall_ballast = params['auxiliary_t_full']
        t_wall_pontoon = params['pontoon_wall_thickness']
        t_wall_tower   = params['tower_t_full']

        E              = params['E']
        G              = params['G']
        rho            = params['material_density']
        sigma_y        = params['yield_stress']
        
        ncolumn        = params['number_of_auxiliary_columns']
        z_base         = params['base_z_full']
        z_ballast      = params['auxiliary_z_full']
        z_tower        = params['tower_z_full']
        z_attach_upper = params['base_pontoon_attach_upper']
        z_attach_lower = params['base_pontoon_attach_lower']
        z_fairlead     = -params['fairlead']
        
        m_base         = params['base_column_mass']
        m_ballast      = params['auxiliary_column_mass']
        m_tower        = params['tower_mass']
        
        m_rna          = params['rna_mass']
        F_rna          = params['rna_force']
        M_rna          = params['rna_moment']
        I_rna          = params['rna_I']
        cg_rna         = params['rna_cg']
        
        rhoWater       = params['water_density']
        
        V_base         = params['base_column_displaced_volume']
        V_ballast      = params['auxiliary_column_displaced_volume']

        z_cb_base      = params['base_column_center_of_buoyancy']
        z_cb_ballast   = params['auxiliary_column_center_of_buoyancy']
        
        cg_base        = np.r_[0.0, 0.0, params['base_column_center_of_mass']]
        cg_ballast     = np.r_[0.0, 0.0, params['auxiliary_column_center_of_mass']]
        cg_tower       = np.r_[0.0, 0.0, params['tower_center_of_mass']]
        
        coeff          = params['pontoon_cost_rate']
        
        gamma_f        = params['gamma_f']
        gamma_m        = params['gamma_m']
        gamma_n        = params['gamma_n']
        gamma_b        = params['gamma_b']
        gamma_fatigue  = params['gamma_fatigue']

        # Quick ratio for unknowns
        unknowns['base_connection_ratio']    = params['connection_ratio_max'] - R_od_pontoon/R_od_base
        unknowns['auxiliary_connection_ratio'] = params['connection_ratio_max'] - R_od_pontoon/R_od_ballast
        unknowns['pontoon_base_attach_upper'] = (z_attach_upper - z_base[0]) / (z_base[-1] - z_base[0]) #0.5<x<1.0
        unknowns['pontoon_base_attach_lower'] = (z_attach_lower - z_base[0]) / (z_base[-1] - z_base[0]) #0.0<x<0.5
        
        # ---NODES---
        # Senu TODO: Should tower and rna have nodes at their CGs?
        # Senu TODO: Mooring tension on column nodes?

        # Add nodes for base column: Using 4 nodes/3 elements per section
        # Make sure there is a node at upper and lower attachment points
        baseBeginID = 0 + 1
        if ncolumn > 0:
            idx = find_nearest(z_base, z_attach_lower)
            z_base[idx] = z_attach_lower
            baseLowerID = idx + 1
            
            idx = find_nearest(z_base, z_attach_upper)
            z_base[idx] = z_attach_upper
            baseUpperID = idx + 1
        
        baseEndID = z_base.size
        freeboard = z_base[-1]

        fairleadID  = []
        # Need reaction attachment point if just running a spar
        if ncolumn == 0:
            idx = find_nearest(z_base, z_fairlead)
            z_base[idx] = z_fairlead
            fairleadID.append( idx + 1 )
        
        znode = np.copy( z_base )
        xnode = np.zeros(znode.shape)
        ynode = np.zeros(znode.shape)

        towerBeginID = baseEndID
        myz = np.zeros(len(z_tower)-1)
        xnode = np.append(xnode, myz)
        ynode = np.append(ynode, myz)
        znode = np.append(znode, z_tower[1:] + freeboard )
        towerEndID = xnode.size

        # Create dummy node so that the tower isn't the last in a chain.
        # This avoids a Frame3DD bug
        dummyID = xnode.size + 1
        xnode = np.append(xnode, 0.0)
        ynode = np.append(ynode, 0.0)
        znode = np.append(znode, znode[-1]+1.0 )
        
        # Get x and y positions of surrounding ballast columns
        ballastLowerID = []
        ballastUpperID = []
        ballastx = R_semi * np.cos( np.linspace(0, 2*np.pi, ncolumn+1) )
        ballasty = R_semi * np.sin( np.linspace(0, 2*np.pi, ncolumn+1) )
        ballastx = ballastx[:-1]
        ballasty = ballasty[:-1]

        # Add in ballast column nodes around the circle, make sure there is a node at the fairlead
        idx = find_nearest(z_ballast, z_fairlead)
        myones = np.ones(z_ballast.shape)
        for k in xrange(ncolumn):
            ballastLowerID.append( xnode.size + 1 )
            fairleadID.append( xnode.size + idx + 1 )
            xnode = np.append(xnode, ballastx[k]*myones)
            ynode = np.append(ynode, ballasty[k]*myones)
            znode = np.append(znode, z_ballast )
            ballastUpperID.append( xnode.size )

        # Add nodes midway around outer ring for cross bracing
        if outerCrossFlag and ncolumn > 0:
            crossx = 0.5*(ballastx + np.roll(ballastx,1))
            crossy = 0.5*(ballasty + np.roll(ballasty,1))

            crossOuterLowerID = xnode.size + np.arange(ncolumn) + 1
            xnode = np.append(xnode, crossx)
            ynode = np.append(ynode, crossy)
            znode = np.append(znode, z_ballast[0]*np.ones(ncolumn))

            #crossOuterUpperID = xnode.size + np.arange(ncolumn) + 1
            #xnode = np.append(xnode, crossx)
            #ynode = np.append(ynode, crossy)
            #znode = np.append(znode, z_ballast[-1]*np.ones(ncolumn))

        # Create Node Data object
        nnode = 1 + np.arange(xnode.size)
        rnode = np.zeros(xnode.shape)
        nodes = frame3dd.NodeData(nnode, xnode, ynode, znode, rnode)

        
        # ---REACTIONS---
        # Pin (3DOF) the nodes at the mooring connections.  Otherwise free
        # Free=0, Rigid=1
        rid = np.array(fairleadID)
        Rx = Ry = Rz = Rxx = Ryy = Rzz = np.ones(rid.shape)
        #if ncolumn > 0:
        #    Rxx[1:] = Ryy[1:] = Rzz[1:] = 0.0
        # First approach
        # Pinned windward column lower node (first ballastLowerID)
        #rid = ballastLowerID[0]
        #Rx = Ry = Rz = Rxx = Ryy = Rzz = 1
        # Rollers for other lower column nodes, restrict motion
        #rid = ballastLowerID[1:]
        #Rz = Rxx = Ryy = Rzz = 1

        # Get reactions object from frame3dd
        reactions = frame3dd.ReactionData(rid, Rx, Ry, Rz, Rxx, Ryy, Rzz, rigid=1)


        # ---ELEMENTS / EDGES---
        N1 = np.array([], dtype=np.int32)
        N2 = np.array([], dtype=np.int32)
        # Lower connection from central base column to ballast columns
        if lowerAttachFlag:
            lowerAttachEID = N1.size + 1
            for k in xrange(ncolumn):
                N1 = np.append(N1, baseLowerID )
                N2 = np.append(N2, ballastLowerID[k] )
        # Upper connection from central base column to ballast columns
        if upperAttachFlag:
            upperAttachEID = N1.size + 1
            for k in xrange(ncolumn):
                N1 = np.append(N1, baseUpperID )
                N2 = np.append(N2, ballastUpperID[k] )
        # Cross braces from lower central base column to upper ballast columns
        if crossAttachFlag:
            crossAttachEID = N1.size + 1
            for k in xrange(ncolumn):
                N1 = np.append(N1, baseLowerID )
                N2 = np.append(N2, ballastUpperID[k] )
            # Will be used later to convert from local member c.s. to global
            cross_angle = np.arctan( (z_attach_upper - z_attach_lower) / R_semi )
        # Lower ring around ballast columns
        if lowerRingFlag:
            lowerRingEID = N1.size + 1
            for k in xrange(ncolumn-1):
                N1 = np.append(N1, ballastLowerID[k] )
                N2 = np.append(N2, ballastLowerID[k+1] )
            N1 = np.append(N1, ballastLowerID[0] )
            N2 = np.append(N2, ballastLowerID[-1] )
        # Upper ring around ballast columns
        if upperRingFlag:
            upperRingEID = N1.size + 1
            for k in xrange(ncolumn-1):
                N1 = np.append(N1, ballastUpperID[k] )
                N2 = np.append(N2, ballastUpperID[k+1] )
            N1 = np.append(N1, ballastUpperID[0] )
            N2 = np.append(N2, ballastUpperID[-1] )
        # Outer cross braces
        if outerCrossFlag:
            outerCrossEID = N1.size + 1
            for k in xrange(ncolumn-1):
                N1 = np.append(N1, crossOuterLowerID[k] )
                N2 = np.append(N2, ballastUpperID[k] )
                N1 = np.append(N1, crossOuterLowerID[k+1] )
                N2 = np.append(N2, ballastUpperID[k] )
            N1 = np.append(N1, crossOuterLowerID[-1] )
            N2 = np.append(N2, ballastUpperID[-1] )
            N1 = np.append(N1, crossOuterLowerID[0] )
            N2 = np.append(N2, ballastUpperID[-1] )
        # TODO: Parameterize these for upper, lower, cross connections
        # Properties for the inner connectors
        mytube = Tube(2.0*R_od_pontoon, t_wall_pontoon)
        Ax    = mytube.Area * np.ones(N1.shape)
        As    = mytube.Asx  * np.ones(N1.shape)
        Jx    = mytube.J0   * np.ones(N1.shape)
        I     = mytube.Jxx  * np.ones(N1.shape)
        S     = mytube.S    * np.ones(N1.shape)
        C     = mytube.C    * np.ones(N1.shape)
        modE  = E   * np.ones(N1.shape)
        modG  = G   * np.ones(N1.shape)
        roll  = 0.0 * np.ones(N1.shape)
        dens  = rho * np.ones(N1.shape)

        # Now mock up cylindrical columns as truss members even though long, slender assumption breaks down
        # Will set density = 0.0 so that we don't double count the mass
        # First get geometry in each of the elements
        R_od_base,_      = nodal2sectional( R_od_base )
        t_wall_base,_    = nodal2sectional( t_wall_base )
        R_od_ballast,_   = nodal2sectional( R_od_ballast )
        t_wall_ballast,_ = nodal2sectional( t_wall_ballast )
        R_od_tower,_     = nodal2sectional( R_od_tower )
        t_wall_tower,_   = nodal2sectional( t_wall_tower )
        # Senu TODO: Make artificially more stiff?
        baseEID = N1.size + 1
        mytube  = Tube(2.0*R_od_base, t_wall_base)
        myrange = np.arange(R_od_base.size)
        myones  = np.ones(myrange.shape)
        mydens  = m_base / mytube.Area / np.diff(z_base) + eps
        N1   = np.append(N1  , myrange + baseBeginID    )
        N2   = np.append(N2  , myrange + baseBeginID + 1)
        Ax   = np.append(Ax  , mytube.Area )
        As   = np.append(As  , mytube.Asx )
        Jx   = np.append(Jx  , mytube.J0 )
        I    = np.append(I   , mytube.Jxx )
        S    = np.append(S   , mytube.S )
        C    = np.append(C   , mytube.C )
        modE = np.append(modE, E*myones )
        modG = np.append(modG, G*myones )
        roll = np.append(roll, np.zeros(myones.shape) )
        dens = np.append(dens, mydens )

        # Rest of tower
        towerEID = N1.size + 1
        myrange = np.arange(R_od_tower.size)
        myones  = np.ones(myrange.shape)
        mydens  = m_tower / mytube.Area / np.diff(z_tower) + eps
        N1   = np.append(N1  , myrange + towerBeginID    )
        N2   = np.append(N2  , myrange + towerBeginID + 1)
        Ax   = np.append(Ax  , mytube.Area )
        As   = np.append(As  , mytube.Asx )
        Jx   = np.append(Jx  , mytube.J0 )
        I    = np.append(I   , mytube.Jxx )
        S    = np.append(S   , mytube.S )
        C    = np.append(C   , mytube.C )
        modE = np.append(modE, E*myones )
        modG = np.append(modG, G*myones )
        roll = np.append(roll, np.zeros(myones.shape) )
        dens = np.append(dens, mydens ) 

        # Dummy element
        dummyEID = N1.size + 1
        N1   = np.append(N1  , towerEndID )
        N2   = np.append(N2  , dummyID )
        Ax   = np.append(Ax  , Ax[-1] )
        As   = np.append(As  , As[-1] )
        Jx   = np.append(Jx  , Jx[-1] )
        I    = np.append(I   , I[-1] )
        S    = np.append(S   , S[-1] )
        C    = np.append(C   , C[-1] )
        modE = np.append(modE, 1e20 )
        modG = np.append(modG, 1e20 )
        roll = np.append(roll, 0.0 )
        dens = np.append(dens, 1e-6 ) 
        
        ballastEID = []
        mytube     = Tube(2.0*R_od_ballast, t_wall_ballast)
        myrange    = np.arange(R_od_ballast.size)
        myones     = np.ones(myrange.shape)
        mydens     = m_ballast / mytube.Area / np.diff(z_ballast) + eps
        for k in xrange(ncolumn):
            ballastEID.append( N1.size + 1 )
            
            N1   = np.append(N1  , myrange + ballastLowerID[k]    )
            N2   = np.append(N2  , myrange + ballastLowerID[k] + 1)
            Ax   = np.append(Ax  , mytube.Area )
            As   = np.append(As  , mytube.Asx )
            Jx   = np.append(Jx  , mytube.J0 )
            I    = np.append(I   , mytube.Jxx )
            S    = np.append(S   , mytube.S )
            C    = np.append(C   , mytube.C )
            modE = np.append(modE, E*myones )
            modG = np.append(modG, G*myones )
            roll = np.append(roll, np.zeros(myones.shape) )
            dens = np.append(dens, mydens ) # Mass added below


        # ---Get element object from frame3dd---
        nelem    = 1 + np.arange(N1.size)
        elements = frame3dd.ElementData(nelem, N1, N2, Ax, As, As, Jx, I, I, modE, modG, roll, dens)

        # Store data for plotting, also handy for operations below
        plotMat = np.zeros((nelem.size, 3, 2))
        plotMat[:,:,0] = np.c_[xnode[N1-1], ynode[N1-1], znode[N1-1]]
        plotMat[:,:,1] = np.c_[xnode[N2-1], ynode[N2-1], znode[N2-1]]
        
        # Compute length and center of gravity for each element for use below
        elemL   = np.sqrt( np.sum( np.diff(plotMat, axis=2)**2.0, axis=1) ).flatten()
        elemCoG = 0.5*np.sum(plotMat, axis=2)

        # ---Options object---
        shear = True               # 1: include shear deformation
        geom = False               # 1: include geometric stiffness
        dx = -1                    # x-axis increment for internal forces, -1 to skip
        other = frame3dd.Options(shear, geom, dx)

        # Initialize frame3dd object
        myframe = frame3dd.Frame(nodes, reactions, elements, other)

        # Add in extra mass of rna
        inode   = np.array([towerEndID], dtype=np.int32) # rna
        m_extra = np.array([m_rna])
        Ixx = np.array([ I_rna[0] ])
        Iyy = np.array([ I_rna[1] ])
        Izz = np.array([ I_rna[2] ])
        Ixy = np.array([ I_rna[3] ])
        Ixz = np.array([ I_rna[4] ])
        Iyz = np.array([ I_rna[5] ])
        rhox = np.array([ cg_rna[0] ])
        rhoy = np.array([ cg_rna[1] ])
        rhoz = np.array([ cg_rna[2] ])
        myframe.changeExtraNodeMass(inode, m_extra, Ixx, Iyy, Izz, Ixy, Ixz, Iyz, rhox, rhoy, rhoz, True)

        # ---LOAD CASES---
        # Extreme loading
        gx = 0.0
        gy = 0.0
        gz = -gravity
        load = frame3dd.StaticLoadCase(gx, gy, gz)

        # Wind + Wave loading in local base / ballast / tower c.s.
        Px_base,    Py_base,    Pz_base    = params['base_column_Pz'], params['base_column_Py'], -params['base_column_Px']  # switch to local c.s.
        Px_ballast, Py_ballast, Pz_ballast = params['auxiliary_column_Pz'], params['auxiliary_column_Py'], -params['auxiliary_column_Px']  # switch to local c.s.
        Px_tower,   Py_tower,   Pz_tower   = params['tower_Pz'], params['tower_Py'], -params['tower_Px']  # switch to local c.s.
        epsOff = 1e-6
        # Get mass right- ballasts, stiffeners, tower, rna, etc.
        # Also account for buoyancy loads
        # Also apply wind/wave loading as trapezoidal on each element
        # NOTE: Loading is in local element coordinates 0-L, x is along element
        # Base
        nrange  = np.arange(R_od_base.size, dtype=np.int32)
        EL      = baseEID + nrange
        Ux      = V_base * rhoWater * gravity / np.diff(z_base)
        x1 = np.zeros(nrange.shape)
        x2 = np.diff(z_base) - epsOff  # subtract small number b.c. of precision
        wx1, wx2 = Px_base[:-1], Px_base[1:]
        wy1, wy2 = Py_base[:-1], Py_base[1:]
        wz1, wz2 = Pz_base[:-1], Pz_base[1:]
        # Tower
        nrange  = np.arange(R_od_tower.size, dtype=np.int32)
        EL      = np.append(EL, towerEID + nrange)
        Ux      = np.append(Ux, np.zeros(nrange.shape))
        x1      = np.append(x1, np.zeros(nrange.shape))
        x2      = np.append(x2, np.diff(z_tower) - epsOff)
        wx1     = np.append(wx1, Px_tower[:-1])
        wx2     = np.append(wx2, Px_tower[1:])
        wy1     = np.append(wy1, Py_tower[:-1])
        wy2     = np.append(wy2, Py_tower[1:])
        wz1     = np.append(wz1, Pz_tower[:-1])
        wz2     = np.append(wz2, Pz_tower[1:])
        # Buoyancy- ballast columns
        nrange  = np.arange(R_od_ballast.size, dtype=np.int32)
        for k in xrange(ncolumn):
            EL      = np.append(EL, ballastEID[k] + nrange)
            Ux      = np.append(Ux,  V_ballast * rhoWater * gravity / np.diff(z_ballast) )
            x1      = np.append(x1, np.zeros(nrange.shape))
            x2      = np.append(x2, np.diff(z_ballast) - epsOff)
            wx1     = np.append(wx1, Px_ballast[:-1])
            wx2     = np.append(wx2, Px_ballast[1:])
            wy1     = np.append(wy1, Py_ballast[:-1])
            wy2     = np.append(wy2, Py_ballast[1:])
            wz1     = np.append(wz1, Pz_ballast[:-1])
            wz2     = np.append(wz2, Pz_ballast[1:])
            
        # Add mass of base and ballast columns while we've already done the element enumeration
        Uz = Uy = np.zeros(Ux.shape)
        load.changeUniformLoads(EL, Ux, Uy, Uz)
        xx1 = xy1 = xz1 = x1
        xx2 = xy2 = xz2 = x2
        load.changeTrapezoidalLoads(EL, xx1, xx2, wx1, wx2, xy1, xy2, wy1, wy2, xz1, xz2, wz1, wz2)

        # Buoyancy for fully submerged members
        # Note indices to elemL and elemCoG could include -1, but since there is assumed to be more than 1 column, this is not necessary
        nrange  = np.arange(ncolumn, dtype=np.int32)
        Frange  = np.pi * R_od_pontoon**2 * rhoWater * gravity
        F_truss = 0.0
        z_cb    = np.zeros((3,))
        if ncolumn > 0 and znode[ballastLowerID[0]-1] < 0.0:
            if lowerAttachFlag:
                EL       = lowerAttachEID + nrange
                Uz       = Frange * np.ones(nrange.shape)
                F_truss += Frange * elemL[lowerAttachEID-1] * ncolumn
                z_cb    += Frange * elemL[lowerAttachEID-1] * ncolumn * elemCoG[lowerAttachEID-1,:]
                Ux = Uy = np.zeros(Uz.shape)
                load.changeUniformLoads(EL, Ux, Uy, Uz)
            if lowerRingFlag:
                EL       = lowerRingEID + nrange
                Uz       = Frange * np.ones(nrange.shape)
                F_truss += Frange * elemL[lowerRingEID-1] * ncolumn
                z_cb    += Frange * elemL[lowerRingEID-1] * ncolumn * elemCoG[lowerRingEID-1]
                Ux = Uy = np.zeros(Uz.shape)
                load.changeUniformLoads(EL, Ux, Uy, Uz)
            if crossAttachFlag:
                factor   = np.minimum(1.0, (0.0 - z_attach_lower) / (znode[ballastUpperID[0]-1] - z_attach_lower) )
                EL       = crossAttachEID + nrange
                Ux       = factor * Frange * np.sin(cross_angle) * np.ones(nrange.shape)
                Uz       = factor * Frange * np.cos(cross_angle) * np.ones(nrange.shape)
                F_truss += factor * Frange * elemL[crossAttachEID-1] * ncolumn
                z_cb    += factor * Frange * elemL[crossAttachEID-1] * ncolumn * elemCoG[crossAttachEID-1,:]
                Uy = np.zeros(Uz.shape)
                load.changeUniformLoads(EL, Ux, Uy, Uz)
            if outerCrossFlag:
                factor   = np.minimum(1.0, (0.0 - znode[baseLowerID-1]) / (znode[ballastUpperID[0]-1] - znode[baseLowerID-1]) )
                # TODO: This one will take a little more math
                #EL       = outerCrossEID + np.arange(2*ncolumn, dtype=np.int32) 
                #Uz       = factor * Frange * np.ones(nrange.shape)
                F_truss += factor * Frange * elemL[outerCrossEID-1] * ncolumn
                z_cb    += factor * Frange * elemL[outerCrossEID-1] * ncolumn * elemCoG[outerCrossEID-1,:]
                #Ux = Uy = np.zeros(Uz.shape)
                #load.changeUniformLoads(EL, Ux, Uy, Uz)
        if ncolumn > 0 and znode[ballastUpperID[0]-1] < 0.0:
            if upperAttachFlag:
                EL       = upperAttachEID + nrange
                Uz       = Frange * np.ones(nrange.shape)
                F_truss += Frange * elemL[upperAttachEID-1] * ncolumn
                z_cb    += Frange * elemL[upperAttachEID-1] * ncolumn * elemCoG[upperAttachEID-1,:]
                Ux = Uy = np.zeros(Uz.shape)
                load.changeUniformLoads(EL, Ux, Uy, Uz)
            if upperRingFlag:
                EL       = upperRingEID + nrange
                Uz       = Frange * np.ones(nrange.shape)
                F_truss += Frange * elemL[upperRingEID-1] * ncolumn
                z_cb    += Frange * elemL[upperRingEID-1] * ncolumn * elemCoG[upperRingEID-1,:]
                Ux = Uy = np.zeros(Uz.shape)
                load.changeUniformLoads(EL, Ux, Uy, Uz)

        # Point loading for rotor thrust and wind loads at CG
        # Note: extra momemt from mass accounted for below
        nF  = np.array([ baseEndID ], dtype=np.int32)
        Fx  = np.array([ F_rna[0] ])
        Fy  = np.array([ F_rna[1] ])
        Fz  = np.array([ F_rna[2] ])
        Mxx = np.array([ M_rna[0] ])
        Myy = np.array([ M_rna[1] ])
        Mzz = np.array([ M_rna[2] ])
        load.changePointLoads(nF, Fx, Fy, Fz, Mxx, Myy, Mzz)

        # Store load case into frame 3dd object
        myframe.addLoadCase(load)


        # ---DYNAMIC ANALYSIS---
        nM = 6              # number of desired dynamic modes of vibration
        Mmethod = 1         # 1: subspace Jacobi     2: Stodola
        lump = 0            # 0: consistent mass ... 1: lumped mass matrix
        tol = 1e-5          # mode shape tolerance
        shift = 0.0         # shift value ... for unrestrained structures
        
        myframe.enableDynamics(nM, Mmethod, lump, tol, shift)


        # ---RUN ANALYSIS---
        #myframe.write('debug.3dd') # For debugging
        displacements, forces, reactions, internalForces, mass, modal = myframe.run()
        
        # --OUTPUTS--
        nE    = nelem.size
        iCase = 0
        unknowns['plot_matrix'] = plotMat
        
        if ncolumn > 0:
            # Buoyancy assembly from incremental calculations above
            V_pontoon = F_truss/rhoWater/gravity
            z_cb      = z_cb[-1] / F_truss
            unknowns['pontoon_displacement'] = V_pontoon
            unknowns['pontoon_center_of_buoyancy'] = z_cb

            # Sum up mass and compute CofG.  Frame3DD does mass, but not CG
            # TODO: Subtract out extra pontoon length that overlaps with column radii
            ind = baseEID-1
            m_total = Ax[:ind] * rho * elemL[:ind]
            m_pontoon = m_total.sum() #mass.struct_mass
            cg_pontoon = np.sum( m_total[:,np.newaxis] * elemCoG[:ind,:], axis=0 ) / m_total.sum()
            unknowns['pontoon_mass'] = m_pontoon
            unknowns['pontoon_cost'] = coeff * m_pontoon
            unknowns['pontoon_center_of_mass'] = cg_pontoon[-1]
        else:
            V_pontoon = z_cb = m_pontoon = 0.0
            cg_pontoon = np.zeros(3)
            
        # natural frequncies
        unknowns['f1'] = modal.freq[0]
        unknowns['f2'] = modal.freq[1]

        # deflections due to loading (from cylinder top and wind/wave loads)
        unknowns['top_deflection'] = displacements.dx[iCase, towerEndID-1]  # in yaw-aligned direction

        # Summary of mass and volumes
        unknowns['substructure_mass']  = m_pontoon + m_base.sum() + ncolumn*m_ballast.sum()
        unknowns['structural_mass']    = mass.total_mass
        unknowns['total_displacement'] = V_base.sum() + ncolumn*V_ballast.sum() + V_pontoon

        # Find cb (center of buoyancy) for whole system
        z_cb = (V_base.sum()*z_cb_base + ncolumn*V_ballast.sum()*z_cb_ballast + V_pontoon*z_cb) / unknowns['total_displacement']
        unknowns['z_center_of_buoyancy'] = z_cb

        # Find cg (center of gravity) for whole system
        unknowns['substructure_center_of_mass'] = (ncolumn*m_ballast.sum()*cg_ballast + m_base.sum()*cg_base +
                                                   m_pontoon*cg_pontoon) / unknowns['substructure_mass']
        unknowns['center_of_mass'] = (m_rna*cg_rna + m_tower.sum()*cg_tower +
                                      unknowns['substructure_mass']*unknowns['substructure_center_of_mass']) / mass.total_mass
        Fsum = np.zeros(3)
        Msum = np.zeros(3)
        for k in xrange(len(rid)):
            idx = reactions.node[iCase, k] - 1
            pk  = np.array([xnode[idx], ynode[idx], znode[idx]])
            rk  = pk - unknowns['center_of_mass']
            F   = -1*np.array([reactions.Fx[iCase, k], reactions.Fy[iCase, k], reactions.Fz[iCase, k]])
            M   = -1*np.array([reactions.Mxx[iCase, k], reactions.Myy[iCase, k], reactions.Mzz[iCase, k]])
            Fsum += F
            Msum += M + np.cross(rk,F)
        unknowns['total_force'] = -1.0 * np.array([reactions.Fx.sum(), reactions.Fy.sum(), reactions.Fz.sum()])
        unknowns['total_moment'] = -1.0 * np.array([reactions.Mxx.sum(), reactions.Myy.sum(), reactions.Mzz.sum()])

        # shear and bending (convert from local to global c.s.)
        Nx = forces.Nx[iCase, 1::2]
        Vy = forces.Vy[iCase, 1::2]
        Vz = forces.Vz[iCase, 1::2]

        Tx = forces.Txx[iCase, 1::2]
        My = forces.Myy[iCase, 1::2]
        Mz = forces.Mzz[iCase, 1::2]

        # Compute axial and shear stresses in elements given Frame3DD outputs and some geomtry data
        # Method comes from Section 7.14 of Frame3DD documentation
        # http://svn.code.sourceforge.net/p/frame3dd/code/trunk/doc/Frame3DD-manual.html#structuralmodeling
        M = np.sqrt(My*My + Mz*Mz)
        sigma_ax = Nx/Ax - M/S
        sigma_sh = np.sqrt(Vy*Vy + Vz*Vz)/As + Tx/C

        # Extract pontoon for stress check
        idx  = range(baseEID-1)
        npon = len(idx)
        if len(idx) > 0:
            qdyn_pontoon = np.max( np.abs( np.r_[params['base_column_qdyn'], params['auxiliary_column_qdyn']] ) )
            sigma_ax_pon = sigma_ax[idx]
            sigma_sh_pon = sigma_sh[idx]
            sigma_h_pon  = util.hoopStress(2*R_od_pontoon, t_wall_pontoon, qdyn_pontoon) * np.ones(sigma_ax_pon.shape)

            unknowns['pontoon_stress'][:npon] = util.vonMisesStressUtilization(sigma_ax_pon, sigma_h_pon, sigma_sh_pon,
                                                                               gamma_f*gamma_m*gamma_n, sigma_y)
        
        # Extract tower for Eurocode checks
        idx = towerEID-1 + np.arange(R_od_tower.size, dtype=np.int32)
        L_reinforced   = params['tower_buckling_length'] * np.ones(idx.shape)
        sigma_ax_tower = sigma_ax[idx]
        sigma_sh_tower = sigma_sh[idx]
        qdyn_tower,_   = nodal2sectional( params['tower_qdyn'] )
        sigma_h_tower  = util.hoopStressEurocode(z_tower, 2*R_od_tower, t_wall_tower, L_reinforced, qdyn_tower)

        unknowns['tower_stress'] = util.vonMisesStressUtilization(sigma_ax_tower, sigma_h_tower, sigma_sh_tower,
                                                                  gamma_f*gamma_m*gamma_n, sigma_y)

        sigma_y = sigma_y * np.ones(idx.shape)
        unknowns['tower_shell_buckling'] = util.shellBucklingEurocode(2*R_od_tower, t_wall_tower, sigma_ax_tower, sigma_h_tower, sigma_sh_tower,
                                                                      L_reinforced, modE[idx], sigma_y, gamma_f, gamma_b)

        tower_height = z_tower[-1] - z_tower[0]
        unknowns['tower_global_buckling'] = util.bucklingGL(2*R_od_tower, t_wall_tower, Nx[idx], M[idx], tower_height, modE[idx], sigma_y, gamma_f, gamma_b)
        # TODO: FATIGUE
        # Base and ballast columns get API stress/buckling checked in Column Group because that takes into account stiffeners



# -----------------
#  Assembly
# -----------------

class FloatingLoading(Group):

    def __init__(self, nSection, nFull):
        super(FloatingLoading, self).__init__()
        
        # Independent variables that are unique to TowerSE
        self.add('base_pontoon_attach_lower',  IndepVarComp('base_pontoon_attach_lower', 0.0), promotes=['*'])
        self.add('base_pontoon_attach_upper',  IndepVarComp('base_pontoon_attach_upper', 0.0), promotes=['*'])
        self.add('pontoon_outer_diameter',     IndepVarComp('pontoon_outer_diameter', 0.0), promotes=['*'])
        self.add('pontoon_wall_thickness',     IndepVarComp('pontoon_wall_thickness', 0.0), promotes=['*'])
        self.add('outer_cross_pontoons',       IndepVarComp('outer_cross_pontoons', True, pass_by_obj=True), promotes=['*'])
        self.add('cross_attachment_pontoons',  IndepVarComp('cross_attachment_pontoons', True, pass_by_obj=True), promotes=['*'])
        self.add('lower_attachment_pontoons',  IndepVarComp('lower_attachment_pontoons', True, pass_by_obj=True), promotes=['*'])
        self.add('upper_attachment_pontoons',  IndepVarComp('upper_attachment_pontoons', True, pass_by_obj=True), promotes=['*'])
        self.add('lower_ring_pontoons',        IndepVarComp('lower_ring_pontoons', True, pass_by_obj=True), promotes=['*'])
        self.add('upper_ring_pontoons',        IndepVarComp('upper_ring_pontoons', True, pass_by_obj=True), promotes=['*'])
        self.add('pontoon_cost_rate',          IndepVarComp('pontoon_cost_rate', 0.0), promotes=['*'])
        self.add('connection_ratio_max',       IndepVarComp('connection_ratio_max', 0.0), promotes=['*'])

        # All the components
        self.add('wind', PowerWind(nFull), promotes=['z0','Uref','shearExp','zref'])
        self.add('windLoads', CylinderWindDrag(nFull), promotes=['cd_usr','beta'])

        self.add('frame', FloatingFrame(nFull), promotes=['*'])
        
        # Connections for geometry and mass
        self.connect('wind.z', ['windLoads.z', 'tower_z_full'])
        self.connect('windLoads.d', ['tower_d_full'])
        self.connect('wind.U', 'windLoads.U')

        # connections to distLoads1
        self.connect('windLoads.windLoads:Px', 'tower_Px')
        self.connect('windLoads.windLoads:Py', 'tower_Py')
        self.connect('windLoads.windLoads:Pz', 'tower_Pz')
        self.connect('windLoads.windLoads:qdyn', 'tower_qdyn')

