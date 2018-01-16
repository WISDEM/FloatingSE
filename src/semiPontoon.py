import collections
from openmdao.api import Component
import numpy as np
import frame3dd
from floatingInstance import nodal2sectional, NSECTIONS

from constants import gravity
NPTS = 100

def TubeProperties(R_od, R_id):
    # Cross sectional area, Ax (x is down length of tube)
    Ax = np.pi*(R_od**2.0 - R_id**2.0)
    # Shear area (see section 7.4.5 of Frame3dd documentation)
    # http://svn.code.sourceforge.net/p/frame3dd/code/trunk/doc/Frame3DD-manual.html#structuralmodeling
    As = Ax / ( 0.54414 + 2.97294*(R_id/R_od) - 1.51899*(R_id/R_od)**2.0 )
    # Moment of inertia
    I = 0.25 * np.pi * (R_od**4.0 - R_id**4.0)
    # 0-checks
    Ax = np.maximum(1e-16, Ax)
    As = np.maximum(1e-16, As)
    I  = np.maximum(1e-16, I)
    # Other properties for tubular sections
    J = 2.0 * I
    S = I / R_od
    C = J / R_od
    return Ax, As, I, J, S, C


class SemiPontoon(Component):
    """
    OpenMDAO Component class for semisubmersible pontoon / truss structure for floating offshore wind turbines.
    Should be tightly coupled with Semi and Mooring classes for full system representation.
    """

    def __init__(self):
        super(SemiPontoon,self).__init__()

        self.frame = None

        # Environment
        self.add_param('water_density', val=1025.0, units='kg/m**3', desc='density of water')

        # Material properties
        self.add_param('material_density', val=7850., units='kg/m**3', desc='density of material')
        self.add_param('E', val=200.e9, units='Pa', desc='Modulus of elasticity (Youngs) of material')
        self.add_param('G', val=79.3e9, units='Pa', desc='Shear modulus of material')
        self.add_param('yield_stress', val=345e6, units='Pa', desc='yield stress of material')

        # Base cylinder
        self.add_param('base_z_nodes', val=np.zeros((NSECTIONS+1,)), units='m', desc='z-coordinates of section nodes (length = nsection+1)')
        self.add_param('base_outer_radius', val=np.zeros((NSECTIONS+1,)), units='m', desc='outer radius at each section node bottom to top (length = nsection + 1)')
        self.add_param('base_wall_thickness', val=np.zeros((NSECTIONS+1,)), units='m', desc='shell wall thickness at each section node bottom to top (length = nsection + 1)')
        self.add_param('base_cylinder_mass', val=np.zeros((NSECTIONS,)), units='kg', desc='mass of base cylinder by section')
        self.add_param('base_cylinder_displaced_volume', val=np.zeros((NSECTIONS,)), units='m**3', desc='cylinder volume of water displaced by section')
        #self.add_param('base_cylinder_surge_force', val=np.zeros((NPTS,)), units='N', desc='Force vector in surge direction on cylinder')
        #self.add_param('base_cylinder_force_points', val=np.zeros((NPTS,)), units='m', desc='zpts for force vector')

        # Ballast cylinders
        self.add_param('ballast_z_nodes', val=np.zeros((NSECTIONS+1,)), units='m', desc='z-coordinates of section nodes (length = nsection+1)')
        self.add_param('ballast_outer_radius', val=np.zeros((NSECTIONS+1,)), units='m', desc='outer radius at each section node bottom to top (length = nsection + 1)')
        self.add_param('ballast_wall_thickness', val=np.zeros((NSECTIONS+1,)), units='m', desc='shell wall thickness at each section node bottom to top (length = nsection + 1)')
        self.add_param('ballast_cylinder_mass', val=np.zeros((NSECTIONS,)), units='kg', desc='mass of ballast cylinder by section')
        self.add_param('ballast_cylinder_displaced_volume', val=np.zeros((NSECTIONS,)), units='m**3', desc='cylinder volume of water displaced by section')
        #self.add_param('ballast_cylinder_surge_force', val=np.zeros((NPTS,)), units='N', desc='Force vector in surge direction on cylinder')
        #self.add_param('ballast_cylinder_force_points', val=np.zeros((NPTS,)), units='m', desc='zpts for force vector')

        # Semi geometry
        self.add_param('radius_to_ballast_cylinder', val=10.0, units='m',desc='Distance from base cylinder centerpoint to ballast cylinder centerpoint')
        self.add_param('number_of_ballast_cylinders', val=3, desc='Number of ballast cylinders evenly spaced around base cylinder', pass_by_obj=True)

        # Pontoon properties
        self.add_param('outer_pontoon_radius', val=0.5, units='m',desc='Outer radius of tubular pontoon that connects ballast or base cylinders')
        self.add_param('inner_pontoon_radius', val=0.45, units='m',desc='Inner radius of tubular pontoon that connects ballast or base cylinders')
        self.add_param('cross_attachment_pontoons', val=True, desc='Inclusion of pontoons that connect the bottom of the central base to the tops of the outer ballast columns', pass_by_obj=True)
        self.add_param('lower_attachment_pontoons', val=True, desc='Inclusion of pontoons that connect the central base to the outer ballast columns at their bottoms', pass_by_obj=True)
        self.add_param('upper_attachment_pontoons', val=True, desc='Inclusion of pontoons that connect the central base to the outer ballast columns at their tops', pass_by_obj=True)
        self.add_param('lower_ring_pontoons', val=True, desc='Inclusion of pontoons that ring around outer ballast columns at their bottoms', pass_by_obj=True)
        self.add_param('upper_ring_pontoons', val=True, desc='Inclusion of pontoons that ring around outer ballast columns at their tops', pass_by_obj=True)
        
        # Turbine parameters
        self.add_param('tower_mass', val=0.0, units='kg', desc='mass of tower')
        self.add_param('turbine_surge_force', val=np.zeros((2,)), units='N', desc='Force in surge direction on turbine')
        self.add_param('turbine_force_points', val=np.zeros((2,)), units='m', desc='zpts for force vector')
        self.add_param('tower_base_radius', val=3.25, units='m', desc='outer radius of tower at base')
        self.add_param('rna_mass', val=1e5, units='kg', desc='Mass of rotor nacelle assembly')
        self.add_param('rna_center_of_gravity_x', val=1.0, units='m', desc='Center of gravity along x-axis measured from tower centerline')

        # Costing
        self.add_param('pontoon_cost_rate', val=6.250, units='USD/kg', desc='Finished cost rate of truss components')

        
        # Outputs
        self.add_output('pontoon_radii_ratio', val=0.0, desc='Ratio of inner to outer pontoon tube radius for constraint purposes')
        self.add_output('pontoon_cost', val=0.0, units='USD', desc='Cost of pontoon elements and connecting truss')
        self.add_output('pontoon_mass', val=0.0, units='kg', desc='Mass of pontoon elements and connecting truss')
        self.add_output('pontoon_buoyancy', val=0.0, units='N', desc='Buoyancy force of submerged pontoon elements')
        self.add_output('pontoon_center_of_buoyancy', val=0.0, units='m', desc='z-position of center of pontoon buoyancy force')
        self.add_output('pontoon_center_of_gravity', val=0.0, units='m', desc='z-position of center of pontoon mass')

        self.add_output('axial_stress_factor', val=0.0, desc='Ratio of axial stress to yield stress for all pontoon elements')
        self.add_output('shear_stress_factor', val=0.0, desc='Ratio of shear stress to yield stress for all pontoon elements')
        self.add_output('plot_matrix', val=np.array([]), desc='Ratio of shear stress to yield stress for all pontoon elements', pass_by_obj=True)
        
         
    def solve_nonlinear(self, params, unknowns, resids):
        # Unpack variables
        crossAttachFlag = params['cross_attachment_pontoons']
        lowerAttachFlag = params['lower_attachment_pontoons']
        upperAttachFlag = params['upper_attachment_pontoons']
        lowerRingFlag   = params['lower_ring_pontoons']
        upperRingFlag   = params['upper_ring_pontoons']
        
        R_semi         = params['radius_to_ballast_cylinder']
        R_od_pontoon   = params['outer_pontoon_radius']
        R_id_pontoon   = params['inner_pontoon_radius']
        R_od_base      = params['base_outer_radius']
        R_tower        = params['tower_base_radius']
        R_od_ballast   = params['ballast_outer_radius']

        t_wall_base    = params['base_wall_thickness']
        t_wall_ballast = params['ballast_wall_thickness']

        E              = params['E']
        G              = params['G']
        rho            = params['material_density']
        yield_stress   = params['yield_stress']
        
        ncylinder      = params['number_of_ballast_cylinders']
        z_base         = params['base_z_nodes']
        z_ballast      = params['ballast_z_nodes']
        m_base         = params['base_cylinder_mass']
        m_ballast      = params['ballast_cylinder_mass']
        
        m_tower        = params['tower_mass']
        F_turbine      = params['turbine_surge_force']
        z_Fturbine     = params['turbine_force_points']
        m_rna          = params['rna_mass']
        rna_cg_x       = params['rna_center_of_gravity_x']
        
        rhoWater       = params['water_density']
        V_base         = params['base_cylinder_displaced_volume']
        V_ballast      = params['ballast_cylinder_displaced_volume']

        coeff          = params['pontoon_cost_rate']

        # Derived geometry variables
        nsect          = z_base.size - 1
        freeboard      = z_base[-1] + 1e-2
        cg_turbine     = z_Fturbine[0]
        h_tower        = z_Fturbine[1]
        R_od_base      = nodal2sectional( R_od_base )
        R_od_ballast   = nodal2sectional( R_od_ballast )
        t_wall_base    = nodal2sectional( t_wall_base )
        t_wall_ballast = nodal2sectional( t_wall_ballast )

        # ---NODES---
        # Senu TODO: Should tower and rna have nodes at their CGs?
        # Senu TODO: Mooring tension on column nodes?

        # Add nodes for base column
        baseLowerID = 0 + 1
        znode = np.copy( z_base )
        xnode = np.zeros(znode.shape)
        ynode = np.zeros(znode.shape)
        baseUpperID = xnode.size

        # Get x and y positions of surrounding ballast columns
        ballastx = R_semi * np.cos( np.linspace(0, 2*np.pi, ncylinder+1) )
        ballasty = R_semi * np.sin( np.linspace(0, 2*np.pi, ncylinder+1) )
        ballastx = ballastx[:-1]
        ballasty = ballasty[:-1]

        # Add in ballast column nodes around the circle
        ballastLowerID = []
        ballastUpperID = []
        for k in xrange(ncylinder):
            ballastLowerID.append( xnode.size + 1 )
            xnode = np.append(xnode, ballastx[k]*np.ones(z_ballast.shape) )
            ynode = np.append(ynode, ballasty[k]*np.ones(z_ballast.shape) )
            znode = np.append(znode, z_ballast )
            ballastUpperID.append( xnode.size )

        # Add tower base, tower CG, tower top nodes (skipping RNA CG for now)
        turbineID = xnode.size + 1
        xnode = np.r_[xnode, 0.0, 0.0, 0.0, rna_cg_x]
        ynode = np.r_[ynode, 0.0, 0.0, 0.0, 0.0]
        znode = np.r_[znode, freeboard, freeboard+cg_turbine, freeboard+h_tower, freeboard+h_tower]

        # Create Node ID object
        nnode = 1 + np.arange(xnode.size)
        rnode = np.zeros(xnode.shape)
        nodes = frame3dd.NodeData(nnode, xnode, ynode, znode, rnode)

        
        # ---REACTIONS---
        # Pin the bottom windward column node.  Other bottom column nodes are on "rollers" (z-fixed).  Otherwise free
        # Free=0, Rigid=1
        # Senu TODO: Reconsider this with mooring lines, etc.
        # Senu and Rick TODO: Review BCs
        Rx  = np.zeros(xnode.shape)
        Ry  = np.zeros(xnode.shape)
        Rz  = np.zeros(xnode.shape)
        Rxx = np.zeros(xnode.shape)
        Ryy = np.zeros(xnode.shape)
        Rzz = np.zeros(xnode.shape)
        # Pinned windward column lower node (first ballastLowerID)
        nid = ballastLowerID[0]
        Rx[nid] = Ry[nid] = Rz[nid] = Rxx[nid] = Ryy[nid] = Rzz[nid] = 1
        # Rollers for other lower column nodes, restrict motion
        nid = ballastLowerID[1:]
        Rz[nid] = Rxx[nid] = Ryy[nid] = Rzz[nid] = 1

        # Get reactions object from frame3dd
        reactions = frame3dd.ReactionData(nnode, Rx, Ry, Rz, Rxx, Ryy, Rzz, rigid=1)


        # ---ELEMENTS / EDGES---
        N1 = np.array([], dtype=np.int32)
        N2 = np.array([], dtype=np.int32)
        # Lower connection from central base column to ballast columns
        if lowerAttachFlag:
            lowerAttachEID = N1.size + 1
            for k in xrange(ncylinder):
                N1 = np.append(N1, baseLowerID )
                N2 = np.append(N2, ballastLowerID[k] )
        # Upper connection from central base column to ballast columns
        if upperAttachFlag:
            upperAttachEID = N1.size + 1
            for k in xrange(ncylinder):
                N1 = np.append(N1, baseUpperID )
                N2 = np.append(N2, ballastUpperID[k] )
        # Cross braces from lower central base column to upper ballast columns
        if crossAttachFlag:
            crossAttachEID = N1.size + 1
            for k in xrange(ncylinder):
                N1 = np.append(N1, baseLowerID )
                N2 = np.append(N2, ballastUpperID[k] )
        # Lower ring around ballast columns
        if lowerRingFlag:
            lowerRingEID = N1.size + 1
            for k in xrange(ncylinder-1):
                N1 = np.append(N1, ballastLowerID[k] )
                N2 = np.append(N2, ballastLowerID[k+1] )
            N1 = np.append(N1, ballastLowerID[0] )
            N2 = np.append(N2, ballastLowerID[-1] )
        # Upper ring around ballast columns
        if upperRingFlag:
            upperRingEID = N1.size + 1
            for k in xrange(ncylinder-1):
                N1 = np.append(N1, ballastUpperID[k] )
                N2 = np.append(N2, ballastUpperID[k+1] )
            N1 = np.append(N1, ballastUpperID[0] )
            N2 = np.append(N2, ballastUpperID[-1] )
        # TODO: Parameterize these for upper, lower, cross connections
        # Properties for the inner connectors
        Ax, As, I, Jx, S, C = TubeProperties(R_od_pontoon, R_id_pontoon)
        Ax    = Ax  * np.ones(N1.shape)
        As    = As  * np.ones(N1.shape)
        Jx    = Jx  * np.ones(N1.shape)
        I     = I   * np.ones(N1.shape)
        S     = S   * np.ones(N1.shape)
        C     = C   * np.ones(N1.shape)
        modE  = E   * np.ones(N1.shape)
        modG  = G   * np.ones(N1.shape)
        roll  = 0.0 * np.ones(N1.shape)
        dens  = rho * np.ones(N1.shape)

        # Now mock up cylindrical columns as truss members even though long, slender assumption breaks down
        # Senu TODO: Make artificially more stiff?
        baseEID = N1.size + 1
        for ii in xrange(nsect):
            N1 = np.append(N1, baseLowerID + ii)
            N2 = np.append(N2, baseLowerID + ii + 1)

            c_Ax, c_As, c_I, c_Jx, c_S, c_C = TubeProperties(R_od_base[ii], R_od_base[ii]-t_wall_base[ii])
            Ax   = np.append(Ax  , c_Ax )
            As   = np.append(As  , c_As )
            Jx   = np.append(Jx  , c_Jx )
            I    = np.append(I   , c_I )
            S    = np.append(S   , c_S )
            C    = np.append(C   , c_C )
            modE = np.append(modE, E )
            modG = np.append(modG, G )
            roll = np.append(roll, 0.0 )
            dens = np.append(dens, 1e-16 ) #rho

        ballastEID = []
        for k in xrange(ncylinder):
            ballastEID.append( N1.size + 1 )
            
            for ii in xrange(nsect):
                N1 = np.append(N1, ballastLowerID[k] + ii)
                N2 = np.append(N2, ballastLowerID[k] + ii + 1)

                c_Ax, c_As, c_I, c_Jx, c_S, c_C = TubeProperties(R_od_ballast[ii], R_od_ballast[ii]-t_wall_ballast[ii])
                Ax   = np.append(Ax  , c_Ax )
                As   = np.append(As  , c_As )
                Jx   = np.append(Jx  , c_Jx )
                I    = np.append(I   , c_I )
                S    = np.append(S   , c_S )
                C    = np.append(C   , c_C )
                modE = np.append(modE, E )
                modG = np.append(modG, G )
                roll = np.append(roll, 0.0 )
                dens = np.append(dens, 1e-16 ) #rho

        # Add in tower contributions: truss to freeboard
        N1 = np.append(N1, baseUpperID )
        N2 = np.append(N2, turbineID )
        # Add in tower contributions: freeboard to towerCG
        N1 = np.append(N1, turbineID )
        N2 = np.append(N2, turbineID + 1)
        # Add in tower contributions: towerCG to tower top
        N1 = np.append(N1, turbineID + 1)
        N2 = np.append(N2, turbineID + 2)
        # Add in tower contributions: tower top to rna
        N1 = np.append(N1, turbineID + 2)
        N2 = np.append(N2, turbineID + 3)
        # Special tower (rigid) properties
        # Senu TODO: Currently assuming a 1m thick tower wall
        # Senu TODO: RNA should be even stiffer?
        c_Ax, c_As, c_I, c_Jx, c_S, c_C = TubeProperties(R_tower, R_tower-1.0)
        Ax   = np.append(Ax  , c_Ax * np.ones((4,)))
        As   = np.append(As  , c_As * np.ones((4,)) )
        Jx   = np.append(Jx  , c_Jx * np.ones((4,)) )
        I    = np.append(I   , c_I  * np.ones((4,)) )
        S    = np.append(S   , c_S  * np.ones((4,)) )
        C    = np.append(C   , c_C  * np.ones((4,)) )
        modE = np.append(modE, E * np.ones((4,)) )
        modG = np.append(modG, G * np.ones((4,)) )
        roll = np.append(roll, 0.0 * np.ones((4,)) )
        dens = np.append(dens, 1e-16 * np.ones((4,)) ) #rho


        # ---Get element object from frame3dd---
        nelem    = 1 + np.arange(N1.size)
        elements = frame3dd.ElementData(nelem, N1, N2, Ax, As, As, Jx, I, I, modE, modG, roll, dens)

        
        # ---Options object---
        shear = 1               # 1: include shear deformation
        geom = 1                # 1: include geometric stiffness
        dx = 0.1              # x-axis increment for internal forces
        other = frame3dd.Options(shear, geom, dx)

        # Initialize frame3dd object
        self.frame = frame3dd.Frame(nodes, reactions, elements, other)

        # Store data for plotting, also handy for operations below
        plotMat = np.zeros((nelem.size, 3, 2))
        plotMat[:,:,0] = np.c_[xnode[N1-1], ynode[N1-1], znode[N1-1]]
        plotMat[:,:,1] = np.c_[xnode[N2-1], ynode[N2-1], znode[N2-1]]
        
        # Compute length and center of gravity for each element for use below
        elemL   = np.sqrt( np.sum( np.diff(plotMat, axis=2)**2.0, axis=1) ).flatten()
        elemCoG = 0.5*np.sum(plotMat, axis=2)

        # ---LOAD CASES---
        # Extreme loading
        gx = 0.0
        gy = 0.0
        gz = -gravity
        load = frame3dd.StaticLoadCase(gx, gy, gz)

        # Get mass right- ballasts, stiffeners, tower, rna, etc.
        # Get point loads- mooring?
        # NOTE: Loading is in local element coordinates 0-L
        # Buoyancy- main column: uniform loads per section (approximation)
        nrange  = np.arange(nsect, dtype=np.int32)
        EL      = baseEID + nrange
        Uz      = V_base * rhoWater * gravity / np.diff(z_base)
        m_extra = m_base
        # Buoyancy- ballast columns
        for k in xrange(ncylinder):
            EL      = np.append(EL, ballastEID[k] + nrange)
            Uz      = np.append(Uz,  V_ballast * rhoWater * gravity / np.diff(z_ballast) )
            m_extra = np.append(m_extra, m_ballast)
            
        # Add mass of base and ballast cylinders while we've already done the element enumeration
        self.frame.changeExtraElementMass(EL, m_extra, False)
        
        # Buoyancy for fully submerged members
        # Note indices to elemL and elemCoG could include -1, but since there is assumed to be more than 1 cylinder, this is not necessary
        nrange  = np.arange(ncylinder, dtype=np.int32)
        Frange  = np.pi * R_od_pontoon**2 * rhoWater * gravity
        F_truss = 0.0
        z_cb    = np.zeros((3,))
        if znode[ballastLowerID[0]-1] < 0.0:
            if lowerAttachFlag:
                EL       = np.append(EL, lowerAttachEID + nrange)
                Uz       = np.append(Uz, Frange * np.ones(nrange.shape))
                F_truss += Frange * elemL[lowerAttachEID-1] * ncylinder
                z_cb    += Frange * elemL[lowerAttachEID-1] * ncylinder * elemCoG[lowerAttachEID-1,:]
            if lowerRingFlag:
                EL       = np.append(EL, lowerRingEID + nrange)
                Uz       = np.append(Uz, Frange * np.ones(nrange.shape))
                F_truss += Frange * elemL[lowerRingEID-1] * ncylinder
                z_cb    += Frange * elemL[lowerRingEID-1] * ncylinder * elemCoG[lowerRingEID-1]
            if crossAttachFlag:
                factor   = np.minimum(1.0, (0.0 - znode[baseLowerID-1]) / (znode[ballastUpperID[0]-1] - znode[baseLowerID-1]) )
                EL       = np.append(EL, crossAttachEID + nrange)
                Uz       = np.append(Uz, factor * Frange * np.ones(nrange.shape))
                F_truss += factor * Frange * elemL[crossAttachEID-1] * ncylinder
                z_cb    += factor * Frange * elemL[crossAttachEID-1] * ncylinder * elemCoG[crossAttachEID-1,:]
        if znode[ballastUpperID[0]-1] < 0.0:
            if upperAttachFlag:
                EL       = np.append(EL, upperAttachEID + nrange)
                Uz       = np.append(Uz, Frange * np.ones(nrange.shape))
                F_truss += Frange * elemL[upperAttachEID-1] * ncylinder
                z_cb    += Frange * elemL[upperAttachEID-1] * ncylinder * elemCoG[upperAttachEID-1,:]
            if upperRingFlag:
                EL       = np.append(EL, upperRingEID + nrange)
                Uz       = np.append(Uz, Frange * np.ones(nrange.shape))
                F_truss += Frange * elemL[upperRingEID-1] * ncylinder
                z_cb    += Frange * elemL[upperRingEID-1] * ncylinder * elemCoG[upperRingEID-1,:]

        # Finalize uniform loads
        Ux = Uy = np.zeros(Uz.shape)
        load.changeUniformLoads(EL, Ux, Uy, Uz)

        # Point loading for rotor thrust and wind loads at CG
        # Note: extra momemt from mass accounted for below
        nF  = turbineID + np.array([1,3], dtype=np.int32)
        Fx  = F_turbine
        Fy = Fz = Mxx = Myy = Mzz = np.zeros(Fx.shape)
        load.changePointLoads(nF, Fx, Fy, Fz, Mxx, Myy, Mzz)

        # Add in extra mass of RNA and tower
        inode   = np.array([turbineID+1, turbineID+3], dtype=np.int32) # rna
        m_extra = np.array([m_tower, m_rna])
        Ixx = Iyy = Izz = Ixy = Ixz = Iyz = rhox = rhoy = rhoz = np.zeros(m_extra.shape)
        self.frame.changeExtraNodeMass(inode, m_extra, Ixx, Iyy, Izz, Ixy, Ixz, Iyz, rhox, rhoy, rhoz, False)
        
        # Senu TODO: Hydrodynamic loading

        # Store load case into frame 3dd object
        self.frame.addLoadCase(load)


        # ---DYNAMIC ANALYSIS---
        #nM = 5              # number of desired dynamic modes of vibration
        #Mmethod = 2         # 1: subspace Jacobi     2: Stodola
        #lump = 1            # 0: consistent mass ... 1: lumped mass matrix
        #tol = 1e-7          # mode shape tolerance
        #shift = 0.0         # shift value ... for unrestrained structures
        
        #self.frame.enableDynamics(nM, Mmethod, lump, tol, shift)


        # ---RUN ANALYSIS---
        try:
            displacements, forces, reactions, internalForces, mass, modal = self.frame.run()
        except:
            errData = collections.namedtuple('errorData', ['Nx','Tx', 'My', 'Mz', 'Vy', 'Vz'])
            errVal  = 1e30 * np.ones((1,2))
            internalForces = []
            for k in xrange(nelem.size):
                internalForces.append( errData(errVal, errVal, errVal, errVal, errVal, errVal) )

        
        # --OUTPUTS--
        unknowns['plot_matrix'] = plotMat
        
        # Buoyancy assembly from incremental calculations above
        unknowns['pontoon_buoyancy'] = F_truss
        unknowns['pontoon_center_of_buoyancy'] = z_cb[-1] / F_truss

        # Sum up mass and compute CofG.  Frame3DD does mass, but not CG
        # TODO: Subtract out extra pontoon length that overlaps with column radii
        ind = baseEID-1
        m_total = Ax[:ind] * rho * elemL[:ind]
        unknowns['pontoon_mass'] = m_total.sum() #mass.struct_mass
        unknowns['pontoon_cost'] = coeff * m_total.sum()
        unknowns['pontoon_center_of_gravity'] = np.sum( m_total * elemCoG[:ind,-1] ) / m_total.sum()
        
        # Compute axial and shear stresses in elements given Frame3DD outputs and some geomtry data
        # Method comes from Section 7.14 of Frame3DD documentation
        # http://svn.code.sourceforge.net/p/frame3dd/code/trunk/doc/Frame3DD-manual.html#structuralmodeling
        nE    = nelem.size
        iCase = 0
        ind = [0, -1]
        sgns = np.array([1.0, -1.0])
        sigma_ax = np.zeros((nE,))
        sigma_sh = np.zeros((nE,))
        for iE in xrange(nE):
            Nx = internalForces[iE].Nx[iCase, ind]
            Tx = internalForces[iE].Tx[iCase, ind]
            My = internalForces[iE].My[iCase, ind]
            Mz = internalForces[iE].Mz[iCase, ind]
            Vy = internalForces[iE].Vy[iCase, ind]
            Vz = internalForces[iE].Vz[iCase, ind]
            
            sigma_ax[iE] = np.max(sgns*Nx/Ax[iE] + np.abs(My)/S[iE] + np.abs(Mz)/S[iE])
            sigma_sh[iE] = np.max( np.r_[ np.abs(Vy)/As[iE] + np.abs(Tx)/C[iE],
                                          np.abs(Vz)/As[iE] + np.abs(Tx)/C[iE] ] )

        # Express stress as ratio relative to yield_stress
        unknowns['axial_stress_factor'] = sigma_ax.max() / yield_stress
        unknowns['shear_stress_factor'] = sigma_sh.max() / yield_stress
        unknowns['pontoon_radii_ratio'] = R_id_pontoon / R_od_pontoon
