import collections
from openmdao.api import Component
import numpy as np
import pyframe3dd.frame3dd as frame3dd
from floatingInstance import nodal2sectional

from commonse import gravity, eps, Tube


def find_nearest(array,value):
    return (np.abs(array-value)).argmin() 


class SemiPontoon(Component):
    """
    OpenMDAO Component class for semisubmersible pontoon / truss structure for floating offshore wind turbines.
    Should be tightly coupled with Semi and Mooring classes for full system representation.
    """

    def __init__(self, nSection):
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
        self.add_param('base_z_nodes', val=np.zeros((nSection+1,)), units='m', desc='z-coordinates of section nodes (length = nsection+1)')
        self.add_param('base_outer_diameter', val=np.zeros((nSection+1,)), units='m', desc='outer radius at each section node bottom to top (length = nsection + 1)')
        self.add_param('base_wall_thickness', val=np.zeros((nSection+1,)), units='m', desc='shell wall thickness at each section node bottom to top (length = nsection + 1)')
        self.add_param('base_cylinder_mass', val=np.zeros((nSection,)), units='kg', desc='mass of base cylinder by section')
        self.add_param('base_cylinder_displaced_volume', val=np.zeros((nSection,)), units='m**3', desc='cylinder volume of water displaced by section')
        self.add_param('base_pontoon_attach_upper', val=0.0, units='m', desc='z-value of upper truss attachment on base column')
        self.add_param('base_pontoon_attach_lower', val=0.0, units='m', desc='z-value of lower truss attachment on base column')

        # Ballast cylinders
        self.add_param('ballast_z_nodes', val=np.zeros((nSection+1,)), units='m', desc='z-coordinates of section nodes (length = nsection+1)')
        self.add_param('ballast_outer_diameter', val=np.zeros((nSection+1,)), units='m', desc='outer radius at each section node bottom to top (length = nsection + 1)')
        self.add_param('ballast_wall_thickness', val=np.zeros((nSection+1,)), units='m', desc='shell wall thickness at each section node bottom to top (length = nsection + 1)')
        self.add_param('ballast_cylinder_mass', val=np.zeros((nSection,)), units='kg', desc='mass of ballast cylinder by section')
        self.add_param('ballast_cylinder_displaced_volume', val=np.zeros((nSection,)), units='m**3', desc='cylinder volume of water displaced by section')
        self.add_param('fairlead', val=1.0, units='m', desc='Depth below water for mooring line attachment')

        # Semi geometry
        self.add_param('radius_to_ballast_cylinder', val=10.0, units='m',desc='Distance from base cylinder centerpoint to ballast cylinder centerpoint')
        self.add_param('number_of_ballast_cylinders', val=3, desc='Number of ballast cylinders evenly spaced around base cylinder', pass_by_obj=True)

        # Pontoon properties
        self.add_param('pontoon_outer_diameter', val=0.5, units='m',desc='Outer radius of tubular pontoon that connects ballast or base cylinders')
        self.add_param('pontoon_wall_thickness', val=0.05, units='m',desc='Inner radius of tubular pontoon that connects ballast or base cylinders')
        self.add_param('cross_attachment_pontoons', val=True, desc='Inclusion of pontoons that connect the bottom of the central base to the tops of the outer ballast columns', pass_by_obj=True)
        self.add_param('lower_attachment_pontoons', val=True, desc='Inclusion of pontoons that connect the central base to the outer ballast columns at their bottoms', pass_by_obj=True)
        self.add_param('upper_attachment_pontoons', val=True, desc='Inclusion of pontoons that connect the central base to the outer ballast columns at their tops', pass_by_obj=True)
        self.add_param('lower_ring_pontoons', val=True, desc='Inclusion of pontoons that ring around outer ballast columns at their bottoms', pass_by_obj=True)
        self.add_param('upper_ring_pontoons', val=True, desc='Inclusion of pontoons that ring around outer ballast columns at their tops', pass_by_obj=True)
        self.add_param('outer_cross_pontoons', val=True, desc='Inclusion of pontoons that ring around outer ballast columns at their tops', pass_by_obj=True)
        
        # Turbine parameters
        self.add_param('turbine_mass', val=eps, units='kg', desc='mass of tower')
        self.add_param('turbine_force', val=np.zeros(3), units='N', desc='Force in xyz-direction on turbine')
        self.add_param('turbine_moment', val=np.zeros(3), units='N*m', desc='Moments about turbine base')
        self.add_param('turbine_I_base', val=np.zeros(6), units='kg*m**2', desc='Moments about turbine base')

        # Manufacturing
        self.add_param('connection_ratio_max', val=0.0, desc='Maximum ratio of pontoon outer diameter to base/ballast outer diameter')
        
        # Costing
        self.add_param('pontoon_cost_rate', val=6.250, units='USD/kg', desc='Finished cost rate of truss components')

        
        # Outputs
        self.add_output('pontoon_cost', val=0.0, units='USD', desc='Cost of pontoon elements and connecting truss')
        self.add_output('pontoon_mass', val=0.0, units='kg', desc='Mass of pontoon elements and connecting truss')
        self.add_output('pontoon_buoyancy', val=0.0, units='N', desc='Buoyancy force of submerged pontoon elements')
        self.add_output('pontoon_center_of_buoyancy', val=0.0, units='m', desc='z-position of center of pontoon buoyancy force')
        self.add_output('pontoon_center_of_gravity', val=0.0, units='m', desc='z-position of center of pontoon mass')

        self.add_output('axial_stress_factor', val=0.0, desc='Ratio of axial stress to yield stress for all pontoon elements')
        self.add_output('shear_stress_factor', val=0.0, desc='Ratio of shear stress to yield stress for all pontoon elements')
        self.add_output('plot_matrix', val=np.array([]), desc='Ratio of shear stress to yield stress for all pontoon elements', pass_by_obj=True)
        self.add_output('base_connection_ratio', val=np.zeros((nSection+1,)), desc='Ratio of pontoon outer diameter to base outer diameter')
        self.add_output('ballast_connection_ratio', val=np.zeros((nSection+1,)), desc='Ratio of pontoon outer diameter to base outer diameter')
        self.add_output('pontoon_base_attach_upper', val=0.0, desc='Fractional distance along base column for upper truss attachment')
        self.add_output('pontoon_base_attach_lower', val=0.0, desc='Fractional distance along base column for lower truss attachment')
        
        
        # Derivatives
        self.deriv_options['type'] = 'fd'
        self.deriv_options['form'] = 'central'
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
        
        R_semi         = params['radius_to_ballast_cylinder']
        R_od_pontoon   = 0.5*params['pontoon_outer_diameter']
        R_od_base      = 0.5*params['base_outer_diameter']
        R_od_ballast   = 0.5*params['ballast_outer_diameter']

        t_wall_base    = params['base_wall_thickness']
        t_wall_ballast = params['ballast_wall_thickness']
        t_wall_pontoon = params['pontoon_wall_thickness']

        E              = params['E']
        G              = params['G']
        rho            = params['material_density']
        yield_stress   = params['yield_stress']
        
        ncylinder      = params['number_of_ballast_cylinders']
        z_base         = params['base_z_nodes']
        z_ballast      = params['ballast_z_nodes']
        z_attach_upper = params['base_pontoon_attach_upper']
        z_attach_lower = params['base_pontoon_attach_lower']
        z_fairlead     = -params['fairlead']
        
        m_base         = params['base_cylinder_mass']
        m_ballast      = params['ballast_cylinder_mass']
        
        m_turbine      = params['turbine_mass']
        F_turbine      = params['turbine_force']
        M_turbine      = params['turbine_moment']
        I_turbine      = params['turbine_I_base']
        
        rhoWater       = params['water_density']
        V_base         = params['base_cylinder_displaced_volume']
        V_ballast      = params['ballast_cylinder_displaced_volume']

        coeff          = params['pontoon_cost_rate']

        # Quick ratio for unknowns
        unknowns['base_connection_ratio']    = params['connection_ratio_max'] - R_od_pontoon/R_od_base
        unknowns['ballast_connection_ratio'] = params['connection_ratio_max'] - R_od_pontoon/R_od_ballast
        unknowns['pontoon_base_attach_upper'] = (z_attach_upper - z_base[0]) / (z_base[-1] - z_base[0]) #0.5<x<1.0
        unknowns['pontoon_base_attach_lower'] = (z_attach_lower - z_base[0]) / (z_base[-1] - z_base[0]) #0.0<x<0.5
        
        # ---NODES---
        # Senu TODO: Should tower and rna have nodes at their CGs?
        # Senu TODO: Mooring tension on column nodes?

        # Add nodes for base column: Using 4 nodes/3 elements per section
        # Make sure there is a node at upper and lower attachment points
        z_base_full = np.linspace(z_base[0], z_base[-1], 4*z_base.size)
        baseBeginID = 0 + 1

        idx = find_nearest(z_base_full, z_attach_lower)
        z_base_full[idx] = z_attach_lower
        baseLowerID = idx + 1

        idx = find_nearest(z_base_full, z_attach_upper)
        z_base_full[idx] = z_attach_upper
        baseUpperID = idx + 1
        baseEndID = z_base_full.size

        znode = np.copy( z_base_full )
        xnode = np.zeros(znode.shape)
        ynode = np.zeros(znode.shape)

        # Get x and y positions of surrounding ballast columns
        ballastx = R_semi * np.cos( np.linspace(0, 2*np.pi, ncylinder+1) )
        ballasty = R_semi * np.sin( np.linspace(0, 2*np.pi, ncylinder+1) )
        ballastx = ballastx[:-1]
        ballasty = ballasty[:-1]

        # Add in ballast column nodes around the circle, make sure there is a node at the fairlead
        ballastLowerID = []
        ballastUpperID = []
        fairleadID     = []
        z_ballast_full = np.linspace(z_ballast[0], z_ballast[-1], 4*z_ballast.size)
        idx = find_nearest(z_ballast_full, z_fairlead)
        myones = np.ones(z_ballast_full.shape)
        for k in xrange(ncylinder):
            ballastLowerID.append( xnode.size + 1 )
            fairleadID.append( xnode.size + idx + 1 )
            xnode = np.append(xnode, ballastx[k]*myones)
            ynode = np.append(ynode, ballasty[k]*myones)
            znode = np.append(znode, z_ballast_full )
            ballastUpperID.append( xnode.size )

        # Add nodes midway around outer ring for cross bracing
        if outerCrossFlag:
            crossx = 0.5*(ballastx + np.roll(ballastx,1))
            crossy = 0.5*(ballasty + np.roll(ballasty,1))

            crossOuterLowerID = xnode.size + np.arange(ncylinder) + 1
            xnode = np.append(xnode, crossx)
            ynode = np.append(ynode, crossy)
            znode = np.append(znode, z_ballast_full[0]*np.ones(ncylinder))

            #crossOuterUpperID = xnode.size + np.arange(ncylinder) + 1
            #xnode = np.append(xnode, crossx)
            #ynode = np.append(ynode, crossy)
            #znode = np.append(znode, z_ballast_full[-1]*np.ones(ncylinder))
                
        # Create Node Data object
        nnode = 1 + np.arange(xnode.size)
        rnode = np.zeros(xnode.shape)
        nodes = frame3dd.NodeData(nnode, xnode, ynode, znode, rnode)

        
        # ---REACTIONS---
        # Pin (3DOF) the nodes at the mooring connections.  Otherwise free
        # Free=0, Rigid=1
        Rx  = np.zeros(xnode.shape)
        Ry  = np.zeros(xnode.shape)
        Rz  = np.zeros(xnode.shape)
        Rxx = np.zeros(xnode.shape)
        Ryy = np.zeros(xnode.shape)
        Rzz = np.zeros(xnode.shape)
        nid = fairleadID
        Rx[nid] = Ry[nid] = Rz[nid] = 1
        # First approach
        # Pinned windward column lower node (first ballastLowerID)
        #nid = ballastLowerID[0]
        #Rx[nid] = Ry[nid] = Rz[nid] = Rxx[nid] = Ryy[nid] = Rzz[nid] = 1
        # Rollers for other lower column nodes, restrict motion
        #nid = ballastLowerID[1:]
        #Rz[nid] = Rxx[nid] = Ryy[nid] = Rzz[nid] = 1

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
        # Outer cross braces
        if outerCrossFlag:
            outerCrossEID = N1.size + 1
            for k in xrange(ncylinder-1):
                N1 = np.append(N1, ballastUpperID[k] )
                N2 = np.append(N2, crossOuterLowerID[k] )
                N1 = np.append(N1, ballastUpperID[k] )
                N2 = np.append(N2, crossOuterLowerID[k+1] )
            N1 = np.append(N1, ballastUpperID[-1] )
            N2 = np.append(N2, crossOuterLowerID[-1] )
            N1 = np.append(N1, ballastUpperID[-1] )
            N2 = np.append(N2, crossOuterLowerID[0] )
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
        R_od_base      = nodal2sectional( np.interp(z_base_full, z_base, R_od_base) )
        t_wall_base    = nodal2sectional( np.interp(z_base_full, z_base, t_wall_base) )
        R_od_ballast   = nodal2sectional( np.interp(z_ballast_full, z_ballast, R_od_ballast) )
        t_wall_ballast = nodal2sectional( np.interp(z_ballast_full, z_ballast, t_wall_ballast) )
        # Senu TODO: Make artificially more stiff?
        baseEID = N1.size + 1
        mytube  = Tube(2.0*R_od_base, t_wall_base)
        myrange = np.arange(R_od_base.size)
        myones  = np.ones(myrange.shape)
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
        dens = np.append(dens, eps*myones ) #rho

        ballastEID = []
        mytube     = Tube(2.0*R_od_ballast, t_wall_ballast)
        myrange    = np.arange(R_od_ballast.size)
        myones     = np.ones(myrange.shape)
        for k in xrange(ncylinder):
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
            dens = np.append(dens, eps*myones ) #rho


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
        nrange  = np.arange(R_od_base.size, dtype=np.int32)
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
                factor   = np.minimum(1.0, (0.0 - z_attach_lower) / (znode[ballastUpperID[0]-1] - z_attach_lower) )
                EL       = np.append(EL, crossAttachEID + nrange)
                Uz       = np.append(Uz, factor * Frange * np.ones(nrange.shape))
                F_truss += factor * Frange * elemL[crossAttachEID-1] * ncylinder
                z_cb    += factor * Frange * elemL[crossAttachEID-1] * ncylinder * elemCoG[crossAttachEID-1,:]
            if outerCrossFlag:
                factor   = np.minimum(1.0, (0.0 - znode[baseLowerID-1]) / (znode[ballastUpperID[0]-1] - znode[baseLowerID-1]) )
                EL       = np.append(EL, outerCrossEID + np.arange(2*ncylinder, dtype=np.int32) )
                Uz       = np.append(Uz, factor * Frange * np.ones(nrange.shape))
                F_truss += factor * Frange * elemL[outerCrossEID-1] * ncylinder
                z_cb    += factor * Frange * elemL[outerCrossEID-1] * ncylinder * elemCoG[outerCrossEID-1,:]
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
        nF  = np.array([ baseEndID ], dtype=np.int32)
        Fx  = np.array([ F_turbine[0] ])
        Fy  = np.array([ F_turbine[1] ])
        Fz  = np.array([ F_turbine[2] ])
        Mxx = np.array([ M_turbine[0] ])
        Myy = np.array([ M_turbine[1] ])
        Mzz = np.array([ M_turbine[2] ])
        load.changePointLoads(nF, Fx, Fy, Fz, Mxx, Myy, Mzz)

        # Add in extra mass of turbine
        inode   = np.array([baseEndID], dtype=np.int32) # rna
        m_extra = np.array([m_turbine])
        Ixx = np.array([ I_turbine[0] ])
        Iyy = np.array([ I_turbine[1] ])
        Izz = np.array([ I_turbine[2] ])
        Ixy = np.array([ I_turbine[3] ])
        Ixz = np.array([ I_turbine[4] ])
        Iyz = np.array([ I_turbine[5] ])
        rhox = rhoy = rhoz = np.zeros(m_extra.shape)
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
