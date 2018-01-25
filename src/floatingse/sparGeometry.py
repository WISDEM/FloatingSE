from openmdao.api import Component
import numpy as np
import commonse.Frustum as frustum

class SparGeometry(Component):
    """
    OpenMDAO Component class for Spar substructure for floating offshore wind turbines.
    Should be tightly coupled with MAP Mooring class for full system representation.
    """

    def __init__(self, nSection):
        super(SparGeometry,self).__init__()

        # Design variables
        self.add_param('water_depth', val=0.0, units='m', desc='water depth')
        self.add_param('freeboard', val=25.0, units='m', desc='Length of spar above water line')
        self.add_param('fairlead', val=1.0, units='m', desc='Depth below water for mooring line attachment')
        self.add_param('section_height', val=np.zeros((nSection,)), units='m', desc='length (height) or each section in the spar bottom to top (length = nsection)')
        self.add_param('outer_radius', val=np.zeros((nSection+1,)), units='m', desc='outer radius at each section node bottom to top (length = nsection + 1)')
        self.add_param('wall_thickness', val=np.zeros((nSection+1,)), units='m', desc='shell wall thickness at each section node bottom to top (length = nsection + 1)')
        self.add_param('fairlead_offset_from_shell', val=0.5, units='m',desc='fairlead offset from shell')
        self.add_param('tower_base_radius', val=3.25, units='m', desc='outer radius of tower at base')

        # Outputs
        self.add_output('draft', val=0.0, units='m', desc='Spar draft (length of body under water)')
        self.add_output('z_nodes', val=np.zeros((nSection+1,)), units='m', desc='z-coordinates of section nodes (length = nsection+1)')
        self.add_output('z_section', val=np.zeros((nSection,)), units='m', desc='z-coordinates of section centers of mass (length = nsection)')
        self.add_output('fairlead_radius', val=0.0, units='m', desc='Outer spar radius at fairlead depth (point of mooring attachment)')

        # Output constraints
        self.add_output('draft_depth_ratio', val=0.0, desc='Ratio of draft to water depth')
        self.add_output('fairlead_draft_ratio', val=0.0, desc='Ratio of fairlead to draft')
        self.add_output('transition_radius', val=0.0, units='m', desc='Buffer between spar top and tower base')


    def solve_nonlinear(self, params, unknowns, resids):
        """Sets nodal points and sectional centers of mass in z-coordinate system with z=0 at the waterline.
        Nodal points are the beginning and end points of each section.
        Nodes and sections start at bottom and move upwards.
        
        INPUTS:
        ----------
        params   : dictionary of input parameters
        unknowns : dictionary of output parameters
        
        OUTPUTS  : none (all unknown dictionary values set)
        """
        # Unpack variables
        D_water   = params['water_depth']
        R_od      = params['outer_radius']
        R_tower   = params['tower_base_radius']
        t_wall    = params['wall_thickness']
        h_section = params['section_height']
        freeboard = params['freeboard'] # length of spar under water
        fairlead  = params['fairlead'] # depth of mooring attachment point
        fair_off  = params['fairlead_offset_from_shell']

        # With waterline at z=0, set the z-position of section nodes
        # Note sections and nodes start at bottom of spar and move up
        z_nodes             = np.flipud( freeboard - np.r_[0.0, np.cumsum(np.flipud(h_section))] )
        unknowns['draft']   = np.abs(z_nodes[0])
        unknowns['z_nodes'] = z_nodes

        # Determine radius at mooring connection point (fairlead)
        unknowns['fairlead_radius'] = fair_off + np.interp(-fairlead, z_nodes, R_od)
        
        # With waterline at z=0, set the z-position of section centroids
        R          = R_od - 0.5*t_wall
        cm_section = frustum.frustumShellCG_radius(R[:-1], R[1:], h_section)
        unknowns['z_section'] = z_nodes[:-1] + cm_section

        # Create constraint output that draft is less than water depth and fairlead is less than draft
        unknowns['draft_depth_ratio'] = unknowns['draft'] / D_water
        unknowns['fairlead_draft_ratio'] = 0.0 if z_nodes[0] == 0.0 else fairlead / unknowns['draft'] 

        # Constrain spar top to be at least greater than tower base
        unknowns['transition_radius'] = R_od[-1] - R_tower

