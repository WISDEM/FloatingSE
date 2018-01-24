from openmdao.api import Component
import numpy as np
import commonse.Frustum as frustum
from floatingInstance import NSECTIONS

class SemiGeometry(Component):
    """
    OpenMDAO Component class for Semi substructure for floating offshore wind turbines.
    Should be tightly coupled with MAP Mooring class for full system representation.
    """

    def __init__(self):
        super(SemiGeometry,self).__init__()

        # Design variables
        self.add_param('base_freeboard', val=25.0, units='m', desc='Length of spar above water line')
        self.add_param('base_section_height', val=np.zeros((NSECTIONS,)), units='m', desc='length (height) or each section in the spar bottom to top (length = nsection)')
        self.add_param('base_outer_radius', val=np.zeros((NSECTIONS+1,)), units='m', desc='outer radius at each section node bottom to top (length = nsection + 1)')
        self.add_param('base_wall_thickness', val=np.zeros((NSECTIONS+1,)), units='m', desc='shell wall thickness at each section node bottom to top (length = nsection + 1)')

        self.add_param('ballast_freeboard', val=25.0, units='m', desc='Length of spar above water line')
        self.add_param('ballast_section_height', val=np.zeros((NSECTIONS,)), units='m', desc='length (height) or each section in the spar bottom to top (length = nsection)')
        self.add_param('ballast_outer_radius', val=np.zeros((NSECTIONS+1,)), units='m', desc='outer radius at each section node bottom to top (length = nsection + 1)')
        self.add_param('ballast_wall_thickness', val=np.zeros((NSECTIONS+1,)), units='m', desc='shell wall thickness at each section node bottom to top (length = nsection + 1)')

        self.add_param('water_depth', val=0.0, units='m', desc='water depth')
        self.add_param('fairlead', val=1.0, units='m', desc='Depth below water for mooring line attachment')
        self.add_param('fairlead_offset_from_shell', val=0.5, units='m',desc='fairlead offset from shell')
        self.add_param('radius_to_ballast_cylinder', val=10.0, units='m',desc='Distance from base cylinder centerpoint to ballast cylinder centerpoint')
        self.add_param('tower_base_radius', val=3.25, units='m', desc='outer radius of tower at base')

        # Outputs
        self.add_output('base_draft', val=0.0, units='m', desc='Spar draft (length of body under water)')
        self.add_output('ballast_draft', val=0.0, units='m', desc='Spar draft (length of body under water)')
        self.add_output('base_z_nodes', val=np.zeros((NSECTIONS+1,)), units='m', desc='z-coordinates of section nodes (length = nsection+1)')
        self.add_output('ballast_z_nodes', val=np.zeros((NSECTIONS+1,)), units='m', desc='z-coordinates of section nodes (length = nsection+1)')
        self.add_output('base_z_section', val=np.zeros((NSECTIONS,)), units='m', desc='z-coordinates of section centers of mass (length = nsection)')
        self.add_output('ballast_z_section', val=np.zeros((NSECTIONS,)), units='m', desc='z-coordinates of section centers of mass (length = nsection)')
        self.add_output('fairlead_radius', val=0.0, units='m', desc='Outer spar radius at fairlead depth (point of mooring attachment)')

        # Output constraints
        self.add_output('base_draft_depth_ratio', val=0.0, desc='Ratio of draft to water depth')
        self.add_output('ballast_draft_depth_ratio', val=0.0, desc='Ratio of draft to water depth')
        self.add_output('fairlead_draft_ratio', val=0.0, desc='Ratio of fairlead to draft')
        self.add_output('base_taper_ratio', val=np.zeros((NSECTIONS,)), desc='Ratio of outer radius change in a section to its starting value')
        self.add_output('ballast_taper_ratio', val=np.zeros((NSECTIONS,)), desc='Ratio of outer radius change in a section to its starting value')
        self.add_output('base_ballast_spacing', val=0.0, desc='Radius of base and ballast cylinders relative to spacing')
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
        R_od_base      = params['base_outer_radius']
        R_tower        = params['tower_base_radius']
        t_wall_base    = params['base_wall_thickness']
        h_section_base = params['base_section_height']
        freeboard_base = params['base_freeboard'] # length of spar under water

        R_od_ballast      = params['ballast_outer_radius']
        t_wall_ballast    = params['ballast_wall_thickness']
        h_section_ballast = params['ballast_section_height']
        freeboard_ballast = params['ballast_freeboard'] # length of spar under water

        fairlead       = params['fairlead'] # depth of mooring attachment point
        fair_off       = params['fairlead_offset_from_shell']
        D_water        = params['water_depth']
        R_semi         = params['radius_to_ballast_cylinder']

        # Set spacing constraint
        unknowns['base_ballast_spacing'] = (R_od_base.max() + R_od_ballast.max()) / R_semi
        
        def cyl_geom(freeboard, h_section, R_od, t_wall):
            z_nodes = np.flipud( freeboard - np.r_[0.0, np.cumsum(np.flipud(h_section))] )
            R       = R_od - 0.5*t_wall
            cm_sec  = frustum.frustumShellCG_radius(R[:-1], R[1:], h_section)
            z_sec   = z_nodes[:-1] + cm_sec
            return z_nodes, z_sec
        
        z_nodes_base   , z_section_base    = cyl_geom(freeboard_base   , h_section_base   , R_od_base   , t_wall_base)
        z_nodes_ballast, z_section_ballast = cyl_geom(freeboard_ballast, h_section_ballast, R_od_ballast, t_wall_ballast)

        # With waterline at z=0, set the z-position of section nodes
        # Note sections and nodes start at bottom of spar and move up
        unknowns['base_draft']     = np.abs(z_nodes_base[0])
        unknowns['base_z_nodes']   = z_nodes_base
        unknowns['base_z_section'] = z_section_base

        unknowns['ballast_draft']     = np.abs(z_nodes_ballast[0])
        unknowns['ballast_z_nodes']   = z_nodes_ballast
        unknowns['ballast_z_section'] = z_section_ballast
        
        # Create constraint output for manufacturability that limits the changes in outer radius from one node to the next
        unknowns['base_taper_ratio']    = np.abs( np.diff(R_od_base   ) / R_od_base[:-1]    )
        unknowns['ballast_taper_ratio'] = np.abs( np.diff(R_od_ballast) / R_od_ballast[:-1] )
        
        # Create constraint output that draft is less than water depth and fairlead is less than draft
        unknowns['base_draft_depth_ratio']    = unknowns['base_draft'] / D_water
        unknowns['ballast_draft_depth_ratio'] = unknowns['ballast_draft'] / D_water
        unknowns['fairlead_draft_ratio']      = fairlead / unknowns['ballast_draft'] 

        # Determine radius at mooring connection point (fairlead)
        unknowns['fairlead_radius'] = R_semi + fair_off + np.interp(-fairlead, z_nodes_ballast, R_od_ballast)

        # Constrain base column top to be at least greater than tower base
        unknowns['transition_radius'] = R_od_base[-1] - R_tower



