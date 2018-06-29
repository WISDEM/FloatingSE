from floating_instance import FloatingInstance, NSECTIONS, NPTS, vecOption
from floatingse.floating import FloatingSE
from commonse import eps
import numpy as np
import time
        
class SparInstance(FloatingInstance):
    def __init__(self):
        super(SparInstance, self).__init__()

        # Parameters beyond those in superclass
        self.params['number_of_auxiliary_columns'] = 0
        self.params['cross_attachment_pontoons_int'] = 0
        self.params['lower_attachment_pontoons_int'] = 0
        self.params['upper_attachment_pontoons_int'] = 0
        self.params['lower_ring_pontoons_int'] = 0
        self.params['upper_ring_pontoons_int'] = 0
        self.params['outer_cross_pontoons_int'] = 0
 
        # Typically design (OC3)
        self.params['base_freeboard'] = 10.0
        self.params['fairlead'] = 5 #70.0
        self.set_length_base(130.0)
        self.params['base_section_height'] = np.array([36.0, 36.0, 36.0, 8.0, 14.0])
        self.params['base_outer_diameter'] = 2*np.array([4.7, 4.7, 4.7, 4.7, 3.25, 3.25])
        self.params['base_wall_thickness'] = 0.05
        self.params['fairlead_offset_from_shell'] = 5.2-4.7
        self.params['base_permanent_ballast_height'] = 10.0
        
        # OC3
        self.params['water_depth'] = 320.0
        self.params['hmax'] = 10.8
        self.params['T'] = 9.8
        self.params['Uref'] = 11.0
        self.params['zref'] = 119.0
        self.params['shearExp'] = 0.11
        self.params['cm'] = 2.0

        self.params['number_of_mooring_lines'] = 3
        self.params['mooring_line_length'] = 902.2
        self.params['anchor_radius'] = 853.87
        self.params['mooring_diameter'] = 0.09
        
        # Change scalars to vectors where needed
        self.check_vectors()
        

    def get_constraints(self):
        conlist = super(SparInstance, self).get_constraints()

        poplist = []
        for k in range(len(conlist)):
            if ( (conlist[k][0].find('aux') >= 0) or
                 (conlist[k][0].find('pontoon') >= 0) or
                 (conlist[k][0].find('base_connection_ratio') >= 0) ):
                poplist.append(k)

        poplist.reverse()
        for k in poplist: conlist.pop(k)

        return conlist
    '''
        conlist = [
            # Try to get tower height to match desired hub height
            ['tow.height_constraint', None, None, 0.0],
            
            # Ensure that draft is greater than 0 (spar length>0) and that less than water depth
            # Ensure that fairlead attaches to draft
            ['base.draft_depth_ratio', 0.0, 0.75, None],
            
            # Ensure that the radius doesn't change dramatically over a section
            ['base.manufacturability', None, 0.0, None],
            ['base.weldability', None, 0.0, None],
            ['tow.manufacturability', None, 0.0, None],
            ['tow.weldability', None, 0.0, None],
            
            # Ensure that the spar top matches the tower base
            ['sg.transition_buffer', -1.0, 1.0, None],
            
            # Ensure max mooring line tension is less than X% of MBL: 60% for intact mooring, 80% for damanged
            ['mm.axial_unity', 0.0, 1.0, None],
            
            # Ensure there is sufficient mooring line length, MAP doesn't give an error about this
            ['mm.mooring_length_max', None, 1.0, None],
            
            # API Bulletin 2U constraints
            ['base.flange_spacing_ratio', None, 1.0, None],
            ['base.stiffener_radius_ratio', None, 0.5, None],
            ['base.flange_compactness', 1.0, None, None],
            ['base.web_compactness', 1.0, None, None],
            ['base.axial_local_api', None, 1.0, None],
            ['base.axial_general_api', None, 1.0, None],
            ['base.external_local_api', None, 1.0, None],
            ['base.external_general_api', None, 1.0, None],
            
            # Pontoon stress safety factor
            ['load.tower_stress', None, 1.0, None],
            ['load.tower_shell_buckling', None, 1.0, None],
            ['load.tower_global_buckling', None, 1.0, None],
            
            # Achieving non-zero variable ballast height means the semi can be balanced with margin as conditions change
            ['subs.variable_ballast_height_ratio', 0.0, 1.0, None],
            ['subs.variable_ballast_mass', 0.0, None, None],
            
            # Metacentric height should be positive for static stability
            ['subs.metacentric_height', 0.1, None, None],
            
            # Center of buoyancy should be above CG (difference should be positive, None],
            #['subs.buoyancy_to_gravity', 0.1, None, None],
            
            # Surge restoring force should be greater than wave-wind forces (ratio < 1, None],
            ['subs.offset_force_ratio', None, 1.0, None],
            
            # Heel angle should be less than 6deg for ordinary operation, less than 10 for extreme conditions
            ['subs.heel_moment_ratio', None, 1.0, None],

            # Wave forcing period should be different than natural periods and structural modes
            ['subs.period_margin_low', None, 1.0, None],
            ['subs.period_margin_high', 1.0, None, None],
            ['subs.modal_margin_low', None, 1.0, None],
            ['subs.modal_margin_high', 1.0, None, None]
        ]
        return conlist
    '''


        
    def visualize(self, fname=None):
        fig = self.init_figure()

        self.draw_ocean(fig)

        self.draw_mooring(fig, self.prob['mm.plot_matrix'])

        self.draw_column(fig, [0.0, 0.0], self.params['base_freeboard'], self.params['base_section_height'],
                           0.5*self.params['base_outer_diameter'], self.params['base_stiffener_spacing'])

        self.draw_column(fig, [0.0, 0.0], self.params['hub_height'], self.params['tower_section_height'],
                         0.5*self.params['tower_outer_diameter'], None, (0.9,)*3)

        self.set_figure(fig, fname)
