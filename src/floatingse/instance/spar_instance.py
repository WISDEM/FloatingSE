from floating_instance import FloatingInstance
import numpy as np


class SparInstance(FloatingInstance):
    def __init__(self):
        super(SparInstance, self).__init__()

        # Parameters beyond those in superclass
        self.params['number_of_offset_columns'] = 0
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
        self.params['Hs'] = 10.8
        self.params['T'] = 9.8
        self.params['Uref'] = 11.0
        self.params['zref'] = 119.0
        self.params['shearExp'] = 0.11
        self.params['cm'] = 2.0

        self.params['number_of_mooring_connections']        = 3
        self.params['mooring_lines_per_connection']         = 1
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

    
    def visualize(self, fname=None):
        fig = self.init_figure()

        self.draw_ocean(fig)

        self.draw_mooring(fig, self.prob['mooring_plot_matrix'])

        zcut = 1.0 + self.params['base_freeboard']
        self.draw_pontoons(fig, self.prob['plot_matrix'], 0.5*self.params['fairlead_support_outer_diameter'], zcut)

        self.draw_column(fig, [0.0, 0.0], self.params['base_freeboard'], self.params['base_section_height'],
                           0.5*self.params['base_outer_diameter'], self.params['base_stiffener_spacing'])

        self.draw_ballast(fig, [0.0, 0.0], self.params['base_freeboard'], self.params['base_section_height'],
                          0.5*self.params['base_outer_diameter']-self.params['base_wall_thickness'],
                          self.params['base_permanent_ballast_height'], self.prob['variable_ballast_height'])

        self.draw_column(fig, [0.0, 0.0], self.params['hub_height'], self.params['tower_section_height'],
                         0.5*self.params['tower_outer_diameter'], None, (0.9,)*3)

        if self.prob['base.ballast_heave_box_mass'] > 0.0:
            self.draw_ballast_heave_box(fig, [0.0, 0.0], self.params['base_freeboard'],
                                        self.params['base_section_height'],
                                        self.params['base_ballast_heave_box_location'],
                                        0.5*self.params['base_ballast_heave_box_diameter'],
                                        self.params['base_ballast_heave_box_height'])
        
        self.set_figure(fig, fname)
