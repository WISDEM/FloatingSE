from floating_instance import FloatingInstance
import numpy as np
        
class SemiInstance(FloatingInstance):
    def __init__(self):
        super(SemiInstance, self).__init__()

        # Change scalars to vectors where needed
        self.check_vectors()

        
    def visualize(self, fname=None):
        fig = self.init_figure()

        self.draw_ocean(fig)

        mooringMat = self.prob['mm.plot_matrix']
        self.draw_mooring(fig, mooringMat)

        pontoonMat = self.prob['load.plot_matrix']
        zcut = 1.0 + np.maximum( self.params['base_freeboard'], self.params['auxiliary_freeboard'] )
        self.draw_pontoons(fig, pontoonMat, 0.5*self.params['pontoon_outer_diameter'], zcut)

        self.draw_column(fig, [0.0, 0.0], self.params['base_freeboard'], self.params['base_section_height'],
                           0.5*self.params['base_outer_diameter'], self.params['base_stiffener_spacing'])

        self.draw_ballast(fig, [0.0, 0.0], self.params['base_freeboard'], self.params['base_section_height'],
                          0.5*self.params['base_outer_diameter']-self.params['base_wall_thickness'],
                          self.params['base_permanent_ballast_height'], self.prob['subs.variable_ballast_height'])

        R_semi  = self.params['radius_to_auxiliary_column']
        ncolumn = int(self.params['number_of_auxiliary_columns'])
        angles = np.linspace(0, 2*np.pi, ncolumn+1)
        x = R_semi * np.cos( angles )
        y = R_semi * np.sin( angles )
        for k in xrange(ncolumn):
            self.draw_column(fig, [x[k], y[k]], self.params['auxiliary_freeboard'], self.params['auxiliary_section_height'],
                               0.5*self.params['auxiliary_outer_diameter'], self.params['auxiliary_stiffener_spacing'])

            self.draw_ballast(fig, [x[k], y[k]], self.params['auxiliary_freeboard'], self.params['auxiliary_section_height'],
                              0.5*self.params['auxiliary_outer_diameter']-self.params['auxiliary_wall_thickness'],
                              self.params['auxiliary_permanent_ballast_height'], 0.0)

            self.draw_heave_plate(fig, [x[k], y[k]], self.params['auxiliary_freeboard'], self.params['auxiliary_section_height'],
                                  0.5*self.params['auxiliary_heave_plate_diameter'])
            
            
        self.draw_column(fig, [0.0, 0.0], self.params['hub_height'], self.params['tower_section_height'],
                         0.5*self.params['tower_outer_diameter'], None, (0.9,)*3)

            
        self.set_figure(fig, fname)


