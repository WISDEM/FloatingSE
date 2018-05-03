from floating_instance import FloatingInstance, NSECTIONS, NPTS, vecOption
from floating import FloatingSE
from commonse import eps
import numpy as np
import time
        
class SemiInstance(FloatingInstance):
    def __init__(self):
        super(SemiInstance, self).__init__()

        # Parameters beyond those in superclass

        # Change scalars to vectors where needed
        self.check_vectors()

    def get_assembly(self): return FloatingSE(NSECTIONS)

    def get_constraints(self):

        conlist = [
            # Try to get tower height to match desired hub height
            ['tow.height_constraint', None, None, 0.0],
            
            # Ensure that draft is greater than 0 (spar length>0) and that less than water depth
            # Ensure that fairlead attaches to draft
            ['base.draft_depth_ratio', 0.0, 0.75, None],
            ['aux.draft_depth_ratio', 0.0, 0.75, None],
            ['aux.fairlead_draft_ratio', 0.0, 1.0, None],
            ['sg.base_auxiliary_spacing', 0.0, 1.0, None],
            
            # Ensure that the radius doesn't change dramatically over a section
            ['base.manufacturability', None, 0.0, None],
            ['base.weldability', None, 0.0, None],
            ['aux.manufacturability', None, 0.0, None],
            ['aux.weldability', None, 0.0, None],
            ['tow.manufacturability', None, 0.0, None],
            ['tow.weldability', None, 0.0, None],
            
            # Ensure that the spar top matches the tower base
            ['sg.transition_buffer', -1.0, 1.0, None],
            
            # Ensure max mooring line tension is less than X% of MBL: 60% for intact mooring, 80% for damanged
            ['mm.axial_unity', 0.0, 1.0, None],
            
            # Ensure there is sufficient mooring line length, MAP doesn't give an error about this
            ['mm.mooring_length_min', 1.0, None, None],
            ['mm.mooring_length_max', None, 1.0, None],
            
            # API Bulletin 2U constraints
            ['base.flange_spacing_ratio', None, 1.0, None],
            ['base.stiffener_radius_ratio', None, 0.5, None],
            ['base.flange_compactness', 1.0, None, None],
            ['base.web_compactness', 1.0, None, None],
            ['base.axial_local_unity', None, 1.0, None],
            ['base.axial_general_unity', None, 1.0, None],
            ['base.external_local_unity', None, 1.0, None],
            ['base.external_general_unity', None, 1.0, None],
            
            ['aux.flange_spacing_ratio', None, 1.0, None],
            ['aux.stiffener_radius_ratio', None, 0.5, None],
            ['aux.flange_compactness', 1.0, None, None],
            ['aux.web_compactness', 1.0, None, None],
            ['aux.axial_local_unity', None, 1.0, None],
            ['aux.axial_general_unity', None, 1.0, None],
            ['aux.external_local_unity', None, 1.0, None],
            ['aux.external_general_unity', None, 1.0, None],
            
            # Pontoon tube radii
            ['load.base_connection_ratio', 0.0, None, None],
            ['load.auxiliary_connection_ratio', 0.0, None, None],
            ['load.pontoon_base_attach_upper', 0.5, 1.0, None],
            ['load.pontoon_base_attach_lower', 0.0, 0.5, None],
            
            # Pontoon stress safety factor
            ['load.pontoon_stress', None, 1.0, None],
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
            ['subs.period_margin', 0.1, None, None],
            ['subs.modal_margin', 0.1, None, None]
        ]
        return conlist


    def add_objective(self):
        # OBJECTIVE FUNCTION: Minimize total cost!
        self.prob.driver.add_objective('total_cost', scaler=1e-9)


        
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

        R_semi  = self.params['radius_to_auxiliary_column']
        ncolumn = int(self.params['number_of_auxiliary_columns'])
        angles = np.linspace(0, 2*np.pi, ncolumn+1)
        x = R_semi * np.cos( angles )
        y = R_semi * np.sin( angles )
        for k in xrange(ncolumn):
            self.draw_column(fig, [x[k], y[k]], self.params['auxiliary_freeboard'], self.params['auxiliary_section_height'],
                               0.5*self.params['auxiliary_outer_diameter'], self.params['auxiliary_stiffener_spacing'])

        self.draw_column(fig, [0.0, 0.0], self.params['base_freeboard']+self.params['hub_height'], self.params['tower_section_height'],
                         0.5*self.params['tower_outer_diameter'], None, (0.9,)*3)

            
        self.set_figure(fig, fname)


