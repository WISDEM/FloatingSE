from floatingInstance import FloatingInstance, NSECTIONS, NPTS, vecOption
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
    
    def get_design_variables(self):
        # Make a neat list of design variables, lower bound, upper bound, scalar
        desvarList = [('fairlead',0.0, 100.0, 1.0),
                      ('fairlead_offset_from_shell',0.0, 5.0, 1e2),
                      ('radius_to_auxillary_column',0.0, 40.0, 1.0),
                      ('base_freeboard',0.0, 50.0, 1.0),
                      ('base_section_height',1e-1, 100.0, 1e1),
                      ('base_outer_diameter',1.1, 40.0, 10.0),
                      ('base_wall_thickness',5e-3, 1.0, 1e3),
                      ('auxillary_freeboard',0.0, 50.0, 1.0),
                      ('auxillary_section_height',1e-1, 100.0, 1e1),
                      ('auxillary_outer_diameter',1.1, 40.0, 10.0),
                      ('auxillary_wall_thickness',5e-3, 1.0, 1e3),
                      ('pontoon_outer_diameter', 1e-1, 3.0, 10.0),
                      ('pontoon_wall_thickness', 5e-3, 1e-1, 100.0),
                      ('base_pontoon_attach_lower',-1e2, 1e2, 1.0),
                      ('base_pontoon_attach_upper',-1e2, 1e2, 1.0),
                      ('scope_ratio', 1.0, 5.0, 1.0),
                      ('anchor_radius', 1.0, 1e3, 1e-2),
                      ('mooring_diameter', 0.05, 1.0, 1e1),
                      ('base_stiffener_web_height', 1e-2, 1.0, 1e2),
                      ('base_stiffener_web_thickness', 1e-3, 5e-1, 1e2),
                      ('base_stiffener_flange_width', 1e-2, 5.0, 1e2),
                      ('base_stiffener_flange_thickness', 1e-3, 5e-1, 1e2),
                      ('base_stiffener_spacing', 1e-1, 1e2, 1e1),
                      ('base_permanent_ballast_height', 1e-1, 50.0, 1.0),
                      ('auxillary_stiffener_web_height', 1e-2, 1.0, 1e2),
                      ('auxillary_stiffener_web_thickness', 1e-3, 5e-1, 1e2),
                      ('auxillary_stiffener_flange_width', 1e-2, 5.0, 1e2),
                      ('auxillary_stiffener_flange_thickness', 1e-3, 5e-1, 1e2),
                      ('auxillary_stiffener_spacing', 1e-1, 1e2, 1e1),
                      ('auxillary_permanent_ballast_height', 1e-1, 50.0, 1.0)]

        # TODO: Integer and Boolean design variables
        #prob.driver.add_desvar('number_of_auxillary_columns', lower=1)
        #prob.driver.add_desvar('number_of_mooring_lines', lower=1)
        #prob.driver.add_desvar('mooring_type')
        #prob.driver.add_desvar('anchor_type')
        #prob.driver.add_desvar('bulkhead_nodes')
        #prob.driver.add_desvar('outer_cross_pontoons')
        #prob.driver.add_desvar('cross_attachment_pontoons')
        #prob.driver.add_desvar('lower_attachment_pontoons')
        #prob.driver.add_desvar('upper_attachment_pontoons')
        #prob.driver.add_desvar('lower_ring_pontoons')
        #prob.driver.add_desvar('upper_ring_pontoons')
        return desvarList

    def get_constraints(self):

        conlist = [
            # Ensure that draft is greater than 0 (spar length>0) and that less than water depth
            # Ensure that fairlead attaches to draft
            ['base.draft_depth_ratio', 0.0, 0.75, None],
            ['aux.draft_depth_ratio', 0.0, 0.75, None],
            ['aux.fairlead_draft_ratio', 0.0, 1.0, None],
            ['sg.base_auxillary_spacing', 0.0, 1.0, None],
            
            # Ensure that the radius doesn't change dramatically over a section
            ['base.manufacturability', None, 0.0, None],
            ['base.weldability', None, 0.0, None],
            ['aux.manufacturability', None, 0.0, None],
            ['aux.weldability', None, 0.0, None],
            ['tow.manufacturability', None, 0.0, None],
            ['tow.weldability', None, 0.0, None],
            
            # Ensure that the spar top matches the tower base
            ['tt.transition_buffer', -1.0, 1.0, None],
            
            # Ensure max mooring line tension is less than X% of MBL: 60% for intact mooring, 80% for damanged
            ['mm.axial_unity', 0.0, 1.0, None],
            
            # Ensure there is sufficient mooring line length, MAP doesn't give an error about this
            ['mm.mooring_length_min', 1.0, None, None],
            ['mm.mooring_length_max', None, 1.0, None],
            
            # API Bulletin 2U constraints
            ['base.flange_spacing_ratio', None, 1.0, None],
            ['base.web_radius_ratio', None, 1.0, None],
            ['base.flange_compactness', 1.0, None, None],
            ['base.web_compactness', 1.0, None, None],
            ['base.axial_local_unity', None, 1.0, None],
            ['base.axial_general_unity', None, 1.0, None],
            ['base.external_local_unity', None, 1.0, None],
            ['base.external_general_unity', None, 1.0, None],
            
            ['aux.flange_spacing_ratio', None, 1.0, None],
            ['aux.web_radius_ratio', None, 1.0, None],
            ['aux.flange_compactness', 1.0, None, None],
            ['aux.web_compactness', 1.0, None, None],
            ['aux.axial_local_unity', None, 1.0, None],
            ['aux.axial_general_unity', None, 1.0, None],
            ['aux.external_local_unity', None, 1.0, None],
            ['aux.external_general_unity', None, 1.0, None],
            
            # Pontoon tube radii
            ['load.base_connection_ratio', 0.0, None, None],
            ['load.auxillary_connection_ratio', 0.0, None, None],
            ['load.pontoon_base_attach_upper', 0.5, 1.0, None],
            ['load.pontoon_base_attach_lower', 0.0, 0.5, None],
            
            # Pontoon stress safety factor
            ['load.pontoon_stress', None, 1.0, None],
            ['load.tower_stress', None, 1.0, None],
            ['load.tower_shell_buckling', None, 1.0, None],
            ['load.tower_global_buckling', None, 1.0, None],
            
            # Achieving non-zero variable ballast height means the semi can be balanced with margin as conditions change
            ['sm.variable_ballast_height_ratio', 0.0, 1.0, None],
            ['sm.variable_ballast_mass', 0.0, None, None],
            
            # Metacentric height should be positive for static stability
            ['sm.metacentric_height', 0.1, None, None],
            
            # Center of buoyancy should be above CG (difference should be positive, None],
            ['sm.static_stability', 0.1, None, None],
            
            # Surge restoring force should be greater than wave-wind forces (ratio < 1, None],
            ['sm.offset_force_ratio', None, 1.0, None],
            
            # Heel angle should be less than 6deg for ordinary operation, less than 10 for extreme conditions
            ['sm.heel_moment_ratio', None, 1.0, None]]
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
        zcut = 1.0 + np.maximum( self.params['base_freeboard'], self.params['auxillary_freeboard'] )
        self.draw_pontoons(fig, pontoonMat, 0.5*self.params['pontoon_outer_diameter'], zcut)

        self.draw_column(fig, [0.0, 0.0], self.params['base_freeboard'], self.params['base_section_height'],
                           0.5*self.params['base_outer_diameter'], self.params['base_stiffener_spacing'])

        R_semi    = self.params['radius_to_auxillary_column']
        ncolumn = self.params['number_of_auxillary_columns']
        angles = np.linspace(0, 2*np.pi, ncolumn+1)
        x = R_semi * np.cos( angles )
        y = R_semi * np.sin( angles )
        for k in xrange(ncolumn):
            self.draw_column(fig, [x[k], y[k]], self.params['auxillary_freeboard'], self.params['auxillary_section_height'],
                               0.5*self.params['auxillary_outer_diameter'], self.params['auxillary_stiffener_spacing'])
            
        self.set_figure(fig, fname)



        


        
def example_semi():
    mysemi = SemiInstance()
    mysemi.evaluate('psqp')
    #mysemi.visualize('semi-initial.jpg')
    #mysemi.run('slsqp')
    return mysemi

def psqp_optimal():
    #OrderedDict([('sm.total_cost', array([0.65987536]))])
    mysemi = SemiInstance()

    mysemi.evaluate('psqp')
    mysemi.visualize('semi-psqp.jpg')
    return mysemi
    
    
if __name__ == '__main__':
    #psqp_optimal()
    example_semi()
