from floatingInstance import FloatingInstance, NSECTIONS, NPTS, vecOption
from floating import FloatingSE
from commonse import eps
import numpy as np
import time
        
class SparInstance(FloatingInstance):
    def __init__(self):
        super(SparInstance, self).__init__()

        # Parameters beyond those in superclass
        self.params['number_of_auxiliary_columns'] = 0
        self.params['cross_attachment_pontoons'] = False
        self.params['lower_attachment_pontoons'] = False
        self.params['upper_attachment_pontoons'] = False
        self.params['lower_ring_pontoons'] = False
        self.params['upper_ring_pontoons'] = False
        self.params['outer_cross_pontoons'] = False
 
        # Typically design (OC3)
        self.params['base_freeboard'] = 10.0
        self.params['fairlead'] = 70.0
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
        self.params['scope_ratio'] = 902.2 / (self.params['water_depth']-self.params['fairlead']) 
        self.params['anchor_radius'] = 853.87
        self.params['mooring_diameter'] = 0.09
        
        # Change scalars to vectors where needed
        self.check_vectors()
        
    def get_assembly(self): return FloatingSE(NSECTIONS)
    
    def get_design_variables(self):
        # Make a neat list of design variables, lower bound, upper bound, scalar
        desvarList = [('fairlead',0.0, 100.0, 1.0),
                      ('fairlead_offset_from_shell',0.0, 5.0, 1e2),
                      ('base_freeboard',0.0, 50.0, 1.0),
                      ('base_section_height',1e-1, 100.0, 1e1),
                      ('base_outer_diameter',1.1, 40.0, 10.0),
                      ('base_wall_thickness',5e-3, 1.0, 1e3),
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
                      ('base_stiffener_spacing', 1e-1, 1e2, 1e1)]

        # TODO: Integer and Boolean design variables
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
            
            # Pontoon stress safety factor
            ['load.tower_stress', None, 1.0, None],
            ['load.tower_shell_buckling', None, 1.0, None],
            ['load.tower_global_buckling', None, 1.0, None],
            
            # Achieving non-zero variable ballast height means the semi can be balanced with margin as conditions change
            ['stab.variable_ballast_height_ratio', 0.0, 1.0, None],
            ['stab.variable_ballast_mass', 0.0, None, None],
            
            # Metacentric height should be positive for static stability
            ['stab.metacentric_height', 0.1, None, None],
            
            # Center of buoyancy should be above CG (difference should be positive, None],
            ['stab.static_stability', 0.1, None, None],
            
            # Surge restoring force should be greater than wave-wind forces (ratio < 1, None],
            ['stab.offset_force_ratio', None, 1.0, None],
            
            # Heel angle should be less than 6deg for ordinary operation, less than 10 for extreme conditions
            ['stab.heel_moment_ratio', None, 1.0, None]]
        return conlist

    def add_objective(self):
        # OBJECTIVE FUNCTION: Minimize total cost!
        self.prob.driver.add_objective('total_cost', scaler=1e-9)


        
    def visualize(self, fname=None):
        fig = self.init_figure()

        self.draw_ocean(fig)

        self.draw_mooring(fig, self.prob['mm.plot_matrix'])

        self.draw_column(fig, [0.0, 0.0], self.params['base_freeboard'], self.params['base_section_height'],
                           0.5*self.params['base_outer_diameter'], self.params['base_stiffener_spacing'])

        self.draw_column(fig, [0.0, 0.0], self.params['base_freeboard']+self.params['hub_height'], self.params['tower_section_height'],
                         0.5*self.params['tower_outer_diameter'], None, (0.9,)*3)

        self.set_figure(fig, fname)


        


        
def example_spar():
    myspar = SparInstance()
    myspar.evaluate('psqp')
    myspar.visualize('spar-initial.jpg')
    return myspar

def optimize_spar(algo='slsqp'):
    myspar = SparInstance()
    myspar.run(algo)
    myspar.visualize('spar-'+algo+'.jpg')
    return myspar

def slsqp_optimal():
    #OrderedDict([('total_cost', array([1.49600082]))])
    myspar = SparInstance()

    mypsar.params['fairlead'] = 36.995002263
    mypsar.params['fairlead_offset_from_shell'] = 0.381237766438
    mypsar.params['base_freeboard'] = 2.62142203326
    mypsar.params['base_section_height'] = np.array([41.89338476, 47.60209766, 43.45997812,  0.18040233, 22.67891054])
    mypsar.params['base_outer_diameter'] = np.array([ 4.98194059,  1.99138766,  1.90623058, 16.46455641,  6.58582333,  4.5])
    mypsar.params['base_wall_thickness'] = np.array([0.00895063, 0.01387583, 0.01331409, 0.00842058, 0.01090195, 0.01024931])
    mypsar.params['pontoon_outer_diameter'] = 3.0
    mypsar.params['pontoon_wall_thickness'] = 0.0175
    mypsar.params['base_pontoon_attach_lower'] = -20.0
    mypsar.params['base_pontoon_attach_upper'] = 10.0
    mypsar.params['scope_ratio'] = 3.2257803294
    mypsar.params['anchor_radius'] = 853.319366299
    mypsar.params['mooring_diameter'] = 0.214794904303
    mypsar.params['base_stiffener_web_height'] = np.array([0.09628871, 0.06689576, 0.10380069, 0.10227783, 0.08345127])
    mypsar.params['base_stiffener_web_thickness'] = np.array([0.00399962, 0.00277977, 0.00431121, 0.00424795, 0.00417288])
    mypsar.params['base_stiffener_flange_width'] = np.array([0.01000001, 0.01 0.01001052, 0.01, 0.01000039])
    mypsar.params['base_stiffener_flange_thickness'] = np.array([0.07579568, 0.07942766, 0.03561651, 0.07014924, 0.01354831])
    mypsar.params['base_stiffener_spacing'] = np.array([0.16531022, 1.18355589, 0.31825361, 0.26153587, 0.76595662])

    myspar.evaluate('slsqp')
    myspar.visualize('spar-slsqp.jpg')
    return myspar
        
if __name__ == '__main__':
    optimize_spar('psqp')
    #example_spar()
