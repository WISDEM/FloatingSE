from floatingInstance import FloatingInstance, NSECTIONS, NPTS, vecOption
from floating import FloatingSE
from commonse import eps
import numpy as np
import time
        
class SparInstance(FloatingInstance):
    def __init__(self):
        super(SparInstance, self).__init__()

        # Parameters beyond those in superclass
        self.params['number_of_ballast_columns'] = 0
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
        self.params['permanent_ballast_height_base'] = 10.0

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
            ['tt.transition_buffer', 0.0, 5.0, None],
            
            # Ensure max mooring line tension is less than X% of MBL: 60% for intact mooring, 80% for damanged
            ['mm.safety_factor', 0.0, 0.8, None],
            
            # Ensure there is sufficient mooring line length, MAP doesn't give an error about this
            ['mm.mooring_length_min', 1.0, None, None],
            ['mm.mooring_length_max', None, 1.0, None],
            
            # API Bulletin 2U constraints
            ['base.flange_spacing_ratio', None, 0.5, None],
            ['base.web_radius_ratio', None, 0.5, None],
            ['base.flange_compactness', 1.0, None, None],
            ['base.web_compactness', 1.0, None, None],
            ['base.axial_local_unity', None, 1.0, None],
            ['base.axial_general_unity', None, 1.0, None],
            ['base.external_local_unity', None, 1.0, None],
            ['base.external_general_unity', None, 1.0, None],
            
            # Pontoon stress safety factor
            ['pon.tower_stress', None, 1.0, None],
            ['pon.tower_shell_buckling', None, 1.0, None],
            ['pon.tower_global_buckling', None, 1.0, None],
            
            # Achieving non-zero variable ballast height means the semi can be balanced with margin as conditions change
            ['sm.variable_ballast_height', 2.0, 100.0, None],
            ['sm.variable_ballast_mass', 0.0, None, None],
            
            # Metacentric height should be positive for static stability
            ['sm.metacentric_height', 0.1, None, None],
            
            # Center of buoyancy should be above CG (difference should be positive, None],
            ['sm.static_stability', 0.1, None, None],
            
            # Surge restoring force should be greater than wave-wind forces (ratio < 1, None],
            ['sm.offset_force_ratio', 0.0, 1.0, None],
            
            # Heel angle should be less than 6deg for ordinary operation, less than 10 for extreme conditions
            ['sm.heel_constraint', 0.0, None, None]]
        return conlist

    def add_objective(self):
        # OBJECTIVE FUNCTION: Minimize total cost!
        self.prob.driver.add_objective('total_cost', scaler=1e-9)


        
    def visualize(self, fname=None):
        fig = self.init_figure()

        self.draw_ocean(fig)

        mooringMat = self.prob['mm.plot_matrix']
        self.draw_mooring(fig, mooringMat)

        self.draw_column(fig, [0.0, 0.0], self.params['base_freeboard'], self.params['base_section_height'],
                           0.5*self.params['base_outer_diameter'], self.params['base_stiffener_spacing'])

        self.set_figure(fig, fname)


        


        
def example_spar():
    myspar = SparInstance()
    myspar.evaluate('psqp')
    #myspar.visualize('spar-initial.jpg')
    #myspar.run('slsqp')
    return myspar

def psqp_optimal():
    #OrderedDict([('sm.total_cost', array([0.65987536]))])
    myspar = SparInstance()

    myspar.fairlead = 22.2366002
    myspar.fairlead_offset_from_shell = 4.99949523
    myspar.radius_to_ballast_column = 26.79698385
    myspar.freeboard_base = 4.97159308
    myspar.section_height_base = np.array([6.72946378, 5.97993104, 5.47072089, 5.71437475, 5.44290777])
    myspar.outer_diameter_base = 2*np.array([2.0179943 , 2.21979373, 2.4417731 , 2.68595041, 2.95454545, 3.25 ])
    myspar.wall_thickness_base = np.array([0.01100738, 0.00722966, 0.00910002, 0.01033024, 0.00639292, 0.00560714])
    myspar.freeboard_ballast = -1.14370386e-20
    myspar.section_height_ballast = np.array([1.44382195, 2.71433629, 6.1047888 , 5.14428218, 6.82937098])
    myspar.outer_diameter_ballast = 2*np.array([2.57228724, 2.82647421, 3.10005118, 3.40594536, 3.74653989, 4.12119389])
    myspar.wall_thickness_ballast = np.array([0.01558312, 0.005 , 0.005 , 0.005 , 0.005 , 0.005 ])
    myspar.pontoon_outer_diameter = 2*0.92428188
    myspar.pontoon_inner_diameter = 2*0.88909984
    myspar.scope_ratio = 4.71531904
    myspar.anchor_radius = 837.58954811
    myspar.mooring_diameter = 0.36574595
    myspar.stiffener_web_height_base = np.array([0.01625364, 0.04807025, 0.07466081, 0.0529478 , 0.03003529])
    myspar.stiffener_web_thickness_base = np.array([0.00263325, 0.00191218, 0.00404707, 0.00495706, 0.00137335])
    myspar.stiffener_flange_width_base = np.array([0.0100722 , 0.06406752, 0.01342377, 0.07119415, 0.01102604])
    myspar.stiffener_flange_thickness_base = np.array([0.06126737, 0.00481305, 0.01584461, 0.00980356, 0.01218029])
    myspar.stiffener_spacing_base = np.array([1.09512893, 0.67001459, 1.60080836, 1.27068546, 0.2687786 ])
    myspar.permanent_ballast_height_base = 5.34047386
    myspar.stiffener_web_height_ballast = np.array([0.04750412, 0.03926778, 0.04484479, 0.04255339, 0.05903525])
    myspar.stiffener_web_thickness_ballast = np.array([0.00197299, 0.00162998, 0.00186254, 0.00176738, 0.00245192])
    myspar.stiffener_flange_width_ballast = np.array([0.01176864, 0.01018018, 0.01062256, 0.01119399, 0.01023957])
    myspar.stiffener_flange_thickness_ballast = np.array([0.00182314, 0.00428608, 0.01616793, 0.0109717 , 0.00814284])
    myspar.stiffener_spacing_ballast = np.array([0.88934305, 0.19623501, 0.29410086, 0.30762027, 0.4208429 ])
    myspar.permanent_ballast_height_ballast = 0.10007504

    myspar.evaluate('psqp')
    myspar.visualize('spar-psqp.jpg')
    return myspar
        
if __name__ == '__main__':
    #psqp_optimal()
    example_spar()
