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
        self.params['scope_ratio'] = 902.2 / (self.params['water_depth']-self.params['fairlead']) 
        self.params['anchor_radius'] = 853.87
        self.params['mooring_diameter'] = 0.09
        
        # Change scalars to vectors where needed
        self.check_vectors()
        
    def get_assembly(self): return FloatingSE(NSECTIONS)
    
    def get_design_variables(self):
        # Make a neat list of design variables, lower bound, upper bound, scalar
        desvarList = [('fairlead',0.0, 100.0, 1.0),
                      #('fairlead_offset_from_shell',0.0, 5.0, 1e2),
                      ('base_freeboard',0.0, 50.0, 1.0),
                      ('base_section_height',1e-1, 100.0, 1e1),
                      ('base_outer_diameter',2.1, 40.0, 10.0),
                      ('base_wall_thickness',1e-2, 5e-1, 1e3),
                      ('tower_section_height',1e-1, 100.0, 1e1),
                      ('tower_outer_diameter',1.1, 40.0, 10.0),
                      ('tower_wall_thickness',1e-2, 1.0, 1e3),
                      ('scope_ratio', 1.0, 5.0, 1.0),
                      ('anchor_radius', 1.0, 1e3, 1e-2),
                      ('mooring_diameter', 0.05, 1.0, 1e1),
                      ('base_permanent_ballast_height', 1e-1, 50.0, 1.0),
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
        return desvarList


    def get_constraints(self):

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
            #['stab.buoyancy_to_gravity', 0.1, None, None],
            
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

        self.draw_column(fig, [0.0, 0.0], self.params['hub_height'], self.params['tower_section_height'],
                         0.5*self.params['tower_outer_diameter'], None, (0.9,)*3)

        self.set_figure(fig, fname)


        


        
def example_spar():
    myspar = SparInstance()
    myspar.evaluate('psqp')
    myspar.visualize('spar-initial.jpg')
    return myspar

def optimize_spar(algo='slsqp', myspar=None):
    if myspar is None: myspar = SparInstance()
    myspar.run(algo)
    myspar.visualize('spar-'+algo+'.jpg')
    return myspar

def psqp_optimal():
    #OrderedDict([('total_cost', array([0.85428322]))])
    myspar = SparInstance()
    myspar.params['fairlead'] = 5.2535195304756925
    myspar.params['base_freeboard'] = 11.266593127013982
    myspar.params['base_section_height'] = np.array( [35.08425750951959, 34.62912259289317, 34.81694953013919, 7.441391626075208, 12.990742629624677] )
    myspar.params['base_outer_diameter'] = np.array( [2.6481670472541663, 3.6644590361949887, 5.914499349789744, 8.304762256602293, 5.616116536284985, 5.508506516934016] )
    myspar.params['base_wall_thickness'] = np.array( [0.006841508279275946, 0.006399425828760462, 0.00677860852296999, 0.006626268366436105, 0.010170203701277653, 0.00550031726871565] )
    myspar.params['tower_section_height'] = np.array( [15.227647356049365, 15.280735953716578, 15.339859351198202, 15.139266648587856, 15.345897563434004] )
    myspar.params['tower_outer_diameter'] = np.array( [6.456257663818232, 5.7536143807842315, 5.681532012577617, 5.02961484018852, 4.136090669182665, 3.9832683794360344] )
    myspar.params['tower_wall_thickness'] = np.array( [0.038070207592394226, 0.015187798073131448, 0.009007699450009954, 0.005591853003006994, 0.005417724373126626, 0.004999999999999961] )
    myspar.params['scope_ratio'] = 2.9298431862588616
    myspar.params['anchor_radius'] = 854.1923042892677
    myspar.params['mooring_diameter'] = 0.1669564274261904
    myspar.params['base_permanent_ballast_height'] = 6.973411581222806
    myspar.params['base_stiffener_web_height'] = np.array( [0.07839776923061138, 0.07161528545467119, 0.09016746097165369, 0.07637969725126702, 0.06533829588658703] )
    myspar.params['base_stiffener_web_thickness'] = np.array( [0.0032561039035359307, 0.0030189646701950733, 0.0037317703705913572, 0.003817035608813318, 0.006693166060995202] )
    myspar.params['base_stiffener_flange_width'] = np.array( [0.016069147298235786, 0.0230660558593931, 0.009999999999999992, 0.012310842677169986, 0.010260004814402379] )
    myspar.params['base_stiffener_flange_thickness'] = np.array( [0.01330260231271094, 0.015309976035834184, 0.032704804943342804, 0.012343218694699802, 0.0426071781731897] )
    myspar.params['base_stiffener_spacing'] = np.array( [0.15442408395236168, 0.1680377737869477, 0.2413573475462672, 0.23671486059235827, 0.4481524581179832] )

    myspar.evaluate('slsqp')
    #myspar.visualize('spar-psqp.jpg')
    return myspar
        
if __name__ == '__main__':
    #optimize_spar('psqp')
    myspar = psqp_optimal()
    optimize_spar('psqp', myspar)
    #example_spar()
