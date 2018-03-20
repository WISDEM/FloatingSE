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
                      ('base_outer_diameter',1.1, 40.0, 10.0),
                      ('base_wall_thickness',5e-3, 1.0, 1e3),
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

        self.draw_column(fig, [0.0, 0.0], self.params['base_freeboard']+self.params['hub_height'], self.params['tower_section_height'],
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
    #OrderedDict([('total_cost', array([0.96963594]))])
    myspar = SparInstance()
    myspar.params['fairlead'] = 4.927202945865353
    myspar.params['base_freeboard'] = 11.51079573762036
    myspar.params['base_section_height'] = np.array( [35.088858368278935, 34.80895102971842, 34.974929395775014, 7.410687297063525, 12.97158337346268] )
    myspar.params['base_outer_diameter'] = np.array( [2.8879418373521997, 4.085966179669333, 6.320531864715026, 8.552395763234195, 5.7346457824168215, 5.514680698692207] )
    myspar.params['base_wall_thickness'] = np.array( [0.007342726623873182, 0.006694729984940154, 0.007245337732535003, 0.009975902410243778, 0.008361582858927942, 0.004999999999999999] )
    myspar.params['scope_ratio'] = 2.928625620780157
    myspar.params['anchor_radius'] = 854.1921746988265
    myspar.params['mooring_diameter'] = 0.1754188030581036
    myspar.params['base_permanent_ballast_height'] = 7.2250448680480455
    myspar.params['base_stiffener_web_height'] = np.array( [0.09837247798364428, 0.09371128257530666, 0.08223199234264876, 0.09776803989025877, 0.015962520996266966] )
    myspar.params['base_stiffener_web_thickness'] = np.array( [0.0040857158653538495, 0.0038921218800465913, 0.0034153511494099662, 0.004060611653939701, 0.000999999999999997] )
    myspar.params['base_stiffener_flange_width'] = np.array( [0.010000000000000009, 0.00999999999999999, 0.010000000000000016, 0.013266411098100681, 0.015118496561641243] )
    myspar.params['base_stiffener_flange_thickness'] = np.array( [0.016737403983002403, 0.027319494467725206, 0.02372704545511987, 0.009539323997907077, 0.0010000000000000013] )
    myspar.params['base_stiffener_spacing'] = np.array( [0.17303295858177742, 0.18691099646865977, 0.2938311789567957, 0.3353137217358804, 0.17675563913173956] )

    myspar.evaluate('slsqp')
    #myspar.visualize('spar-psqp.jpg')
    return myspar
        
if __name__ == '__main__':
    #optimize_spar('psqp')
    myspar = psqp_optimal()
    optimize_spar('psqp', myspar)
    #example_spar()
