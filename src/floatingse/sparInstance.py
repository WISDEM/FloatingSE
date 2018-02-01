from floatingInstance import FloatingInstance, NSECTIONS, NPTS, vecOption
from sparAssembly import SparAssembly
import numpy as np
import time
    
class SparInstance(FloatingInstance):
    def __init__(self):
        super(SparInstance, self).__init__()

        # Parameters beyond those in superclass
        # Typically static- set defaults
        self.params['permanent_ballast_density'] = 4492.0
        self.params['bulkhead_mass_factor'] = 1.0
        self.params['ring_mass_factor'] = 1.0
        self.params['shell_mass_factor'] = 1.0
        self.params['spar_mass_factor'] = 1.05
        self.params['outfitting_mass_fraction'] = 0.06
        self.params['ballast_cost_rate'] = 100.0
        self.params['tapered_col_cost_rate'] = 4720.0
        self.params['outfitting_cost_rate'] = 6980.0
        self.params['morison_mass_coefficient'] = 1.969954
        
        # Typically design (OC3)
        self.params['freeboard'] = 10.0
        self.params['fairlead'] = 70.0
        self.set_length(130.0)
        self.params['spar_section_height'] = np.array([36.0, 36.0, 36.0, 8.0, 14.0])
        self.params['spar_outer_diameter'] = 2*np.array([4.7, 4.7, 4.7, 4.7, 3.25, 3.25])
        self.params['spar_wall_thickness'] = 0.05
        self.params['fairlead_offset_from_shell'] = 5.2-4.7
        self.params['permanent_ballast_height'] = 10.0
        self.params['stiffener_web_height'] = 0.1
        self.params['stiffener_web_thickness'] = 0.04
        self.params['stiffener_flange_width'] = 0.1
        self.params['stiffener_flange_thickness'] = 0.02
        self.params['stiffener_spacing'] = 0.4

        # OC3
        self.params['water_depth'] = 320.0
        self.params['wave_height'] = 10.8
        self.params['wave_period'] = 9.8
        self.params['wind_reference_speed'] = 11.0
        self.params['wind_reference_height'] = 119.0
        self.params['alpha'] = 0.11
        self.params['morison_mass_coefficient'] = 2.0

        self.params['number_of_mooring_lines'] = 3
        self.params['scope_ratio'] = 902.2 / (self.params['water_depth']-self.params['fairlead']) 
        self.params['anchor_radius'] = 853.87
        self.params['mooring_diameter'] = 0.09

        self.params['tower_metric'] = 6.5

        # Change scalars to vectors where needed
        self.check_vectors()

    def set_length(self, inval):
        self.params['spar_section_height'] = vecOption(inval/NSECTIONS, NSECTIONS)

    def check_vectors(self):
        self.params['tower_metric']               = vecOption(self.params['tower_metric'], NSECTIONS+1)
        self.params['spar_outer_diameter']        = vecOption(self.params['spar_outer_diameter'], NSECTIONS+1)
        self.params['spar_wall_thickness']        = vecOption(self.params['spar_wall_thickness'], NSECTIONS+1)
        self.params['stiffener_web_height']       = vecOption(self.params['stiffener_web_height'], NSECTIONS)
        self.params['stiffener_web_thickness']    = vecOption(self.params['stiffener_web_thickness'], NSECTIONS)
        self.params['stiffener_flange_width']     = vecOption(self.params['stiffener_flange_width'], NSECTIONS)
        self.params['stiffener_flange_thickness'] = vecOption(self.params['stiffener_flange_thickness'], NSECTIONS)
        self.params['stiffener_spacing']          = vecOption(self.params['stiffener_spacing'], NSECTIONS)
        self.params['bulkhead_nodes']             = [False] * (NSECTIONS+1)
        self.params['bulkhead_nodes'][0]          = True
        self.params['bulkhead_nodes'][1]          = True

    def get_assembly(self): return SparAssembly(NSECTIONS, NPTS)
    
    def get_design_variables(self):
        # Make a neat list of design variables, lower bound, upper bound, scalar
        desvarList = [('freeboard',0.0, 50.0, 1.0),
                      ('fairlead',0.0, 100.0, 1.0),
                      ('fairlead_offset_from_shell',0.0, 5.0, 1e2),
                      ('spar_section_height',1e-1, 100.0, 1e1),
                      ('spar_outer_diameter',1.1, 40.0, 10.0),
                      ('spar_wall_thickness',5e-3, 1.0, 1e3),
                      ('scope_ratio', 1.0, 5.0, 1.0),
                      ('anchor_radius', 1.0, 1e3, 1e-2),
                      ('mooring_diameter', 0.05, 1.0, 1e1),
                      ('stiffener_web_height', 1e-2, 1.0, 1e2),
                      ('stiffener_web_thickness', 1e-3, 5e-1, 1e2),
                      ('stiffener_flange_width', 1e-2, 5.0, 1e2),
                      ('stiffener_flange_thickness', 1e-3, 5e-1, 1e2),
                      ('stiffener_spacing', 1e-1, 1e2, 1e1),
                      ('permanent_ballast_height', 1e-1, 50.0, 1.0)]
        # TODO: Integer and Boolean design variables
        #prob.driver.add_desvar('number_of_mooring_lines', lower=1)
        #prob.driver.add_desvar('mooring_type')
        #prob.driver.add_desvar('anchor_type')
        #prob.driver.add_desvar('bulkhead_nodes')
        return desvarList

    def add_constraints_objective(self):
        # CONSTRAINTS
        # These are mostly the outputs that were not connected to another model

        # Ensure that draft is greater than 0 (spar length>0) and that less than water depth
        # Ensure that fairlead attaches to draft
        self.prob.driver.add_constraint('sg.draft_depth_ratio',lower=0.0, upper=0.75)
        self.prob.driver.add_constraint('sg.fairlead_draft_ratio',lower=0.0, upper=1.0)

        # Ensure that the radius doesn't change dramatically over a section
        self.prob.driver.add_constraint('gc.manufacturability',upper=0.0)
        self.prob.driver.add_constraint('gc.weldability',upper=0.0)

        # Ensure that the spar top matches the tower base
        self.prob.driver.add_constraint('tt.transition_buffer',lower=0.0, upper=5.0)

        # Ensure max mooring line tension is less than X% of MBL: 60% for intact mooring, 80% for damanged
        self.prob.driver.add_constraint('mm.safety_factor',lower=0.0, upper=0.8)

        # Ensure there is sufficient mooring line length, MAP doesn't give an error about this
        self.prob.driver.add_constraint('mm.mooring_length_min',lower=1.0)
        self.prob.driver.add_constraint('mm.mooring_length_max',upper=1.0)

        # API Bulletin 2U constraints
        self.prob.driver.add_constraint('cyl.flange_spacing_ratio', upper=0.5)
        self.prob.driver.add_constraint('cyl.web_radius_ratio', upper=0.5)
        self.prob.driver.add_constraint('cyl.flange_compactness', lower=1.0)
        self.prob.driver.add_constraint('cyl.web_compactness', lower=1.0)
        self.prob.driver.add_constraint('cyl.axial_local_unity', upper=1.0)
        self.prob.driver.add_constraint('cyl.axial_general_unity', upper=1.0)
        self.prob.driver.add_constraint('cyl.external_local_unity', upper=1.0)
        self.prob.driver.add_constraint('cyl.external_general_unity', upper=1.0)

        # Achieving non-zero variable ballast height means the spar can be balanced with margin as conditions change
        self.prob.driver.add_constraint('sp.variable_ballast_height', lower=2.0, upper=100.0)
        self.prob.driver.add_constraint('sp.variable_ballast_mass', lower=0.0)

        # Metacentric height should be positive for static stability
        self.prob.driver.add_constraint('sp.metacentric_height', lower=0.1)

        # Center of buoyancy should be above CG (difference should be positive)
        self.prob.driver.add_constraint('sp.static_stability', lower=0.1)

        # Surge restoring force should be greater than wave-wind forces (ratio < 1)
        self.prob.driver.add_constraint('sp.offset_force_ratio',lower=0.0, upper=1.0)

        # Heel angle should be less than 6deg for ordinary operation, less than 10 for extreme conditions
        self.prob.driver.add_constraint('sp.heel_angle',lower=0.0, upper=10.0)

        # OBJECTIVE FUNCTION: Minimize total cost!
        self.prob.driver.add_objective('sp.total_cost', scaler=1e-9)

        
    def visualize(self, fname=None):
        fig = self.init_figure()

        self.draw_ocean(fig)

        mooringMat = self.prob['mm.plot_matrix']
        self.draw_mooring(fig, mooringMat)

        self.draw_cylinder(fig, [0.0, 0.0], self.freeboard, self.section_height, 0.5*self.outer_diameter, self.stiffener_spacing)

        self.set_figure(fig, fname)















        
def example_spar():
    myspar = SparInstance()
    myspar.evaluate('psqp')
    #myspar.visualize('spar-initial.jpg')
    #myspar.run('psqp')
    return myspar
    
def psqp_optimal():
    #OrderedDict([('sp.total_cost', array([5.07931743]))])
    myspar = SparInstance()

    myspar.freeboard = 5.10158277e-15
    myspar.fairlead = 71.10786714
    myspar.fairlead_offset_from_shell = 0.89788471
    myspar.section_height = np.array([46.90829885, 47.09591478, 42.98780181, 15.62142311, 27.12647036])
    myspar.outer_diameter = 2*np.array([3.42031956, 3.76211996, 4.13833196, 3.74070578, 3.36663524, 3.25 ])
    myspar.wall_thickness = np.array([0.005 , 0.01798306, 0.01634286, 0.0305431 , 0.01614555, 0.02401723])
    myspar.scope_ratio = 3.5943495
    myspar.anchor_radius = 853.84311789
    myspar.mooring_diameter = 0.07110683
    myspar.stiffener_web_height = np.array([0.12446595, 0.11906005, 0.13367389, 0.09986728, 0.20104476])
    myspar.stiffener_web_thickness = np.array([0.00516946, 0.00494494, 0.00555185, 0.01137231, 0.01044597])
    myspar.stiffener_flange_width = np.array([0.01 , 0.01 , 0.01 , 0.02438453, 0.10485385])
    myspar.stiffener_flange_thickness = np.array([0.27895428, 0.3056761 , 0.49283386, 0.11848842, 0.01322849])
    myspar.stiffener_spacing = np.array([0.21985983, 0.40572512, 1.50992577, 2.76799805, 2.87945321])
    myspar.permanent_ballast_height = 30.18703972

    myspar.evaluate('psqp')
    myspar.visualize('spar-psqp.jpg')
    return myspar
    '''
    '''

def nsga2_optimal():
    #OrderedDict([('sp.total_cost', array([20.63726002]))])
    myspar = SparInstance()
    myspar.freeboard = 5.0
    myspar.fairlead = 7.57
    myspar.fairlead_offset_from_shell = 0.05
    myspar.section_height = np.array([ 18.99987492,  18.9998873 ,  18.99990693,  18.99990914,  18.99990425])
    myspar.outer_diameter = 2*np.array([ 6.99962345,  6.99955813,  6.99973629,  6.99978022,  6.99976883, 6.99988])
    myspar.wall_thickness = np.array([ 0.03712666,  0.02787312,  0.02712097,  0.02206188,  0.02157211, 0.03579269])
    myspar.scope_ratio = 2.40997737
    myspar.anchor_radius = 450.0
    myspar.mooring_diameter = 0.1909802
    myspar.stiffener_web_height= np.array([ 0.10557588,  0.10316776,  0.09795284,  0.09743845,  0.09743956])
    myspar.stiffener_web_thickness = np.array([ 0.03599046,  0.03502903,  0.03323707,  0.03302298,  0.0330262 ])
    myspar.stiffener_flange_width = np.array([ 0.10066915,  0.10029873,  0.09894232,  0.09882406,  0.0988245 ])
    myspar.stiffener_flange_thickness = np.array([ 0.02739561,  0.02327079,  0.01406197,  0.01304515,  0.01304842])
    myspar.stiffener_spacing = np.array([ 0.40020418,  0.40036638,  0.4008825 ,  0.4009331 ,  0.40093272])
    myspar.permanent_ballast_height = 10.0

    myspar.evaluate('nsga2')
    return myspar
    '''
    '''

    
def conmin_optimal():
    #OrderedDict([('sp.total_cost', array([ 8.15839897]))])
    myspar = SparInstance()
    myspar.freeboard = 5.0
    myspar.fairlead = 7.57
    myspar.fairlead_offset_from_shell = 0.05
    myspar.section_height = np.array([ 18.99987492,  18.9998873 ,  18.99990693,  18.99990914,  18.99990425])
    myspar.outer_diameter = 2*np.array([ 6.99962345,  6.99955813,  6.99973629,  6.99978022,  6.99976883, 6.99988])
    myspar.wall_thickness = np.array([ 0.03712666,  0.02787312,  0.02712097,  0.02206188,  0.02157211, 0.03579269])
    myspar.scope_ratio = 2.40997737
    myspar.anchor_radius = 450.0
    myspar.mooring_diameter = 0.1909802
    myspar.stiffener_web_height= np.array([ 0.10557588,  0.10316776,  0.09795284,  0.09743845,  0.09743956])
    myspar.stiffener_web_thickness = np.array([ 0.03599046,  0.03502903,  0.03323707,  0.03302298,  0.0330262 ])
    myspar.stiffener_flange_width = np.array([ 0.10066915,  0.10029873,  0.09894232,  0.09882406,  0.0988245 ])
    myspar.stiffener_flange_thickness = np.array([ 0.02739561,  0.02327079,  0.01406197,  0.01304515,  0.01304842])
    myspar.stiffener_spacing = np.array([ 0.40020418,  0.40036638,  0.4008825 ,  0.4009331 ,  0.40093272])
    myspar.permanent_ballast_height = 10.0

    myspar.evaluate('conmin')
    return myspar
    '''
    '''


def cobyla_optimal():
    #OrderedDict([('sp.total_cost', array([ 6.83851908]))])
    myspar = SparInstance()
    myspar.freeboard = 7.56789854
    myspar.fairlead = 9.41184644
    myspar.fairlead_offset_from_shell = 0.0471558864
    myspar.section_height = np.array([ 18.708914991,  18.784270853,  18.799716693,  18.648435942, 18.711380637])
    myspar.outer_diameter = 2*np.array([ 5.764219519,  5.657993694,  6.159558061,  6.125155506, 6.293851894,  6.606570305])
    myspar.wall_thickness = np.array([ 0.043758918  ,  0.03934623132,  0.04101795034,  0.03947006871, 0.03855182803,  0.04268526778])
    myspar.scope_ratio = 2.39202552
    myspar.anchor_radius = 442.036507
    myspar.mooring_diameter = 0.153629334
    myspar.stiffener_web_height= np.array([ 0.1433863028,  0.1192863504,  0.1102913546,   0.0959098443,   0.0760210847])
    myspar.stiffener_web_thickness = np.array([ 0.0059552804,  0.0049543342,  0.004580744 ,  0.003983435 ,  0.0031573928 ])
    myspar.stiffener_flange_width = np.array([ 0.0924192057,  0.0977347306,  0.0800589589,  0.0797488027,  0.0861943184 ])
    myspar.stiffener_flange_thickness = np.array([ 0.02739561,  0.02327079,  0.01406197,  0.01304515,  0.01304842])
    myspar.stiffener_spacing = np.array([ 0.937472777,   0.913804583,   0.975992681,   0.940785141,  1.077950861])
    myspar.permanent_ballast_height = 2.1531719

    myspar.evaluate('cobyla')
    return myspar
    '''
    '''
        
if __name__ == '__main__':
    example_spar()
    #psqp_optimal()
