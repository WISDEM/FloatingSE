from floatingInstance import FloatingInstance, NSECTIONS, NPTS, vecOption
from sparAssembly import SparAssembly
import numpy as np
import time
    
class SparInstance(FloatingInstance):
    def __init__(self):
        super(SparInstance, self).__init__()

        # Parameters beyond those in superclass
        # Typically static- set defaults
        self.permanent_ballast_density = 4492.0
        self.bulkhead_mass_factor = 1.0
        self.ring_mass_factor = 1.0
        self.shell_mass_factor = 1.0
        self.spar_mass_factor = 1.05
        self.outfitting_mass_fraction = 0.06
        self.ballast_cost_rate = 100.0
        self.tapered_col_cost_rate = 4720.0
        self.outfitting_cost_rate = 6980.0
        self.morison_mass_coefficient = 1.969954
        
        # Typically design (OC3)
        self.freeboard = 10.0
        self.fairlead = 70.0
        self.set_length(130.0)
        self.section_height = np.array([36.0, 36.0, 36.0, 8.0, 14.0])
        self.outer_radius = np.array([4.7, 4.7, 4.7, 4.7, 3.25, 3.25])
        self.wall_thickness = 0.05
        self.fairlead_offset_from_shell = 5.2-4.7
        self.permanent_ballast_height = 10.0
        self.stiffener_web_height= 0.1
        self.stiffener_web_thickness = 0.04
        self.stiffener_flange_width = 0.1
        self.stiffener_flange_thickness = 0.02
        self.stiffener_spacing = 0.4

        # OC3
        self.water_depth = 320.0
        self.wave_height = 10.8
        self.wave_period = 9.8
        self.wind_reference_speed = 11.0
        self.wind_reference_height = 119.0
        self.alpha = 0.11
        self.morison_mass_coefficient = 2.0

        self.max_offset  = 0.1*self.water_depth # Assumption        
        self.number_of_mooring_lines = 3
        self.scope_ratio = 902.2 / (self.water_depth-self.fairlead) 
        self.anchor_radius = 853.87
        self.mooring_diameter = 0.09

        # Change scalars to vectors where needed
        self.check_vectors()

    def set_length(self, inval):
        self.section_height = vecOption(inval/NSECTIONS, NSECTIONS)

    def check_vectors(self):
        self.outer_radius               = vecOption(self.outer_radius, NSECTIONS+1)
        self.wall_thickness             = vecOption(self.wall_thickness, NSECTIONS+1)
        self.stiffener_web_height       = vecOption(self.stiffener_web_height, NSECTIONS)
        self.stiffener_web_thickness    = vecOption(self.stiffener_web_thickness, NSECTIONS)
        self.stiffener_flange_width     = vecOption(self.stiffener_flange_width, NSECTIONS)
        self.stiffener_flange_thickness = vecOption(self.stiffener_flange_thickness, NSECTIONS)
        self.stiffener_spacing          = vecOption(self.stiffener_spacing, NSECTIONS)
        self.bulkhead_nodes             = [False] * (NSECTIONS+1)
        self.bulkhead_nodes[0]          = True
        self.bulkhead_nodes[1]          = True

    def get_assembly(self): return SparAssembly(NSECTIONS, NPTS)
    
    def get_design_variables(self):
        # Make a neat list of design variables, lower bound, upper bound, scalar
        desvarList = [('freeboard.x',0.0, 50.0, 1.0),
                      ('fairlead.x',0.0, 100.0, 1.0),
                      ('fairlead_offset_from_shell.x',0.0, 5.0, 1e2),
                      ('section_height.x',1e-1, 100.0, 1e1),
                      ('outer_radius.x',1.1, 25.0, 10.0),
                      ('wall_thickness.x',5e-3, 1.0, 1e3),
                      ('scope_ratio.x', 1.0, 5.0, 1.0),
                      ('anchor_radius.x', 1.0, 1e3, 1e-2),
                      ('mooring_diameter.x', 0.05, 1.0, 1e1),
                      ('stiffener_web_height.x', 1e-2, 1.0, 1e2),
                      ('stiffener_web_thickness.x', 1e-3, 5e-1, 1e2),
                      ('stiffener_flange_width.x', 1e-2, 5.0, 1e2),
                      ('stiffener_flange_thickness.x', 1e-3, 5e-1, 1e2),
                      ('stiffener_spacing.x', 1e-1, 1e2, 1e1),
                      ('permanent_ballast_height.x', 1e-1, 50.0, 1.0)]
        # TODO: Integer and Boolean design variables
        #prob.driver.add_desvar('number_of_mooring_lines.x', lower=1)
        #prob.driver.add_desvar('mooring_type.x')
        #prob.driver.add_desvar('anchor_type.x')
        #prob.driver.add_desvar('bulkhead_nodes.x')
        return desvarList

    def add_constraints_objective(self):
        # CONSTRAINTS
        # These are mostly the outputs that were not connected to another model

        # Ensure that draft is greater than 0 (spar length>0) and that less than water depth
        # Ensure that fairlead attaches to draft
        self.prob.driver.add_constraint('sg.draft_depth_ratio',lower=0.0, upper=0.75)
        self.prob.driver.add_constraint('sg.fairlead_draft_ratio',lower=0.0, upper=1.0)

        # Ensure that the radius doesn't change dramatically over a section
        self.prob.driver.add_constraint('sg.taper_ratio',upper=0.1)

        # Ensure that the spar top matches the tower base
        self.prob.driver.add_constraint('sg.transition_radius',lower=0.0, upper=5.0)

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

        self.draw_cylinder(fig, [0.0, 0.0], self.freeboard, self.section_height, self.outer_radius, self.stiffener_spacing)

        self.set_figure(fig, fname)















        
def example_spar():
    myspar = SparInstance()
    #myspar.evaluate('psqp')
    #myspar.visualize('spar-initial.jpg')
    myspar.run('psqp')
    return myspar
    
def psqp_optimal():
    #OrderedDict([('sp.total_cost', array([5.07931743]))])
    myspar = SparInstance()

    myspar.freeboard = 5.10158277e-15
    myspar.fairlead = 71.10786714
    myspar.fairlead_offset_from_shell = 0.89788471
    myspar.section_height = np.array([46.90829885, 47.09591478, 42.98780181, 15.62142311, 27.12647036])
    myspar.outer_radius = np.array([3.42031956, 3.76211996, 4.13833196, 3.74070578, 3.36663524, 3.25 ])
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
OrderedDict([('freeboard.x', array([5.10158277e-15])), ('fairlead.x', array([71.10786714])), ('fairlead_offset_from_shell.x', array([0.89788471])), ('section_height.x', array([46.90829885, 47.09591478, 42.98780181, 15.62142311, 27.12647036])), ('outer_radius.x', array([3.42031956, 3.76211996, 4.13833196, 3.74070578, 3.36663524,
       3.25      ])), ('wall_thickness.x', array([0.005     , 0.01798306, 0.01634286, 0.0305431 , 0.01614555,
       0.02401723])), ('scope_ratio.x', array([3.5943495])), ('anchor_radius.x', array([853.84311789])), ('mooring_diameter.x', array([0.07110683])), ('stiffener_web_height.x', array([0.12446595, 0.11906005, 0.13367389, 0.09986728, 0.20104476])), ('stiffener_web_thickness.x', array([0.00516946, 0.00494494, 0.00555185, 0.01137231, 0.01044597])), ('stiffener_flange_width.x', array([0.01      , 0.01      , 0.01      , 0.02438453, 0.10485385])), ('stiffener_flange_thickness.x', array([0.27895428, 0.3056761 , 0.49283386, 0.11848842, 0.01322849])), ('stiffener_spacing.x', array([0.21985983, 0.40572512, 1.50992577, 2.76799805, 2.87945321])), ('permanent_ballast_height.x', array([30.18703972]))])
    '''

def nsga2_optimal():
    #OrderedDict([('sp.total_cost', array([20.63726002]))])
    myspar = SparInstance()
    myspar.freeboard = 5.0
    myspar.fairlead = 7.57
    myspar.fairlead_offset_from_shell = 0.05
    myspar.section_height = np.array([ 18.99987492,  18.9998873 ,  18.99990693,  18.99990914,  18.99990425])
    myspar.outer_radius = np.array([ 6.99962345,  6.99955813,  6.99973629,  6.99978022,  6.99976883, 6.99988])
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
OrderedDict([('freeboard.x', array([0.00069167])), ('fairlead.x', array([56.66277188])), ('fairlead_offset_from_shell.x', array([1.6697556])), ('section_height.x', array([ 0.10006407,  0.14512127,  0.10581745, 57.69806749, 68.18564309])), ('outer_radius.x', array([7.31308737, 8.04138111, 8.4117183 , 8.81644148, 8.29272943,
       7.49543486])), ('wall_thickness.x', array([0.30333932, 0.21646852, 0.17457961, 0.06684975, 0.00500993,
       0.08599076])), ('scope_ratio.x', array([2.11730873])), ('anchor_radius.x', array([472.81702723])), ('mooring_diameter.x', array([0.30137977])), ('stiffener_web_height.x', array([0.40461526, 0.68960035, 0.5429272 , 0.69121104, 0.51499444])), ('stiffener_web_thickness.x', array([0.2180976 , 0.3769686 , 0.19939452, 0.03318099, 0.03211645])), ('stiffener_flange_width.x', array([1.5147774 , 2.40651869, 3.39142326, 0.01592095, 0.04746711])), ('stiffener_flange_thickness.x', array([0.47712255, 0.26231218, 0.26817643, 0.23783247, 0.16175234])), ('stiffener_spacing.x', array([78.61377022, 27.38214287, 12.19569693,  1.1797688 ,  4.05719905])), ('permanent_ballast_height.x', array([18.26590056]))])
    '''

    
def conmin_optimal():
    #OrderedDict([('sp.total_cost', array([ 8.15839897]))])
    myspar = SparInstance()
    myspar.freeboard = 5.0
    myspar.fairlead = 7.57
    myspar.fairlead_offset_from_shell = 0.05
    myspar.section_height = np.array([ 18.99987492,  18.9998873 ,  18.99990693,  18.99990914,  18.99990425])
    myspar.outer_radius = np.array([ 6.99962345,  6.99955813,  6.99973629,  6.99978022,  6.99976883, 6.99988])
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
OrderedDict([('sg.draft_depth_ratio', array([ 0.41284162])), ('mm.safety_factor', array([ 0.7999935])), ('mm.mooring_length_min', array([ 1.03413212])), ('mm.mooring_length_max', array([ 0.76788083])), ('sp.flange_spacing_ratio', array([ 0.25154447,  0.25051738,  0.24681128,  0.24648516,  0.2464865 ])), ('sp.web_radius_ratio', array([ 0.01508315,  0.01473899,  0.01399375,  0.01392023,  0.01392029])), ('sp.flange_compactness', array([ 4.91418164,  4.18969521,  2.56643842,  2.38370799,  2.38429482])), ('sp.web_compactness', array([ 8.20782661,  8.17503344,  8.16979471,  8.16002162,  8.160726  ])), ('sp.axial_local_unity', array([ 0.51507608,  0.46336477,  0.38573459,  0.25387818,  0.06900377])), ('sp.axial_general_unity', array([ 0.9982523 ,  0.95677559,  0.9846294 ,  0.65050248,  0.19042545])), ('sp.external_local_unity', array([ 0.43199964,  0.39057533,  0.32676712,  0.21654406,  0.05750315])), ('sp.external_general_unity', array([ 1.00397317,  0.96693407,  0.99605626,  0.66093149,  0.1917524 ])), ('sp.metacentric_height', array([ 20.55260275])), ('sp.static_stability', array([ 20.41649121])), ('sp.variable_ballast_height', array([ 28.52903672])), ('sp.variable_ballast_mass', array([ 4464694.51334896])), ('sp.offset_force_ratio', array([ 0.98529446])), ('sp.heel_angle', array([ 2.39612487]))])
    '''


def cobyla_optimal():
    #OrderedDict([('sp.total_cost', array([ 6.83851908]))])
    myspar = SparInstance()
    myspar.freeboard = 7.56789854
    myspar.fairlead = 9.41184644
    myspar.fairlead_offset_from_shell = 0.0471558864
    myspar.section_height = np.array([ 18.708914991,  18.784270853,  18.799716693,  18.648435942, 18.711380637])
    myspar.outer_radius = np.array([ 5.764219519,  5.657993694,  6.159558061,  6.125155506, 6.293851894,  6.606570305])
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
OrderedDict([('sg.draft_depth_ratio', array([ 0.3948845])), ('mm.safety_factor', array([ 0.79998459])), ('mm.mooring_length_min', array([ 1.03296286])), ('mm.mooring_length_max', array([ 0.76687562])), ('sp.flange_spacing_ratio', array([ 0.09858335,  0.10695364,  0.08202824,  0.08476835,  0.07996127])), ('sp.web_radius_ratio', array([ 0.02510657,  0.020188  ,  0.01795587,  0.01544565,  0.01178583])), ('sp.flange_compactness', array([ 4.34435796,  3.92278448,  4.35797943,  2.16402123,  1.        ])), ('sp.web_compactness', array([ 1.,  1.,  1.,  1.,  1.])), ('sp.axial_local_unity', array([ 0.47642979,  0.38788415,  0.28705696,  0.17106678,  0.05175507])), ('sp.axial_general_unity', array([ 0.97397846,  0.97695689,  0.98037179,  0.98931662,  0.5567153 ])), ('sp.external_local_unity', array([ 0.41057698,  0.33470586,  0.24904213,  0.14822411,  0.04496548])), ('sp.external_general_unity', array([ 1.        ,  1.        ,  1.        ,  1.        ,  0.56070475])), ('sp.metacentric_height', array([ 6.12375997])), ('sp.static_stability', array([ 5.98379185])), ('sp.variable_ballast_height', array([ 60.25812533])), ('sp.variable_ballast_mass', array([ 6785190.91932651])), ('sp.offset_force_ratio', array([ 1.0000217])), ('sp.heel_angle', array([ 9.99999845]))])
    '''
        
if __name__ == '__main__':
    example_spar()
    #psqp_optimal()
