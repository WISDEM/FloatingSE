from floatingInstance import FloatingInstance, NSECTIONS, NPTS, vecOption
from semiAssembly import SemiAssembly
from commonse import eps
import numpy as np
import time
        
class SparInstance(FloatingInstance):
    def __init__(self):
        super(SparInstance, self).__init__()

        # Parameters beyond those in superclass
        self.params['number_of_ballast_columns'] = 0

        # Environmental parameters
        self.params['water_depth'] = 218.0
        #self.params['air_density'] = 1.198
        self.params['base.windLoads.rho'] = 1.198
        #self.params['air_viscosity'] = 1.81e-5
        self.params['base.windLoads.mu'] = 1.81e-5
        self.params['water_density'] = 1025.0
        #self.params['water_viscosity'] = 8.9e-4
        self.params['base.waveLoads.mu'] = 8.9e-4
        #self.params['wave_height'] = 10.8
        self.params['hmax'] = 10.8
        #self.params['wave_period'] = 9.8
        self.params['T'] = 9.8
        self.params['Uc'] = 0.0
        #self.params['wind_reference_speed'] = 11.73732
        self.params['Uref'] = 11.73732
        #self.params['wind_reference_height'] = 90.0
        self.params['zref'] = 90.0
        #self.params['alpha'] = 0.11
        self.params['shearExp'] = 0.11
        #self.params['morison_mass_coefficient'] = 2.0
        self.params['cm'] = 2.0
        self.params['z0'] = 0.0
        self.params['yaw'] = 0.0
        self.params['beta'] = 0.0
        self.params['cd_usr'] = np.inf

        # Mooring parameters
        self.params['mooring_max_offset'] = 0.1*self.params['water_depth'] # Assumption        
        self.params['mooring_max_heel'] = 10.0
        self.params['number_of_mooring_lines'] = 3
        self.params['mooring_type'] = 'chain'
        self.params['anchor_type'] = 'suctionpile'
        self.params['mooring_cost_rate'] = 1.1
        self.params['scope_ratio'] = 2.6
        self.params['anchor_radius'] = 420.0
        self.params['drag_embedment_extra_length'] = 300.0
        
        self.params['mooring_diameter'] = 0.14

        # Turbine parameters
        self.params['turbine_mass'] = 371690.0 + 285599.0
        self.params['turbine_I_base'] = np.array([3.05284574e9, 2.96031642e9, 2.13639924e7, 0.0, 2.89884849e7, 0.0])
        self.params['turbine_center_of_gravity'] = np.array([0.0, 0.0, 43.4])
        self.params['turbine_force'] = np.array( [1.30015900e+06, -3.14321369e-08, -9.12169902e+06] )
        self.params['turbine_moment'] = np.array( [-1.68366922e+06, 1.02556564e+08, 1.47301970e+05] )

        # Steel properties
        self.params['material_density'] = 7850.0
        self.params['E'] = 200e9
        self.params['G'] = 79.3e9
        self.params['nu'] = 0.26
        self.params['yield_stress'] = 3.45e8
        self.params['loading'] = 'hydrostatic'

        # Design parameters
        self.params['min_taper'] = 0.4
        self.params['min_d_to_t'] = 120.0

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
        self.params['cross_attachment_pontoons'] = False
        self.params['lower_attachment_pontoons'] = False
        self.params['upper_attachment_pontoons'] = False
        self.params['lower_ring_pontoons'] = False
        self.params['upper_ring_pontoons'] = False
        self.params['outer_cross_pontoons'] = False
        self.params['pontoon_cost_rate'] = 6.250

        # Typically design (start at OC4 semi)
        self.params['outer_diameter_ballast'] = 2.0
        self.set_length_ballast(1.0)
        self.params['wall_thickness_ballast'] = 2e-2
        self.params['freeboard_ballast'] = 0.1
        self.params['radius_to_ballast_column'] = 28.867513459481287
        self.params['wall_thickness_ballast'] = 0.06
        self.params['stiffener_web_height_base'] = 0.1
        self.params['stiffener_web_thickness_base'] = 0.04
        self.params['stiffener_flange_width_base'] = 0.1
        self.params['stiffener_flange_thickness_base'] = 0.02
        self.params['stiffener_spacing_base'] = 0.4
        self.params['permanent_ballast_height_ballast'] = 0.1
        self.params['stiffener_web_height_ballast'] = 0.1
        self.params['stiffener_web_thickness_ballast'] = 0.04
        self.params['stiffener_flange_width_ballast'] = 0.1
        self.params['stiffener_flange_thickness_ballast'] = 0.02
        self.params['stiffener_spacing_ballast'] = 0.4
        self.params['pontoon_outer_diameter'] = 2*1.6
        self.params['pontoon_wall_thickness'] = 0.0175
        self.params['connection_ratio_max'] = 0.25
        self.params['base_pontoon_attach_lower'] = -20.0
        self.params['base_pontoon_attach_upper'] = 10.0
 
        # Typically design (OC3)
        self.params['freeboard_base'] = 10.0
        self.params['fairlead'] = 70.0
        self.set_length_base(130.0)
        self.params['section_height_base'] = np.array([36.0, 36.0, 36.0, 8.0, 14.0])
        self.params['outer_diameter_base'] = 2*np.array([4.7, 4.7, 4.7, 4.7, 3.25, 3.25])
        self.params['wall_thickness_base'] = 0.05
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

        self.params['tower_metric'] = 6.5
        
        # Change scalars to vectors where needed
        self.check_vectors()

    def set_length_base(self, inval):
        self.params['section_height_base'] =  vecOption(inval/NSECTIONS, NSECTIONS)
    def set_length_ballast(self, inval):
        self.params['section_height_ballast'] =  vecOption(inval/NSECTIONS, NSECTIONS)

    def check_vectors(self):
        self.params['tower_metric']                    = vecOption(self.params['tower_metric'], NSECTIONS+1)
        self.params['outer_diameter_base']             = vecOption(self.params['outer_diameter_base'], NSECTIONS+1)
        self.params['wall_thickness_base']             = vecOption(self.params['wall_thickness_base'], NSECTIONS+1)
        self.params['stiffener_web_height_base']       = vecOption(self.params['stiffener_web_height_base'], NSECTIONS)
        self.params['stiffener_web_thickness_base']    = vecOption(self.params['stiffener_web_thickness_base'], NSECTIONS)
        self.params['stiffener_flange_width_base']     = vecOption(self.params['stiffener_flange_width_base'], NSECTIONS)
        self.params['stiffener_flange_thickness_base'] = vecOption(self.params['stiffener_flange_thickness_base'], NSECTIONS)
        self.params['stiffener_spacing_base']          = vecOption(self.params['stiffener_spacing_base'], NSECTIONS)
        self.params['bulkhead_nodes_base']             = [False] * (NSECTIONS+1)
        self.params['bulkhead_nodes_base'][0]          = True
        self.params['bulkhead_nodes_base'][1]          = True
        
        self.params['outer_diameter_ballast']             = vecOption(self.params['outer_diameter_ballast'], NSECTIONS+1)
        self.params['wall_thickness_ballast']             = vecOption(self.params['wall_thickness_ballast'], NSECTIONS+1)
        self.params['stiffener_web_height_ballast']       = vecOption(self.params['stiffener_web_height_ballast'], NSECTIONS)
        self.params['stiffener_web_thickness_ballast']    = vecOption(self.params['stiffener_web_thickness_ballast'], NSECTIONS)
        self.params['stiffener_flange_width_ballast']     = vecOption(self.params['stiffener_flange_width_ballast'], NSECTIONS)
        self.params['stiffener_flange_thickness_ballast'] = vecOption(self.params['stiffener_flange_thickness_ballast'], NSECTIONS)
        self.params['stiffener_spacing_ballast']          = vecOption(self.params['stiffener_spacing_ballast'], NSECTIONS)
        self.params['bulkhead_nodes_ballast']             = [False] * (NSECTIONS+1)
        self.params['bulkhead_nodes_ballast'][0]          = True
        self.params['bulkhead_nodes_ballast'][1]          = True
        
    def get_assembly(self): return SemiAssembly(NSECTIONS)
    
    def get_design_variables(self):
        # Make a neat list of design variables, lower bound, upper bound, scalar
        desvarList = [('fairlead',0.0, 100.0, 1.0),
                      ('fairlead_offset_from_shell',0.0, 5.0, 1e2),
                      ('freeboard_base',0.0, 50.0, 1.0),
                      ('section_height_base',1e-1, 100.0, 1e1),
                      ('outer_diameter_base',1.1, 40.0, 10.0),
                      ('wall_thickness_base',5e-3, 1.0, 1e3),
                      ('pontoon_outer_diameter', 1e-1, 3.0, 10.0),
                      ('pontoon_wall_thickness', 5e-3, 1e-1, 100.0),
                      ('base_pontoon_attach_lower',-1e2, 1e2, 1.0),
                      ('base_pontoon_attach_upper',-1e2, 1e2, 1.0),
                      ('scope_ratio', 1.0, 5.0, 1.0),
                      ('anchor_radius', 1.0, 1e3, 1e-2),
                      ('mooring_diameter', 0.05, 1.0, 1e1),
                      ('stiffener_web_height_base', 1e-2, 1.0, 1e2),
                      ('stiffener_web_thickness_base', 1e-3, 5e-1, 1e2),
                      ('stiffener_flange_width_base', 1e-2, 5.0, 1e2),
                      ('stiffener_flange_thickness_base', 1e-3, 5e-1, 1e2),
                      ('stiffener_spacing_base', 1e-1, 1e2, 1e1)]

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

    def add_constraints_objective(self):

        # CONSTRAINTS
        # These are mostly the outputs that were not connected to another model

        # Ensure that draft is greater than 0 (spar length>0) and that less than water depth
        # Ensure that fairlead attaches to draft
        self.prob.driver.add_constraint('base.draft_depth_ratio',lower=0.0, upper=0.75)

        # Ensure that the radius doesn't change dramatically over a section
        self.prob.driver.add_constraint('base.manufacturability',upper=0.0)
        self.prob.driver.add_constraint('base.weldability',upper=0.0)

        # Ensure that the spar top matches the tower base
        self.prob.driver.add_constraint('tt.transition_buffer',lower=0.0, upper=5.0)
        
        # Ensure max mooring line tension is less than X% of MBL: 60% for intact mooring, 80% for damanged
        self.prob.driver.add_constraint('mm.safety_factor',lower=0.0, upper=0.8)

        # Ensure there is sufficient mooring line length, MAP doesn't give an error about this
        self.prob.driver.add_constraint('mm.mooring_length_min',lower=1.0)
        self.prob.driver.add_constraint('mm.mooring_length_max',upper=1.0)

        # API Bulletin 2U constraints
        self.prob.driver.add_constraint('base.flange_spacing_ratio', upper=0.5)
        self.prob.driver.add_constraint('base.web_radius_ratio', upper=0.5)
        self.prob.driver.add_constraint('base.flange_compactness', lower=1.0)
        self.prob.driver.add_constraint('base.web_compactness', lower=1.0)
        self.prob.driver.add_constraint('base.axial_local_unity', upper=1.0)
        self.prob.driver.add_constraint('base.axial_general_unity', upper=1.0)
        self.prob.driver.add_constraint('base.external_local_unity', upper=1.0)
        self.prob.driver.add_constraint('base.external_general_unity', upper=1.0)

        # Pontoon stress safety factor
        #self.prob.driver.add_constraint('pon.axial_stress_factor', upper=0.8)
        #self.prob.driver.add_constraint('pon.shear_stress_factor', upper=0.8)
        
        # Achieving non-zero variable ballast height means the semi can be balanced with margin as conditions change
        self.prob.driver.add_constraint('sm.variable_ballast_height', lower=2.0, upper=100.0)
        self.prob.driver.add_constraint('sm.variable_ballast_mass', lower=0.0)

        # Metacentric height should be positive for static stability
        self.prob.driver.add_constraint('sm.metacentric_height', lower=0.1)

        # Center of buoyancy should be above CG (difference should be positive)
        self.prob.driver.add_constraint('sm.static_stability', lower=0.1)

        # Surge restoring force should be greater than wave-wind forces (ratio < 1)
        self.prob.driver.add_constraint('sm.offset_force_ratio',lower=0.0, upper=1.0)

        # Heel angle should be less than 6deg for ordinary operation, less than 10 for extreme conditions
        self.prob.driver.add_constraint('sm.heel_constraint',lower=0.0, upper=1.0)


        # OBJECTIVE FUNCTION: Minimize total cost!
        self.prob.driver.add_objective('total_cost', scaler=1e-9)


        
    def visualize(self, fname=None):
        fig = self.init_figure()

        self.draw_ocean(fig)

        mooringMat = self.prob['mm.plot_matrix']
        self.draw_mooring(fig, mooringMat)

        self.draw_column(fig, [0.0, 0.0], self.params['freeboard_base'], self.params['section_height_base'],
                           0.5*self.params['outer_diameter_base'], self.params['stiffener_spacing_base'])

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
    
    '''
OrderedDict([('fairlead', array([13.48595326])), ('fairlead_offset_from_shell', array([4.99157996])), ('radius_to_ballast_column', array([30.18481219])), ('freeboard_base', array([9.26170226])), ('section_height_base', array([3.7604404 , 5.01424445, 4.33725659, 3.33214966, 3.2105037 ])), ('outer_diameter_base', array([1.10000012, 1.20837321, 2.89237172, 4.53026726, 3.63197871,
       6.4586864 ])), ('wall_thickness_base', array([0.00649224, 0.00676997, 0.005     , 0.005     , 0.00625276,
       0.00912208])), ('freeboard_ballast', array([5.21222562])), ('section_height_ballast', array([0.46854124, 0.26703874, 6.17196415, 6.07947859, 5.44340214])), ('outer_diameter_ballast', array([ 9.98087107, 11.65105409,  9.93591842, 11.24439331,  9.02905809,
        7.08313227])), ('wall_thickness_ballast', array([0.005, 0.005, 0.005, 0.005, 0.005, 0.005])), ('pontoon_outer_diameter', array([4.43693504])), ('pontoon_inner_diameter', array([4.4333937])), ('scope_ratio', array([4.70199041])), ('anchor_radius', array([837.57485028])), ('mooring_diameter', array([0.66993187])), ('stiffener_web_height_base', array([0.01518644, 0.01      , 0.01452209, 0.05063256, 0.02239869])), ('stiffener_web_thickness_base', array([0.001     , 0.001     , 0.001     , 0.00965457, 0.00119531])), ('stiffener_flange_width_base', array([0.01      , 0.01      , 0.01      , 0.06754642, 0.01      ])), ('stiffener_flange_thickness_base', array([0.00109441, 0.001     , 0.001     , 0.04809957, 0.001     ])), ('stiffener_spacing_base', array([0.62138686, 0.55886008, 0.17846304, 0.21853178, 0.33977136])), ('permanent_ballast_height_base', array([5.46194272])), ('stiffener_web_height_ballast', array([0.01      , 0.01000003, 0.01      , 0.01299782, 0.01      ])), ('stiffener_web_thickness_ballast', array([0.001, 0.001, 0.001, 0.001, 0.001])), ('stiffener_flange_width_ballast', array([0.01      , 0.06168182, 0.01      , 0.01      , 0.01      ])), ('stiffener_flange_thickness_ballast', array([0.001     , 0.00494582, 0.001     , 0.001     , 0.001     ])), ('stiffener_spacing_ballast', array([2.07754793, 0.14845455, 0.71661921, 0.32008663, 1.9621595 ])), ('permanent_ballast_height_ballast', array([0.1]))])
OrderedDict([('total_cost', array([0.61991178]))])
    '''
    
if __name__ == '__main__':
    #psqp_optimal()
    example_spar()
