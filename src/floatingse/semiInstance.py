from floatingInstance import FloatingInstance, NSECTIONS, NPTS, vecOption
from semiAssembly import SemiAssembly
import numpy as np
import time
        
class SemiInstance(FloatingInstance):
    def __init__(self):
        super(SemiInstance, self).__init__()

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
        self.cross_attachment_pontoons = True
        self.lower_attachment_pontoons = True
        self.upper_attachment_pontoons = True
        self.lower_ring_pontoons = True
        self.upper_ring_pontoons = True
        self.pontoon_cost_rate = 6.250

        # Typically design (start at OC4 semi)
        self.radius_to_ballast_cylinder = 28.867513459481287
        self.number_of_ballast_columns = 3
        self.freeboard_base = 10.0
        self.freeboard_ballast = 12.0
        self.fairlead = 14.0
        self.fairlead_offset_from_shell = 40.868-28.867513459481287-6.0
        self.outer_radius_base = 3.25
        self.wall_thickness_base = 0.03
        self.outer_radius_ballast = 7.0
        self.wall_thickness_ballast = 0.06
        self.permanent_ballast_height_base = 10.0
        self.stiffener_web_height_base= 0.1
        self.stiffener_web_thickness_base = 0.04
        self.stiffener_flange_width_base = 0.1
        self.stiffener_flange_thickness_base = 0.02
        self.stiffener_spacing_base = 0.4
        self.permanent_ballast_height_ballast = 0.1
        self.stiffener_web_height_ballast= 0.1
        self.stiffener_web_thickness_ballast = 0.04
        self.stiffener_flange_width_ballast = 0.1
        self.stiffener_flange_thickness_ballast = 0.02
        self.stiffener_spacing_ballast = 0.4
        self.outer_pontoon_radius = 1.6
        self.inner_pontoon_radius = 1.6-0.0175

        # OC4
        self.water_depth = 200.0
        self.wave_height = 10.8
        self.wave_period = 9.8
        self.wind_reference_speed = 11.0
        self.wind_reference_height = 119.0
        self.alpha = 0.11
        self.morison_mass_coefficient = 2.0

        self.max_offset  = 0.1*self.water_depth # Assumption        
        self.number_of_mooring_lines = 3
        self.scope_ratio = 835.5 / (self.water_depth-self.fairlead) 
        self.anchor_radius = 837.6
        self.mooring_diameter = 0.0766

        self.set_length_base( 30.0 )
        self.set_length_ballast( 32.0 )

        self.section_height_ballast = np.array([6.0, 0.1, 7.9, 8.0, 10.0])
        self.outer_radius_ballast = np.array([12.0, 12.0, 6.0, 6.0, 6.0, 6.0])
        
        # Change scalars to vectors where needed
        self.check_vectors()

    def set_length_base(self, inval):
        self.section_height_base =  vecOption(inval/NSECTIONS, NSECTIONS)
    def set_length_ballast(self, inval):
        self.section_height_ballast =  vecOption(inval/NSECTIONS, NSECTIONS)

    def check_vectors(self):
        self.outer_radius_base               = vecOption(self.outer_radius_base, NSECTIONS+1)
        self.wall_thickness_base             = vecOption(self.wall_thickness_base, NSECTIONS+1)
        self.stiffener_web_height_base       = vecOption(self.stiffener_web_height_base, NSECTIONS)
        self.stiffener_web_thickness_base    = vecOption(self.stiffener_web_thickness_base, NSECTIONS)
        self.stiffener_flange_width_base     = vecOption(self.stiffener_flange_width_base, NSECTIONS)
        self.stiffener_flange_thickness_base = vecOption(self.stiffener_flange_thickness_base, NSECTIONS)
        self.stiffener_spacing_base          = vecOption(self.stiffener_spacing_base, NSECTIONS)
        self.bulkhead_nodes_base             = [False] * (NSECTIONS+1)
        self.bulkhead_nodes_base[0]          = True
        self.bulkhead_nodes_base[1]          = True
        
        self.outer_radius_ballast               = vecOption(self.outer_radius_ballast, NSECTIONS+1)
        self.wall_thickness_ballast             = vecOption(self.wall_thickness_ballast, NSECTIONS+1)
        self.stiffener_web_height_ballast       = vecOption(self.stiffener_web_height_ballast, NSECTIONS)
        self.stiffener_web_thickness_ballast    = vecOption(self.stiffener_web_thickness_ballast, NSECTIONS)
        self.stiffener_flange_width_ballast     = vecOption(self.stiffener_flange_width_ballast, NSECTIONS)
        self.stiffener_flange_thickness_ballast = vecOption(self.stiffener_flange_thickness_ballast, NSECTIONS)
        self.stiffener_spacing_ballast          = vecOption(self.stiffener_spacing_ballast, NSECTIONS)
        self.bulkhead_nodes_ballast             = [False] * (NSECTIONS+1)
        self.bulkhead_nodes_ballast[0]          = True
        self.bulkhead_nodes_ballast[1]          = True
        
    def get_assembly(self): return SemiAssembly(NSECTIONS, NPTS)
    
    def get_design_variables(self):
        # Make a neat list of design variables, lower bound, upper bound, scalar
        desvarList = [('fairlead.x',0.0, 100.0, 1.0),
                      ('fairlead_offset_from_shell.x',0.0, 5.0, 1e2),
                      ('radius_to_ballast_cylinder.x',0.0, 40.0, 1.0),
                      ('freeboard_base.x',0.0, 50.0, 1.0),
                      ('section_height_base.x',1e-1, 100.0, 1e1),
                      ('outer_radius_base.x',1.1, 25.0, 10.0),
                      ('wall_thickness_base.x',5e-3, 1.0, 1e3),
                      ('freeboard_ballast.x',0.0, 50.0, 1.0),
                      ('section_height_ballast.x',1e-1, 100.0, 1e1),
                      ('outer_radius_ballast.x',1.1, 25.0, 10.0),
                      ('wall_thickness_ballast.x',5e-3, 1.0, 1e3),
                      ('outer_pontoon_radius.x', 0.1, 5.0, 1.0),
                      ('inner_pontoon_radius.x', 0.02, 4.95, 1.0),
                      ('scope_ratio.x', 1.0, 5.0, 1.0),
                      ('anchor_radius.x', 1.0, 1e3, 1e-2),
                      ('mooring_diameter.x', 0.05, 1.0, 1e1),
                      ('stiffener_web_height_base.x', 1e-2, 1.0, 1e2),
                      ('stiffener_web_thickness_base.x', 1e-3, 5e-1, 1e2),
                      ('stiffener_flange_width_base.x', 1e-2, 5.0, 1e2),
                      ('stiffener_flange_thickness_base.x', 1e-3, 5e-1, 1e2),
                      ('stiffener_spacing_base.x', 1e-1, 1e2, 1e1),
                      ('permanent_ballast_height_base.x', 1e-1, 50.0, 1.0),
                      ('stiffener_web_height_ballast.x', 1e-2, 1.0, 1e2),
                      ('stiffener_web_thickness_ballast.x', 1e-3, 5e-1, 1e2),
                      ('stiffener_flange_width_ballast.x', 1e-2, 5.0, 1e2),
                      ('stiffener_flange_thickness_ballast.x', 1e-3, 5e-1, 1e2),
                      ('stiffener_spacing_ballast.x', 1e-1, 1e2, 1e1),
                      ('permanent_ballast_height_ballast.x', 1e-1, 50.0, 1.0)]

        # TODO: Integer and Boolean design variables
        #prob.driver.add_desvar('number_of_ballast_columns.x', lower=1)
        #prob.driver.add_desvar('number_of_mooring_lines.x', lower=1)
        #prob.driver.add_desvar('mooring_type.x')
        #prob.driver.add_desvar('anchor_type.x')
        #prob.driver.add_desvar('bulkhead_nodes.x')
        #prob.driver.add_desvar('cross_attachment_pontoons.x')
        #prob.driver.add_desvar('lower_attachment_pontoons.x')
        #prob.driver.add_desvar('upper_attachment_pontoons.x')
        #prob.driver.add_desvar('lower_ring_pontoons.x')
        #prob.driver.add_desvar('upper_ring_pontoons.x')
        return desvarList

    def add_constraints_objective(self):

        # CONSTRAINTS
        # These are mostly the outputs that were not connected to another model

        # Ensure that draft is greater than 0 (spar length>0) and that less than water depth
        # Ensure that fairlead attaches to draft
        self.prob.driver.add_constraint('sg.base_draft_depth_ratio',lower=0.0, upper=0.75)
        self.prob.driver.add_constraint('sg.ballast_draft_depth_ratio',lower=0.0, upper=0.75)
        self.prob.driver.add_constraint('sg.fairlead_draft_ratio',lower=0.0, upper=1.0)
        self.prob.driver.add_constraint('sg.base_ballast_spacing',lower=0.0, upper=1.0)

        # Ensure that the radius doesn't change dramatically over a section
        self.prob.driver.add_constraint('sg.base_taper_ratio',upper=0.1)
        self.prob.driver.add_constraint('sg.ballast_taper_ratio',upper=0.1)

        # Ensure that the spar top matches the tower base
        self.prob.driver.add_constraint('sg.transition_radius',lower=0.0, upper=5.0)
        
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

        self.prob.driver.add_constraint('ball.flange_spacing_ratio', upper=0.5)
        self.prob.driver.add_constraint('ball.web_radius_ratio', upper=0.5)
        self.prob.driver.add_constraint('ball.flange_compactness', lower=1.0)
        self.prob.driver.add_constraint('ball.web_compactness', lower=1.0)
        self.prob.driver.add_constraint('ball.axial_local_unity', upper=1.0)
        self.prob.driver.add_constraint('ball.axial_general_unity', upper=1.0)
        self.prob.driver.add_constraint('ball.external_local_unity', upper=1.0)
        self.prob.driver.add_constraint('ball.external_general_unity', upper=1.0)

        # Pontoon tube radii
        self.prob.driver.add_constraint('pon.pontoon_radii_ratio', upper=1.0)

        # Pontoon stress safety factor
        self.prob.driver.add_constraint('pon.axial_stress_factor', upper=0.8)
        self.prob.driver.add_constraint('pon.shear_stress_factor', upper=0.8)
        
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
        self.prob.driver.add_constraint('sm.heel_angle',lower=0.0, upper=10.0)


        # OBJECTIVE FUNCTION: Minimize total cost!
        self.prob.driver.add_objective('sm.total_cost', scaler=1e-9)


        
    def visualize(self, fname=None):
        fig = self.init_figure()

        self.draw_ocean(fig)

        mooringMat = self.prob['mm.plot_matrix']
        self.draw_mooring(fig, mooringMat)

        pontoonMat = self.prob['pon.plot_matrix']
        zcut = 1.0 + np.maximum( self.freeboard_base, self.freeboard_ballast )
        self.draw_pontoons(fig, pontoonMat, self.outer_pontoon_radius, zcut)

        self.draw_cylinder(fig, [0.0, 0.0], self.freeboard_base, self.section_height_base,
                           self.outer_radius_base, self.stiffener_spacing_base)

        R_semi    = self.radius_to_ballast_cylinder
        ncylinder = self.number_of_ballast_columns
        angles = np.linspace(0, 2*np.pi, ncylinder+1)
        x = R_semi * np.cos( angles )
        y = R_semi * np.sin( angles )
        for k in xrange(ncylinder):
            self.draw_cylinder(fig, [x[k], y[k]], self.freeboard_ballast, self.section_height_ballast,
                               self.outer_radius_ballast, self.stiffener_spacing_ballast)
            
        self.set_figure(fig, fname)








        


        
def example_semi():
    mysemi = SemiInstance()
    #mysemi.evaluate('psqp')
    #mysemi.visualize('semi-initial.jpg')
    mysemi.run('psqp')
    return mysemi

def psqp_optimal():
    #OrderedDict([('sm.total_cost', array([0.65987536]))])
    mysemi = SemiInstance()

    mysemi.fairlead = 22.2366002
    mysemi.fairlead_offset_from_shell = 4.99949523
    mysemi.radius_to_ballast_cylinder = 26.79698385
    mysemi.freeboard_base = 4.97159308
    mysemi.section_height_base = np.array([6.72946378, 5.97993104, 5.47072089, 5.71437475, 5.44290777])
    mysemi.outer_radius_base = np.array([2.0179943 , 2.21979373, 2.4417731 , 2.68595041, 2.95454545, 3.25 ])
    mysemi.wall_thickness_base = np.array([0.01100738, 0.00722966, 0.00910002, 0.01033024, 0.00639292, 0.00560714])
    mysemi.freeboard_ballast = -1.14370386e-20
    mysemi.section_height_ballast = np.array([1.44382195, 2.71433629, 6.1047888 , 5.14428218, 6.82937098])
    mysemi.outer_radius_ballast = np.array([2.57228724, 2.82647421, 3.10005118, 3.40594536, 3.74653989, 4.12119389])
    mysemi.wall_thickness_ballast = np.array([0.01558312, 0.005 , 0.005 , 0.005 , 0.005 , 0.005 ])
    mysemi.outer_pontoon_radius = 0.92428188
    mysemi.inner_pontoon_radius = 0.88909984
    mysemi.scope_ratio = 4.71531904
    mysemi.anchor_radius = 837.58954811
    mysemi.mooring_diameter = 0.36574595
    mysemi.stiffener_web_height_base = np.array([0.01625364, 0.04807025, 0.07466081, 0.0529478 , 0.03003529])
    mysemi.stiffener_web_thickness_base = np.array([0.00263325, 0.00191218, 0.00404707, 0.00495706, 0.00137335])
    mysemi.stiffener_flange_width_base = np.array([0.0100722 , 0.06406752, 0.01342377, 0.07119415, 0.01102604])
    mysemi.stiffener_flange_thickness_base = np.array([0.06126737, 0.00481305, 0.01584461, 0.00980356, 0.01218029])
    mysemi.stiffener_spacing_base = np.array([1.09512893, 0.67001459, 1.60080836, 1.27068546, 0.2687786 ])
    mysemi.permanent_ballast_height_base = 5.34047386
    mysemi.stiffener_web_height_ballast = np.array([0.04750412, 0.03926778, 0.04484479, 0.04255339, 0.05903525])
    mysemi.stiffener_web_thickness_ballast = np.array([0.00197299, 0.00162998, 0.00186254, 0.00176738, 0.00245192])
    mysemi.stiffener_flange_width_ballast = np.array([0.01176864, 0.01018018, 0.01062256, 0.01119399, 0.01023957])
    mysemi.stiffener_flange_thickness_ballast = np.array([0.00182314, 0.00428608, 0.01616793, 0.0109717 , 0.00814284])
    mysemi.stiffener_spacing_ballast = np.array([0.88934305, 0.19623501, 0.29410086, 0.30762027, 0.4208429 ])
    mysemi.permanent_ballast_height_ballast = 0.10007504

    mysemi.evaluate('psqp')
    mysemi.visualize('semi-psqp.jpg')
    return mysemi
    
    '''
OrderedDict([('fairlead.x', array([22.2366002])), ('fairlead_offset_from_shell.x', array([4.99949523])), ('radius_to_ballast_cylinder.x', array([26.79698385])), ('freeboard_base.x', array([4.97159308])), ('section_height_base.x', array([6.72946378, 5.97993104, 5.47072089, 5.71437475, 5.44290777])), ('outer_radius_base.x', array([2.0179943 , 2.21979373, 2.4417731 , 2.68595041, 2.95454545,
       3.25      ])), ('wall_thickness_base.x', array([0.01100738, 0.00722966, 0.00910002, 0.01033024, 0.00639292,
       0.00560714])), ('freeboard_ballast.x', array([-1.14370386e-20])), ('section_height_ballast.x', array([1.44382195, 2.71433629, 6.1047888 , 5.14428218, 6.82937098])), ('outer_radius_ballast.x', array([2.57228724, 2.82647421, 3.10005118, 3.40594536, 3.74653989,
       4.12119389])), ('wall_thickness_ballast.x', array([0.01558312, 0.005     , 0.005     , 0.005     , 0.005     ,
       0.005     ])), ('outer_pontoon_radius.x', array([0.92428188])), ('inner_pontoon_radius.x', array([0.88909984])), ('scope_ratio.x', array([4.71531904])), ('anchor_radius.x', array([837.58954811])), ('mooring_diameter.x', array([0.36574595])), ('stiffener_web_height_base.x', array([0.01625364, 0.04807025, 0.07466081, 0.0529478 , 0.03003529])), ('stiffener_web_thickness_base.x', array([0.00263325, 0.00191218, 0.00404707, 0.00495706, 0.00137335])), ('stiffener_flange_width_base.x', array([0.0100722 , 0.06406752, 0.01342377, 0.07119415, 0.01102604])), ('stiffener_flange_thickness_base.x', array([0.06126737, 0.00481305, 0.01584461, 0.00980356, 0.01218029])), ('stiffener_spacing_base.x', array([1.09512893, 0.67001459, 1.60080836, 1.27068546, 0.2687786 ])), ('permanent_ballast_height_base.x', array([5.34047386])), ('stiffener_web_height_ballast.x', array([0.04750412, 0.03926778, 0.04484479, 0.04255339, 0.05903525])), ('stiffener_web_thickness_ballast.x', array([0.00197299, 0.00162998, 0.00186254, 0.00176738, 0.00245192])), ('stiffener_flange_width_ballast.x', array([0.01176864, 0.01018018, 0.01062256, 0.01119399, 0.01023957])), ('stiffener_flange_thickness_ballast.x', array([0.00182314, 0.00428608, 0.01616793, 0.0109717 , 0.00814284])), ('stiffener_spacing_ballast.x', array([0.88934305, 0.19623501, 0.29410086, 0.30762027, 0.4208429 ])), ('permanent_ballast_height_ballast.x', array([0.10007504]))])
    '''
    
if __name__ == '__main__':
    #psqp_optimal()
    example_semi()
