from floatingInstance import FloatingInstance, NSECTIONS, vecOption
from semiAssembly import SemiAssembly
import numpy as np

        
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
        
    def get_assembly(self): return SemiAssembly()
    
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
        
