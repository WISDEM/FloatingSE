from floatingInstance import FloatingInstance, NSECTIONS, vecOption
from sparAssembly import SparAssembly
import numpy as np
    
class SparInstance(FloatingInstance):
    def __init__(self):
        super(SparInstance, self).__init__()

        # Parameters beyond those in superclass
        # Typically static- set defaults
        self.material_density = 7850.0
        self.E = 200e9
        self.nu = 0.3
        self.yield_stress = 3.45e8
        self.permanent_ballast_density = 4492.0
        self.bulkhead_mass_factor = 1.0
        self.ring_mass_factor = 1.0
        self.shell_mass_factor = 1.0
        self.spar_mass_factor = 1.05
        self.outfitting_mass_fraction = 0.06
        self.ballast_cost_rate = 100.0
        self.tapered_col_cost_rate = 4720.0
        self.outfitting_cost_rate = 6980.0

        # Typically design
        self.freeboard = 5.0
        self.fairlead = 7.57
        self.set_length(95.0)
        self.outer_radius = 7.0
        self.wall_thickness = 0.05
        self.fairlead_offset_from_shell = 0.05
        self.permanent_ballast_height = 10.0
        self.stiffener_web_height= 0.1
        self.stiffener_web_thickness = 0.04
        self.stiffener_flange_width = 0.1
        self.stiffener_flange_thickness = 0.02
        self.stiffener_spacing = 0.4

        # Change scalars to vectors where needed
        self.check_vectors()

    def set_length(self, inval):
        self.section_height =  vecOption(inval/NSECTIONS, NSECTIONS)

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

    def get_assembly(self): return SparAssembly()
    
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
