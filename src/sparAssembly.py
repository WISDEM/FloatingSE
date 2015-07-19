from openmdao.main.api import Component, Assembly, convert_units
from openmdao.main.datatypes.api import Float, Array, Enum, Str, Int, Bool
from openmdao.lib.drivers.api import COBYLAdriver,SLSQPdriver
from spar import Spar
from mooring import Mooring
from tower_RNA import Tower_RNA
#from spar_discrete import spar_discrete
import numpy as np
import time
#from spar_utils import filtered_stiffeners_table

class sparAssembly(Assembly):
    tower_base_outer_diameter = Float(iotype='in',units='m',desc='outer diameter of tower base')
    tower_top_outer_diameter = Float(iotype='in',units='m',desc='outer diameter of tower top')
    tower_length = Float(iotype='in',units='m',desc='tower length')
    air_density = Float(1.198,iotype='in', units='kg/m**3', desc='density of air') 
    wind_reference_speed = Float(iotype='in', units='m/s', desc='reference wind speed')
    wind_reference_height = Float(iotype='in', units='m', desc='reference height')
    gust_factor = Float(1.0,iotype='in', desc='gust factor')
    alpha = Float(iotype='in', desc='power law exponent')
    spar_start_elevation = Array(iotype='in', units='m',desc = 'start elevation of each section')
    spar_end_elevation = Array(iotype='in', units='m',desc = 'end elevation of each section')
    example_turbine_size = Str(iotype='in',desc='for example cases, 3MW, 6MW, or 10 MW')
    rotor_diameter = Float(iotype='in', units='m',desc='rotor diameter')
    RNA_center_of_gravity_x = Float(iotype='in', units='m',desc='rotor center of gravity') 
    RNA_center_of_gravity_y = Float(iotype='in', units='m',desc='rotor center of gravity') 
    cut_out_speed = Float(25.,iotype='in', units='m/s',desc='cut-out speed of turbine') 
    tower_mass = Float(iotype='in', units='kg',desc='tower mass') 
    RNA_mass = Float(iotype='in', units='kg',desc='RNA mass') 


    fairlead_depth = Float(13.0,iotype='in',units='m',desc = 'fairlead depth')
    scope_ratio = Float(1.5,iotype='in',units='m',desc = 'scope to fairlead height ratio')
    pretension_percent = Float(5.0,iotype='in',desc='Pre-Tension Percentage of MBL (match PreTension)')
    mooring_diameter = Float(0.09,iotype='in',units='m',desc='diameter of mooring chain')
    number_of_mooring_lines = Int(3,iotype='in',desc='number of mooring lines')
    water_depth = Float(iotype='in',units='m',desc='water depth')
    mooring_type = Str('CHAIN',iotype='in',desc='CHAIN, STRAND, IWRC, or FIBER')
    anchor_type = Str('PILE',iotype='in',desc='PILE or DRAG')
    fairlead_offset_from_shell = Float(0.5,iotype='in',units='m',desc='fairlead offset from shell')
    user_MBL = Float(0.0,iotype='in',units='N',desc='user defined minimum breaking load ')
    user_WML = Float(0.0,iotype='in',units='kg/m',desc='user defined wet mass/length')
    user_AE_storm = Float(0.0,iotype='in',units='Pa',desc='user defined E modulus')
    user_MCPL = Float(0.0,iotype='in',units='USD/m',desc='user defined mooring cost per length')
    user_anchor_cost = Float(0.0,iotype='in',units='USD',desc='user defined cost per anchor')
    misc_cost_factor = Float(10.0,iotype='in',desc='miscellaneous cost factor in percent')
    number_of_discretizations = Int(20,iotype='in',desc='number of segments for mooring discretization')

    spar_outer_diameter = Array(iotype='in',units='m',desc='top outer diameter')
    water_density = Float(1025,iotype='in',units='kg/m**3',desc='density of water')

    wall_thickness = Array([0.0362,0.03427,0.03427,0.0526],iotype='in', units='m',desc = 'wall thickness of each section')
    number_of_rings = Array([1,4,5,38],iotype='in',desc = 'number of stiffeners in each section')
    neutral_axis = Float(0.38,iotype='in',units='m',desc = 'neutral axis location')
    stiffener_curve_fit = Bool(True, iotype='in', desc='flag for using optimized stiffener dimensions or discrete stiffeners')
    stiffener_index = Int(iotype='in',desc='index of stiffener from filtered table')
    number_of_sections = Int(iotype='in',desc='number of sections in the spar')
    length = Array(iotype='in', units='m',desc = 'wlength of each section')
   
    bulk_head = Array(iotype='in',desc = 'N for none, T for top, B for bottom') 
    # enviroentnment inputs
    straight_col_cost = Float(3490.,iotype='in',units='USD',desc='cost of straight columns in $/ton')
    tapered_col_cost = Float(4720.,iotype='in',units='USD',desc='cost of tapered columns in $/ton')
    outfitting_cost = Float(6980.,iotype='in',units='USD',desc='cost of tapered columns in $/ton')
    ballast_cost = Float(100.,iotype='in',units='USD',desc='cost of tapered columns in $/ton')
    gravity = Float(9.806,iotype='in', units='m/s**2', desc='gravity')
    
    
    load_condition =  Str(iotype='in',desc='Load condition - N for normal or E for extreme')
    significant_wave_height = Float(iotype='in', units='m', desc='significant wave height')
    significant_wave_period = Float(iotype='in', units='m', desc='significant wave period')
    
    
    material_density = Float(7850.,iotype='in', units='kg/m**3', desc='density of spar material')
    E = Float(200.e9,iotype='in', units='Pa', desc='young"s modulus of spar material')
    nu = Float(0.3,iotype='in', desc='poisson"s ratio of spar material')
    yield_stress = Float(345000000.,iotype='in', units='Pa', desc='yield stress of spar material')
    # ballast stuff inputs
    shell_mass_factor = Float(1.0,iotype='in',desc='shell mass factor')
    bulkhead_mass_factor = Float(1.0,iotype='in',desc='bulkhead mass factor')
    ring_mass_factor = Float(1.0,iotype='in',desc='ring mass factor')
    outfitting_factor = Float(0.06,iotype='in',desc='outfitting factor')
    spar_mass_factor = Float(1.05,iotype='in',desc='spar mass factor')
    permanent_ballast_height = Float(3.,iotype='in',units='m',desc='height of permanent ballast')
    fixed_ballast_height = Float(5.,iotype='in',units='m',desc='height of fixed ballast')
    permanent_ballast_density = Float(4492.,iotype='in',units='kg/m**3',desc='density of permanent ballast')
    fixed_ballast_density = Float(4000.,iotype='in',units='kg/m**3',desc='density of fixed ballast')
    offset_amplification_factor = Float(1.0,iotype='in',desc='amplification factor for offsets') 
    

    def configure(self):
        self.add('driver',COBYLAdriver())
        self.driver.maxfun = 100000
        # select components
        self.add('tower_RNA',Tower_RNA())
        self.add('spar',Spar())
        self.add('mooring',Mooring())
        
        # workflow
        self.driver.workflow.add(['tower_RNA', 'mooring', 'spar'])
        # connect inputs
        self.connect('air_density',['spar.air_density','tower_RNA.air_density'])
        
        # connect outputs to inputs
        self.connect('spar.shell_mass','mooring.shell_mass')
        
        

        # objective
        self.driver.add_objective('spar.spar_mooring_cost')
       
        # design variables
        self.driver.add_parameter('neutral_axis',low=100.,high=419.,scaler=0.001)
        self.driver.add_parameter('spar_number_of_rings[0]',low=1,high=5)
        self.driver.add_parameter('spar_number_of_rings[1]',low=1,high=10)
        self.driver.add_parameter('spar_number_of_rings[2]',low=1,high=10)
        self.driver.add_parameter('spar_number_of_rings[3]',low=1,high=50)
        self.driver.add_parameter('spar_wall_thickness[0]',low=100.,high=1000.,scaler=0.0001)
        self.driver.add_parameter('spar_wall_thickness[1]',low=100.,high=1000.,scaler=0.0001)
        self.driver.add_parameter('spar_wall_thickness[2]',low=100.,high=1000.,scaler=0.0001)
        self.driver.add_parameter('spar_wall_thickness[3]',low=100.,high=1000.,scaler=0.0001)
        self.driver.add_parameter('scope_ratio',low=15.,high=50.,scaler=0.1)
        self.driver.add_parameter('pretension_percent',low=2.5,high=20.)
        self.driver.add_parameter('mooring_diameter',low=3.,high=10.,scaler=0.01)
        self.driver.add_parameter('fixed_ballast_height',low=3.,high=10.)
        self.driver.add_parameter('permanent_ballast_height',low=3.,high=10.)

        # Constraints
        self.driver.add_constraint('spar.flange_compactness <= 1.')
        self.driver.add_constraint('spar.web_compactness <= 1.')

        self.driver.add_constraint('spar.VAL[0] <= 1.')
        self.driver.add_constraint('spar.VAL[1] <= 1.')
        self.driver.add_constraint('spar.VAL[2] <= 1.')
        self.driver.add_constraint('spar.VAL[3] <= 1.')

        self.driver.add_constraint('spar.VAG[0] <= 1.')
        self.driver.add_constraint('spar.VAG[1] <= 1.')
        self.driver.add_constraint('spar.VAG[2] <= 1.')
        self.driver.add_constraint('spar.VAG[3] <= 1.')

        self.driver.add_constraint('spar.VEL[0] <= 1.')
        self.driver.add_constraint('spar.VEL[1] <= 1.')
        self.driver.add_constraint('spar.VEL[2] <= 1.')
        self.driver.add_constraint('spar.VEL[3] <= 1.')

        self.driver.add_constraint('spar.VEG[0] <= 1.')
        self.driver.add_constraint('spar.VEG[1] <= 1.')
        self.driver.add_constraint('spar.VEG[2] <= 1.')
        self.driver.add_constraint('spar.VEG[3] <= 1.')

        self.driver.add_constraint('mooring.heel_angle <= 6.')
        self.driver.add_constraint('mooring.min_offset_unity < 1.0')
        self.driver.add_constraint('mooring.max_offset_unity < 1.0')

def example_218WD_3MW():
    

if __name__ == "__main__":
    example_218WD_3MW()
    #example_218WD_6MW()
    #example_218WD_10MW()