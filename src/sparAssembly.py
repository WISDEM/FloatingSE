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
    ##### ENVIRONEMENT & MATERIALS #####
    # tower_RNA inputs
    air_density = Float(1.198,iotype='in', units='kg/m**3', desc='density of air')
    wind_reference_speed = Float(iotype='in', units='m/s', desc='reference wind speed')
    wind_reference_height = Float(iotype='in', units='m', desc='reference height')
    gust_factor = Float(1.0,iotype='in', desc='gust factor')
    alpha = Float(iotype='in', desc='power law exponent')
    # mooring 
    water_depth = Float(iotype='in',units='m',desc='water depth')
    water_density = Float(1025,iotype='in',units='kg/m**3',desc='density of water')
    permanent_ballast_density = Float(4492.,iotype='in',units='kg/m**3',desc='density of permanent ballast')
    fixed_ballast_density = Float(4000.,iotype='in',units='kg/m**3',desc='density of fixed ballast')
    gravity = Float(9.806,iotype='in', units='m/s**2', desc='gravity')
    # spar
    significant_wave_height = Float(iotype='in', units='m', desc='significant wave height')
    significant_wave_period = Float(iotype='in', units='m', desc='significant wave period')
    spar_material_density = Float(7850.,iotype='in', units='kg/m**3', desc='density of spar material')
    spar_E = Float(200.e9,iotype='in', units='Pa', desc='young"s modulus of spar material')
    spar_nu = Float(0.3,iotype='in', desc='poisson"s ratio of spar material')
    spar_yield_stress = Float(345000000.,iotype='in', units='Pa', desc='yield stress of spar material')
    cut_out_speed = Float(25.,iotype='in', units='m/s',desc='cut out speed of turbine')
    ##### INPUTS #####
    # spar
    initial_pass = Bool(True, iotype='in', desc='flag for using optimized stiffener dimensions or discrete stiffeners')
    stiffener_index = Int(iotype='in',desc='index of stiffener from filtered table')
    load_condition =  Str('N',iotype='in',desc='Load condition - N for normal or E for extreme')
    number_of_sections = Int(iotype='in',desc='number of sections in the spar')
    spar_lengths = Array(iotype='in', units='m',desc = 'wlength of each section')
    spar_end_elevation = Array(iotype='in', units='m',desc = 'end elevation of each section')
    spar_start_elevation = Array(iotype='in', units='m',desc = 'start elevation of each section')
    bulk_head = Array(['N','T','N','B'],iotype='in',desc = 'N for none, T for top, B for bottom')
    # mooring
    number_of_mooring_lines = Int(3,iotype='in',desc='number of mooring lines')
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
    shell_mass_factor = Float(1.0,iotype='in',desc='shell mass factor')
    bulkhead_mass_factor = Float(1.0,iotype='in',desc='bulkhead mass factor')
    ring_mass_factor = Float(1.0,iotype='in',desc='ring mass factor')
    outfitting_factor = Float(0.06,iotype='in',desc='outfitting factor')
    spar_mass_factor = Float(1.05,iotype='in',desc='spar mass factor')
    amplification_factor = Float(1.0,iotype='in',desc='amplification factor for offsets') 
    # tower_RNA
    tower_base_outer_diameter = Float(iotype='in',units='m',desc='outer diameter of tower base')
    tower_top_outer_diameter = Float(iotype='in',units='m',desc='outer diameter of tower top')
    tower_length = Float(iotype='in',units='m',desc='tower length')
    RNA_mass = Float(iotype='in', units='kg',desc='mass of RNA')
    tower_mass =  Float(iotype='in',units='kg',desc='tower mass')
    example_turbine_size = Str(iotype='in',desc='for example cases, 3MW, 6MW, or 10 MW')
    rotor_diameter = Float(iotype='in', units='m',desc='rotor diameter')
    RNA_center_of_gravity_x = Float(iotype='in', units='m',desc='rotor center of gravity') 
    RNA_center_of_gravity_y = Float(iotype='in', units='m',desc='rotor center of gravity') 
    ##### (POSSIBLE) DESIGN VARIABLES #####
    fairlead_depth = Float(13.0,iotype='in',units='m',desc = 'fairlead depth')
    scope_ratio = Float(1.5,iotype='in',units='m',desc = 'scope to fairlead height ratio')
    pretension_percent = Float(5.0,iotype='in',desc='Pre-Tension Percentage of MBL (match PreTension)')
    mooring_diameter = Float(0.09,iotype='in',units='m',desc='diameter of mooring chain')
    permanent_ballast_height = Float(iotype='in',units='m',desc='height of permanent ballast')
    fixed_ballast_height = Float(iotype='in',units='m',desc='height of fixed ballast')
    spar_wall_thickness = Array([0.027,0.025,0.027,0.047],iotype='in', units='m',desc = 'wall thickness of each section')
    spar_number_of_rings = Array([1,4,4,23],iotype='in',desc = 'number of stiffeners in each section')
    neutral_axis = Float(0.285,iotype='in',units='m',desc = 'neutral axis location')
    spar_outer_diameters = Array(iotype='in', units='m',desc = 'outer diameter of each section')

    ##### CONSTRAINTS & OUTPUTS #####
    #flange_compactness = Float(iotype='out',desc = 'check for flange compactness')
    #web_compactness = Float(iotype='out',desc = 'check for web compactness')
    #VAL = Array(iotype='out',desc = 'unity check for axial load - local buckling')
    #VAG = Array(iotype='out',desc = 'unity check for axial load - genenral instability')
    #VEL = Array(iotype='out',desc = 'unity check for external pressure - local buckling')
    #VEG = Array(iotype='out',desc = 'unity check for external pressure - general instability')
    #heel_angle = Float(iotype='out',desc='heel angle unity check')
    #min_offset_unity = Float(iotype='out',desc='minimum offset unity check')
    #max_offset_unity = Float(iotype='out',desc='maximum offset unity check')
    #spar_mooring_cost = Float(iotype='out',units='USD',desc='total cost for anchor + legs + miscellaneous costs')
    
    def configure(self):
        self.add('driver',COBYLAdriver())
        self.driver.maxfun = 100000
        # select components
        self.add('spar',Spar())
        self.add('mooring',Mooring())
        self.add('tower_RNA',Tower_RNA())
        # workflow
        self.driver.workflow.add(['spar', 'mooring', 'tower_RNA'])
        # connect inputs
        self.connect('air_density',['spar.air_density','tower_RNA.air_density'])
        self.connect('wind_reference_speed',['spar.wind_reference_speed','tower_RNA.wind_reference_speed'])
        self.connect('wind_reference_height',['spar.wind_reference_height','tower_RNA.wind_reference_height'])
        self.connect('gust_factor',['spar.gust_factor','tower_RNA.gust_factor'])
        self.connect('alpha',['spar.alpha','tower_RNA.alpha'])
        self.connect('water_depth',['spar.water_depth','mooring.water_depth'])
        self.connect('water_density',['spar.water_density','mooring.water_density'])
        self.connect('permanent_ballast_density','mooring.permanent_ballast_density')
        self.connect('fixed_ballast_density','mooring.fixed_ballast_density')
        self.connect('gravity',['spar.gravity','mooring.gravity'])
        self.connect('significant_wave_height','spar.significant_wave_height')
        self.connect('significant_wave_period','spar.significant_wave_period')
        self.connect('spar_material_density','spar.material_density')
        self.connect('spar_E','spar.E')
        self.connect('spar_nu','spar.nu')
        self.connect('spar_yield_stress','spar.yield_stress')
        self.connect('cut_out_speed','tower_RNA.cut_out_speed')
        self.connect('initial_pass','spar.initial_pass')
        self.connect('stiffener_index','spar.stiffener_index')
        self.connect('load_condition',['spar.load_condition','mooring.load_condition'])
        self.connect('number_of_sections','spar.number_of_sections')
        self.connect('spar_lengths','spar.length')
        self.connect('spar_end_elevation',['spar.end_elevation','tower_RNA.spar_end_elevation'])
        self.connect('spar_start_elevation','spar.start_elevation')
        self.connect('bulk_head','spar.bulk_head')
        self.connect('number_of_mooring_lines','mooring.number_of_mooring_lines')
        self.connect('mooring_type','mooring.mooring_type')
        self.connect('anchor_type','mooring.anchor_type')
        self.connect('fairlead_offset_from_shell','mooring.fairlead_offset_from_shell')
        self.connect('user_MBL','mooring.user_MBL')
        self.connect('user_WML','mooring.user_WML')
        self.connect('user_AE_storm','mooring.user_AE_storm')
        self.connect('user_MCPL','mooring.user_MCPL')
        self.connect('user_anchor_cost','mooring.user_anchor_cost')
        self.connect('misc_cost_factor','mooring.misc_cost_factor')
        self.connect('number_of_discretizations','mooring.number_of_discretizations')
        self.connect('shell_mass_factor','mooring.shell_mass_factor')
        self.connect('bulkhead_mass_factor','mooring.bulkhead_mass_factor')
        self.connect('ring_mass_factor','mooring.ring_mass_factor')
        self.connect('outfitting_factor','mooring.outfitting_factor')
        self.connect('spar_mass_factor','mooring.spar_mass_factor')
        self.connect('amplification_factor','mooring.amplification_factor')
        self.connect('tower_base_outer_diameter','tower_RNA.base_outer_diameter')
        self.connect('tower_top_outer_diameter','tower_RNA.top_outer_diameter')
        self.connect('tower_length','tower_RNA.length')
        self.connect('RNA_mass',['mooring.RNA_mass','spar.RNA_mass'])
        self.connect('tower_mass',['mooring.tower_mass','spar.tower_mass'])
        self.connect('example_turbine_size','tower_RNA.example_turbine_size')
        self.connect('rotor_diameter','tower_RNA.rotor_diameter')
        self.connect('RNA_center_of_gravity_x',['tower_RNA.RNA_center_of_gravity_x','mooring.RNA_center_of_gravity_x'])
        self.connect('RNA_center_of_gravity_y','tower_RNA.RNA_center_of_gravity_y')
        self.connect('fairlead_depth','mooring.fairlead_depth')
        self.connect('scope_ratio','mooring.scope_ratio')
        self.connect('pretension_percent','mooring.pretension_percent')
        self.connect('mooring_diameter','mooring.mooring_diameter')
        self.connect('permanent_ballast_height','mooring.permanent_ballast_height')
        self.connect('fixed_ballast_height','mooring.fixed_ballast_height')
        self.connect('spar_wall_thickness',['spar.wall_thickness','mooring.wall_thickness'])
        self.connect('spar_number_of_rings','spar.number_of_rings')
        self.connect('neutral_axis','spar.neutral_axis')
        self.connect('spar_outer_diameters',['spar.outer_diameter','mooring.spar_outer_diameter'])
        # connect outputs to inputs
        self.connect('spar.shell_mass','mooring.shell_mass')
        self.connect('spar.shell_buoyancy','mooring.shell_buoyancy')
        self.connect('spar.spar_wind_force','mooring.spar_wind_force')
        self.connect('spar.spar_wind_moment','mooring.spar_wind_moment')
        self.connect('spar.spar_current_force','mooring.spar_current_force')
        self.connect('spar.spar_current_moment','mooring.spar_current_moment')
        self.connect('spar.bulkhead_mass','mooring.bulkhead_mass')
        self.connect('spar.ring_mass','mooring.ring_mass')
        self.connect('spar.keel_cg_shell','mooring.spar_keel_to_CG')
        self.connect('spar.keel_cb_shell','mooring.spar_keel_to_CB')
        self.connect('mooring.keel_to_CG_operating_system',['spar.keel_to_CG_operating_system','tower_RNA.keel_to_CG_operating_system'])
        self.connect('mooring.fixed_ballast_mass','spar.fixed_ballast_mass')
        self.connect('mooring.permanent_ballast_mass','spar.permanent_ballast_mass')
        self.connect('mooring.variable_ballast_mass','spar.variable_ballast_mass')  
        self.connect('mooring.mooring_total_cost','spar.mooring_total_cost')
        self.connect('tower_RNA.tower_center_of_gravity','mooring.tower_center_of_gravity')
        self.connect('tower_RNA.tower_wind_force',['spar.tower_wind_force','mooring.tower_wind_force'])
        self.connect('tower_RNA.tower_wind_moment','mooring.tower_wind_moment')
        self.connect('tower_RNA.RNA_keel_to_CG','mooring.RNA_keel_to_CG')
        self.connect('tower_RNA.RNA_wind_force',['mooring.RNA_wind_force','spar.RNA_wind_force'])
        self.connect('tower_RNA.RNA_wind_moment','mooring.RNA_wind_moment')
        
        #self.connect('flange_compactness','spar.flange_compactness')
        #self.connect('web_compactness','spar.web_compactness')
        #self.connect('VAL','spar.VAL')
        #self.connect('VAG','spar.VAG')
        #self.connect('VEL','spar.VEL')
        #self.connect('VEG','spar.VEG')
        #self.connect('heel_angle','mooring.heel_angle')
        #self.connect('min_offset_unity','mooring.min_offset_unity')
        #self.connect('max_offset_unity','mooring.max_offset_unity')

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
    example = sparAssembly()
    example.wind_reference_speed = 11.
    example.wind_reference_height = 75.
    example.alpha = 0.110
    example.water_depth = 218.0
    example.significant_wave_height = 10.820 
    example.significant_wave_period =9.800
    example.number_of_sections = 4
    example.spar_lengths = [6., 12., 15., 47.]
    example.spar_end_elevation = [7., -5., -20., -67.]
    example.spar_start_elevation = [13., 7., -5., -20.]
    example.number_of_mooring_lines = 3
    example.tower_base_outer_diameter = 4.890
    example.tower_top_outer_diameter = 2.500
    example.tower_length = 60.50
    example.RNA_mass = 125000.000
    example.tower_mass = 127877.000
    example.example_turbine_size = '3MW'
    example.rotor_diameter = 101.0
    example.RNA_center_of_gravity_x = 4.1
    example.RNA_center_of_gravity_y = 1.5
    example.fairlead_depth = 13.0
    #example.permanent_ballast_height = 3.0
    #example.fixed_ballast_height = 5.0
    #example.spar_wall_thickness = [0.017,0.017,0.018,0.037]
    #example.spar_number_of_rings = [1,4,4,14]
    #example.neutral_axis = 0.285
    #example.spar_outer_diameters = [5., 6., 6., 9.]
    example.run()

if __name__ == "__main__":
    example_218WD_3MW()
    #example_218WD_6MW()
    #example_218WD_10MW()