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
    # envioroenmental factors 
    air_density = Float(1.198,iotype='in', units='kg/m**3', desc='density of air') 
    wind_reference_speed = Float(iotype='in', units='m/s', desc='reference wind speed')
    wind_reference_height = Float(iotype='in', units='m', desc='reference height')
    gust_factor = Float(1.0,iotype='in', desc='gust factor')
    alpha = Float(iotype='in', desc='power law exponent')
    water_depth = Float(iotype='in',units='m',desc='water depth')
    water_density = Float(1025,iotype='in',units='kg/m**3',desc='density of water')
    # inputs from tower_RNA compoenent
    tower_base_outer_diameter = Float(iotype='in',units='m',desc='outer diameter of tower base')
    tower_top_outer_diameter = Float(iotype='in',units='m',desc='outer diameter of tower top')
    tower_length = Float(iotype='in',units='m',desc='tower length')
    rotor_diameter = Float(iotype='in', units='m',desc='rotor diameter')
    RNA_center_of_gravity_x = Float(iotype='in', units='m',desc='rotor center of gravity') 
    RNA_center_of_gravity_y = Float(iotype='in', units='m',desc='rotor center of gravity') 
    cut_out_speed = Float(25.,iotype='in', units='m/s',desc='cut-out speed of turbine') 
    tower_mass = Float(iotype='in', units='kg',desc='tower mass') 
    RNA_mass = Float(iotype='in', units='kg',desc='RNA mass') 
    gravity = Float(9.806,iotype='in', units='m/s**2', desc='gravity')
    load_condition =  Str(iotype='in',desc='Load condition - N for normal or E for extreme')
    significant_wave_height = Float(iotype='in', units='m', desc='significant wave height')
    significant_wave_period = Float(iotype='in', units='m', desc='significant wave period')
    # spar
    spar_elevations = Array(iotype='in', units='m',desc = 'elevations of each section')
    example_turbine_size = Str(iotype='in',desc='for example cases, 3MW, 6MW, or 10 MW')
    spar_outer_diameter = Array(iotype='in',units='m',desc='top outer diameter')
    wall_thickness = Array([0.05,0.05,0.05,0.05],iotype='in', units='m',desc = 'wall thickness of each section')
    number_of_rings = Array([1,4,4,20],iotype='in',desc = 'number of stiffeners in each section')
    neutral_axis = Float(0.2,iotype='in',units='m',desc = 'neutral axis location')
    stiffener_curve_fit = Bool(iotye='in', desc='flag for using optimized stiffener dimensions or discrete stiffeners')
    stiffener_index = Int(iotype='in',desc='index of stiffener from filtered table')
    number_of_sections = Int(iotype='in',desc='number of sections in the spar')
    bulk_head = Array(iotype='in',desc = 'N for none, T for top, B for bottom')     
    material_density = Float(7850.,iotype='in', units='kg/m**3', desc='density of spar material')
    E = Float(200.e9,iotype='in', units='Pa', desc='young"s modulus of spar material')
    nu = Float(0.3,iotype='in', desc='poisson"s ratio of spar material')
    yield_stress = Float(345000000.,iotype='in', units='Pa', desc='yield stress of spar material')
    # mooring
    fairlead_depth = Float(13.0,iotype='in',units='m',desc = 'fairlead depth')
    scope_ratio = Float(1.5,iotype='in',units='m',desc = 'scope to fairlead height ratio')
    pretension_percent = Float(5.0,iotype='in',desc='Pre-Tension Percentage of MBL (match PreTension)')
    mooring_diameter = Float(0.09,iotype='in',units='m',desc='diameter of mooring chain')
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
    # costs
    straight_col_cost = Float(3490.,iotype='in',units='USD',desc='cost of straight columns in $/ton')
    tapered_col_cost = Float(4720.,iotype='in',units='USD',desc='cost of tapered columns in $/ton')
    outfitting_cost = Float(6980.,iotype='in',units='USD',desc='cost of tapered columns in $/ton')
    ballast_cost = Float(100.,iotype='in',units='USD',desc='cost of tapered columns in $/ton')

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
        self.connect('tower_base_outer_diameter','tower_RNA.base_outer_diameter')
        self.connect('tower_top_outer_diameter','tower_RNA.top_outer_diameter')
        self.connect('tower_length','tower_RNA.length')
        self.connect('air_density',['tower_RNA.air_density','spar.air_density'])
        self.connect('wind_reference_speed',['tower_RNA.wind_reference_speed','spar.wind_reference_speed'])
        self.connect('wind_reference_height',['tower_RNA.wind_reference_height','spar.wind_reference_height'])
        self.connect('gust_factor',['tower_RNA.gust_factor','spar.gust_factor'])
        self.connect('alpha',['tower_RNA.alpha','spar.alpha'])
        self.connect('spar_elevations',['tower_RNA.spar_elevations','mooring.spar_elevations','spar.elevations'])
        self.connect('example_turbine_size','tower_RNA.example_turbine_size')
        self.connect('rotor_diameter','tower_RNA.rotor_diameter')
        self.connect('RNA_center_of_gravity_x',['tower_RNA.RNA_center_of_gravity_x','spar.RNA_center_of_gravity_x'])
        self.connect('RNA_center_of_gravity_y','tower_RNA.RNA_center_of_gravity_y')
        self.connect('cut_out_speed','tower_RNA.cut_out_speed')
        self.connect('tower_mass',['tower_RNA.tower_mass','spar.tower_mass'])
        self.connect('RNA_mass',['tower_RNA.RNA_mass','spar.RNA_mass'])

        self.connect('fairlead_depth','mooring.fairlead_depth')
        self.connect('scope_ratio','mooring.scope_ratio')
        self.connect('pretension_percent','mooring.pretension_percent')
        self.connect('mooring_diameter','mooring.mooring_diameter')
        self.connect('number_of_mooring_lines','mooring.number_of_mooring_lines')
        self.connect('water_depth',['mooring.water_depth','spar.water_depth'])
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
        self.connect('spar_outer_diameter',['mooring.spar_outer_diameter','spar.outer_diameter'])
        self.connect('water_density',['mooring.water_density','spar.water_density'])

        self.connect('wall_thickness','spar.wall_thickness')
        self.connect('number_of_rings','spar.number_of_rings')
        self.connect('neutral_axis','spar.neutral_axis')
        #self.connect('stiffener_curve_fit','spar.stiffener_curve_fit')
        self.connect('stiffener_index','spar.stiffener_index')
        self.connect('number_of_sections','spar.number_of_sections')
        self.connect('bulk_head','spar.bulk_head')
        self.connect('straight_col_cost','spar.straight_col_cost')
        self.connect('tapered_col_cost','spar.tapered_col_cost')
        self.connect('outfitting_cost','spar.outfitting_cost')
        self.connect('ballast_cost','spar.ballast_cost')
        self.connect('gravity','spar.gravity')
        self.connect('load_condition','spar.load_condition')
        self.connect('significant_wave_height','spar.significant_wave_height')
        self.connect('significant_wave_period','spar.significant_wave_period')
        self.connect('material_density','spar.material_density')
        self.connect('E','spar.E')
        self.connect('nu','spar.nu')
        self.connect('yield_stress','spar.yield_stress')
        self.connect('shell_mass_factor','spar.shell_mass_factor')
        self.connect('bulkhead_mass_factor','spar.bulkhead_mass_factor')
        self.connect('ring_mass_factor','spar.ring_mass_factor')
        self.connect('outfitting_factor','spar.outfitting_factor')
        self.connect('spar_mass_factor','spar.spar_mass_factor')
        self.connect('permanent_ballast_height','spar.permanent_ballast_height')
        self.connect('fixed_ballast_height','spar.fixed_ballast_height')
        self.connect('permanent_ballast_density','spar.permanent_ballast_density')
        self.connect('fixed_ballast_density','spar.fixed_ballast_density')
        self.connect('offset_amplification_factor','spar.offset_amplification_factor')

        # connect outputs to inputs
        self.connect('tower_RNA.RNA_keel_to_CG','spar.RNA_keel_to_CG')
        self.connect('tower_RNA.tower_center_of_gravity','spar.tower_center_of_gravity')
        self.connect('tower_RNA.tower_wind_force','spar.tower_wind_force')
        self.connect('tower_RNA.RNA_wind_force','spar.RNA_wind_force')
        self.connect('mooring.mooring_total_cost','spar.mooring_total_cost')
        self.connect('mooring.mooring_keel_to_CG','spar.mooring_keel_to_CG')
        self.connect('mooring.mooring_vertical_load','spar.mooring_vertical_load')
        self.connect('mooring.mooring_horizontal_stiffness','spar.mooring_horizontal_stiffness')
        self.connect('mooring.mooring_vertical_stiffness','spar.mooring_vertical_stiffness')
        self.connect('mooring.sum_forces_x','spar.sum_forces_x')
        self.connect('mooring.offset_x','spar.offset_x')
        self.connect('mooring.damaged_mooring','spar.damaged_mooring')
        self.connect('mooring.intact_mooring','spar.intact_mooring')
        self.connect('mooring.mooring_mass','spar.mooring_mass')
        
        # objective
        self.driver.add_objective('spar.system_total_mass')
       
        # design variables
        self.driver.add_parameter('neutral_axis',low=1.,high=4.19,scaler=0.1)
        #self.driver.add_parameter('number_of_rings[0]',low=1,high=5)
        self.driver.add_parameter('number_of_rings[1]',low=1,high=10)
        self.driver.add_parameter('number_of_rings[2]',low=1,high=10)
        self.driver.add_parameter('number_of_rings[3]',low=1,high=50)
        self.driver.add_parameter('wall_thickness[0]',low=1.,high=10.,scaler=0.01)
        self.driver.add_parameter('wall_thickness[1]',low=1.,high=10.,scaler=0.01)
        self.driver.add_parameter('wall_thickness[2]',low=1.,high=10.,scaler=0.01)
        self.driver.add_parameter('wall_thickness[3]',low=1.,high=10.,scaler=0.01)
        self.driver.add_parameter('scope_ratio',low=15.,high=45.,scaler=0.1)
        self.driver.add_parameter('pretension_percent',low=2.5,high=10.)
        self.driver.add_parameter('mooring_diameter',low=30.,high=100.,scaler=0.001)
        self.driver.add_parameter('fixed_ballast_height',low=30.,high=100.,scaler=0.1)
        self.driver.add_parameter('permanent_ballast_height',low=30.,high=100.,scaler=0.1)

        # Constraints

        self.driver.add_constraint('spar.water_ballast_height < 7.5')
        self.driver.add_constraint('spar.water_ballast_height > 5.5')
        self.driver.add_constraint('spar.flange_compactness < 1.')
        self.driver.add_constraint('spar.web_compactness < 1.')

        self.driver.add_constraint('spar.VAL[0] < 0.99')
        self.driver.add_constraint('spar.VAL[1] < 0.99')
        self.driver.add_constraint('spar.VAL[2] < 0.99')
        self.driver.add_constraint('spar.VAL[3] < 0.99')

        self.driver.add_constraint('spar.VAG[0] < 0.99')
        self.driver.add_constraint('spar.VAG[1] < 0.99')
        self.driver.add_constraint('spar.VAG[2] < 0.99')
        self.driver.add_constraint('spar.VAG[3] < 0.99')

        self.driver.add_constraint('spar.VEL[0] < 0.99')
        self.driver.add_constraint('spar.VEL[1] < 0.99')
        self.driver.add_constraint('spar.VEL[2] < 0.99')
        self.driver.add_constraint('spar.VEL[3] < 0.99')

        self.driver.add_constraint('spar.VEG[0] < 0.99')
        self.driver.add_constraint('spar.VEG[1] < 0.99')
        self.driver.add_constraint('spar.VEG[2] < 0.99')
        self.driver.add_constraint('spar.VEG[3] < 0.99')
        self.driver.add_constraint('spar.platform_stability_check < 1.')
        self.driver.add_constraint('spar.heel_angle <= 6.')
        self.driver.add_constraint('spar.min_offset_unity < 1.0')
        self.driver.add_constraint('spar.max_offset_unity < 1.0')

def sys_print(example):
    print 'scope ratio: ',example.scope_ratio
    print 'pretension percent: ',example.pretension_percent
    print 'mooring diameter: ',example.mooring_diameter
    print 'PBH: ', example.permanent_ballast_height
    print 'FBH: ', example.fixed_ballast_height
    print 'YNA: ',example.spar.neutral_axis
    print 'number of stiffeners: ',example.number_of_rings
    print 'wall thickness: ',example.wall_thickness
    print 'spar outer diameters', example.spar.outer_diameter
    print '-------------------------------'
    print 'WBH: ', example.spar.water_ballast_height
    print 'heel angle: ',example.spar.heel_angle
    print 'min offset unity: ',example.spar.min_offset_unity
    print 'max offset unity: ',example.spar.max_offset_unity 
    print 'VAL: ',example.spar.VAL
    print 'VAG: ',example.spar.VAG
    print 'VEL: ',example.spar.VEL
    print 'VEG: ',example.spar.VEG
    print 'web compactness: ',example.spar.web_compactness
    print 'flange compactness: ',example.spar.flange_compactness
    print '-------------------------------'
    print 'spar mass: ', example.spar.spar_mass
    print 'ballast mass: ', example.spar.ballast_mass
    print 'mooring mass: ', example.mooring.mooring_mass
    print 'total mass: ', example.spar.system_total_mass

def example_218WD_3MW():
    example = sparAssembly()
    example.tower_base_outer_diameter = 4.890
    example.tower_top_outer_diameter = 2.5
    example.tower_length = 60.5
    example.tower_mass =  127877.
    example.wind_reference_speed = 11.
    example.wind_reference_height = 75.
    example.alpha = 0.110
    example.spar_elevations = [13.,7.,-5.,-20.,-67.]
    example.example_turbine_size = '3MW'
    example.rotor_diameter = 101.0
    example.RNA_mass = 125000.
    example.RNA_center_of_gravity_x = 4.1
    example.RNA_center_of_gravity_y = 1.5
    example.fairlead_depth = 13. 
    example.scope_ratio = 1.5
    example.pretension_percent = 5.
    example.mooring_diameter = 0.09
    example.number_of_mooring_lines = 3
    example.water_depth = 218.
    example.mooring_type = 'CHAIN'
    example.anchor_type =  'PILE'
    example.fairlead_offset_from_shell = 0.5
    example.spar_outer_diameter= [5.000,6.000,6.000,9.000]
    #example.wall_thickness=[0.05,0.05,0.05,0.05]
    example.spar.stiffener_curve_fit = False
    example.stiffener_index = 232
    #example.spar.stiffener_curve_fit = True
    example.permanent_ballast_height = 3.
    example.fixed_ballast_height = 5.
    example.wall_thickness=[0.05,0.05,0.05,0.05]
    #example.neutral_axis = 0.272
    example.number_of_rings = [1,4,4,14]
    example.number_of_sections = 4
    example.bulk_head = ['N', 'T', 'N', 'B']
    example.load_condition = 'N'
    example.significant_wave_height = 10.820
    example.significant_wave_period = 9.800
    example.run()
    print '----------218WD_3MW------------'
    sys_print(example)

def example_218WD_6MW():
    example = sparAssembly()
    example.tower_base_outer_diameter = 6.0
    example.tower_top_outer_diameter = 3.51
    example.tower_length = 80.5
    example.tower_mass =  366952.000
    example.wind_reference_speed = 11.
    example.wind_reference_height = 97.
    example.alpha = 0.110
    #example.spar_lengths = [6.,12.,15.,52.]
    example.spar_elevations = [13.,7.,-5.,-20.,-72.]
    example.example_turbine_size = '6MW'
    example.rotor_diameter = 154.
    example.RNA_mass = 365500.000
    example.RNA_center_of_gravity_x = 5.750
    example.RNA_center_of_gravity_y = 3.5
    example.fairlead_depth = 13.
    example.scope_ratio = 1.5
    example.pretension_percent = 5.0
    example.mooring_diameter = 0.090
    example.number_of_mooring_lines = 3
    example.water_depth = 218.
    example.mooring_type = 'CHAIN'
    example.anchor_type =  'PILE'
    example.fairlead_offset_from_shell = 0.5
    example.spar_outer_diameter= [7.000,8.000,8.000,13.000]
    example.wall_thickness=[0.05,0.05,0.05,0.05]
    example.spar.stiffener_curve_fit = False
    example.stiffener_index = 271
    example.fixed_ballast_height = 7.0
    example.permanent_ballast_height = 3.0
    #example.wall_thickness=[0.0263,0.0251,0.0262,0.038]
    #example.neutral_axis = 0.27
    example.number_of_rings = [1,4,4,19]
    example.scope_ratio = 1.5
    example.pretension_percent = 6.5
    example.mooring_diameter = 0.075
    example.number_of_sections = 4
    example.bulk_head = ['N', 'T', 'N', 'B']
    example.load_condition = 'N'
    example.significant_wave_height = 10.820
    example.significant_wave_period = 9.800
    example.run()
    print '----------218WD_6MW------------'
    sys_print(example)

def example_218WD_10MW():
    example = sparAssembly()
    example.tower_base_outer_diameter = 7.72
    example.tower_top_outer_diameter = 4.050
    example.tower_length = 102.630
    example.tower_mass =  698235.000
    example.wind_reference_speed = 11.
    example.wind_reference_height = 119.
    example.alpha = 0.110
    #example.spar_lengths = [6.,12.,15.,72.]
    example.spar_elevations = [13.,7.,-5.,-20.,-92.]
    example.example_turbine_size = '10MW'
    example.rotor_diameter = 194.
    example.RNA_mass = 677000.000
    example.RNA_center_of_gravity_x = 7.07
    example.RNA_center_of_gravity_y = 3.370
    example.fairlead_depth = 13.
    example.scope_ratio = 1.5
    example.pretension_percent = 5.0
    example.mooring_diameter = 0.090
    example.number_of_mooring_lines = 3
    example.water_depth = 218.
    example.mooring_type = 'CHAIN'
    example.anchor_type =  'PILE'
    example.fairlead_offset_from_shell = 0.5
    example.spar_outer_diameter= [8.0,9.0,9.0,15.0]
    example.wall_thickness=[0.05,0.05,0.05,0.05]
    example.spar.stiffener_curve_fit = False
    example.stiffener_index = 282
    example.fixed_ballast_height = 9.0
    example.permanent_ballast_height = 4.0
    # note: these are slightly off from excel: ie the stiffener AR, which resulted in diff in RGM 
    #example.wall_thickness=[0.0366,0.035,0.035,0.059]
    #example.neutral_axis = 0.38
    example.number_of_rings = [1,4,4,40]
    example.number_of_sections = 4
    example.bulk_head = ['N', 'T', 'N', 'B']
    example.load_condition = 'N'
    example.significant_wave_height = 10.820
    example.significant_wave_period = 9.800
    example.run()
    print '----------218WD_10MW------------'
    sys_print(example)

if __name__ == "__main__":
    #example_218WD_3MW()
    example_218WD_6MW()
    #example_218WD_10MW()
    