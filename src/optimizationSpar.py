from openmdao.main.api import Component, Assembly, convert_units
from openmdao.main.datatypes.api import Float, Array, Enum, Str, Int, Bool
from openmdao.lib.drivers.api import COBYLAdriver,SLSQPdriver
from spar import Spar
#from spar_discrete import spar_discrete
import numpy as np
import time
from spar_utils import filtered_stiffeners_table

class optimizationSpar(Assembly):
    # variables 
    wall_thickness = Array(iotype='in', units='m',desc = 'wall thickness of each section')
    number_of_rings = Array(iotype='in',desc = 'number of stiffeners in each section')
    neutral_axis = Float(iotype='in',units='m',desc = 'neutral axis location')
    # inputs 
    initial_pass = Bool(True, iotype='in', desc='flag for using optimized stiffener dimensions or discrete stiffeners')
    stiffener_index = Int(iotype='in',desc='index of stiffener from filtered table')
    gravity = Float(9.806,iotype='in', units='m/s**2', desc='gravity')
    air_density = Float(1.198,iotype='in', units='kg/m**3', desc='density of air')
    water_density = Float(1025,iotype='in',units='kg/m**3',desc='density of water')
    water_depth = Float(iotype='in', units='m', desc='water depth')
    load_condition =  Str('N',iotype='in',desc='Load condition - N for normal or E for extreme')
    significant_wave_height = Float(iotype='in', units='m', desc='significant wave height')
    significant_wave_period = Float(iotype='in', units='m', desc='significant wave period')
    keel_cg_mooring = Float(iotype='in', units='m', desc='center of gravity above keel of mooring line')
    keel_cg_operating_system = Float(iotype='in', units='m', desc='center of gravity above keel of operating system')
    reference_wind_speed = Float(iotype='in', units='m/s', desc='reference wind speed')
    reference_height = Float(iotype='in', units='m', desc='reference height')
    alpha = Float(iotype='in', desc='power law exponent')
    material_density = Float(7850.,iotype='in', units='kg/m**3', desc='density of spar material')
    E = Float(200.e9,iotype='in', units='Pa', desc='young"s modulus of spar material')
    nu = Float(0.3,iotype='in', desc='poisson"s ratio of spar material')
    yield_stress = Float(345000000.,iotype='in', units='Pa', desc='yield stress of spar material')
    rotor_mass = Float(iotype='in', units='kg', desc='rotor mass')
    tower_mass = Float(iotype='in', units='kg', desc='tower mass')
    free_board = Float(iotype='in', units='m', desc='free board length')
    draft = Float(iotype='in', units='m', desc='draft length')
    fixed_ballast_mass = Float(iotype='in', units='kg', desc='fixed ballast mass')
    hull_mass = Float(iotype='in', units='kg', desc='hull mass')
    permanent_ballast_mass = Float(iotype='in', units='kg', desc='permanent ballast mass')
    variable_ballast_mass = Float(iotype='in', units='kg', desc='variable ballast mass')
    number_of_sections = Int(4,iotype='in',desc='number of sections in the spar')
    outer_diameter = Array(iotype='in', units='m',desc = 'outer diameter of each section')
    length = Array(iotype='in', units='m',desc = 'wlength of each section')
    end_elevation = Array(iotype='in', units='m',desc = 'end elevation of each section')
    start_elevation = Array(iotype='in', units='m',desc = 'start elevation of each section')
    bulk_head = Array(iotype='in',desc = 'N for none, T for top, B for bottom')
    #system_acceleration = Float(1.2931,iotype='in', units='m/s**2', desc='acceleration')
    tower_base_OD = Float(iotype='in', units='m', desc='outer diameter of tower base')
    tower_top_OD = Float(iotype='in', units='m', desc='outer diameter of tower top')
    tower_length = Float(iotype='in', units='m', desc='length of tower')
    gust_factor = Float(iotype='in', desc='gust factor')
    cut_out_speed = Float(iotype='in', units='m/s',desc='cut out speed of turbine')
    turbine_size = Str(iotype='in',desc='for example cases, 3MW, 6MW, or 10 MW')
    rotor_diameter = Float(iotype='in', units='m',desc='rotor diameter')
    def configure(self):
        self.add('driver',COBYLAdriver())
        self.driver.maxfun = 10000
    
        self.add('spar',Spar())
        self.driver.workflow.add('spar')
        # connect inputs
        self.connect('initial_pass','spar.initial_pass')
        self.connect('stiffener_index','spar.stiffener_index')
        self.connect('gravity','spar.gravity')
        self.connect('air_density','spar.air_density')
        self.connect('water_density','spar.water_density')
        self.connect('load_condition','spar.load_condition')
        self.connect('significant_wave_height','spar.significant_wave_height')
        self.connect('significant_wave_period','spar.significant_wave_period')
        self.connect('keel_cg_mooring','spar.keel_cg_mooring')
        self.connect('keel_cg_operating_system','spar.keel_cg_operating_system')
        self.connect('reference_wind_speed','spar.reference_wind_speed')
        self.connect('reference_height','spar.reference_height')
        self.connect('alpha','spar.alpha')
        self.connect('material_density','spar.material_density')
        self.connect('E','spar.E')
        self.connect('nu','spar.nu')
        self.connect('yield_stress','spar.yield_stress')
        self.connect('rotor_mass','spar.rotor_mass')
        self.connect('tower_mass','spar.tower_mass')
        self.connect('free_board','spar.free_board')
        self.connect('draft','spar.draft')
        self.connect('fixed_ballast_mass','spar.fixed_ballast_mass')
        self.connect('hull_mass','spar.hull_mass')
        self.connect('permanent_ballast_mass','spar.permanent_ballast_mass')
        self.connect('variable_ballast_mass','spar.variable_ballast_mass')
        self.connect('number_of_sections','spar.number_of_sections')
        self.connect('outer_diameter','spar.outer_diameter')
        self.connect('length','spar.length')
        self.connect('end_elevation','spar.end_elevation')
        self.connect('start_elevation','spar.start_elevation')
        #self.connect('system_acceleration','spar.system_acceleration')
        self.connect('bulk_head','spar.bulk_head')
        self.connect('water_depth','spar.water_depth')
        self.connect('tower_base_OD','spar.tower_base_OD')
        self.connect('tower_top_OD','spar.tower_top_OD')
        self.connect('tower_length','spar.tower_length')
        self.connect('gust_factor','spar.gust_factor')
        self.connect('cut_out_speed','spar.cut_out_speed')
        self.connect('turbine_size','spar.turbine_size')
        self.connect('rotor_diameter','spar.rotor_diameter')
        # objective
        self.driver.add_objective('spar.shell_ring_bulkhead_mass')
        # design variables
        

        self.driver.add_parameter('spar.neutral_axis',low=10.,high=500.,scaler=0.01)
        #self.driver.add_parameter('spar.wall_thickness',low=[0.01,0.01,0.01,0.01],high=[0.06,0.06,0.06,0.06])

        self.driver.add_parameter('spar.wall_thickness[0]',low=10.,high=60.,scaler=0.001)
        self.driver.add_parameter('spar.wall_thickness[1]',low=10.,high=60.,scaler=0.001)
        self.driver.add_parameter('spar.wall_thickness[2]',low=10.,high=60.,scaler=0.001)
        self.driver.add_parameter('spar.wall_thickness[3]',low=10.,high=60.,scaler=0.001)
        #self.driver.add_parameter('spar.number_of_rings',low=[1,1,1,1],high=[5,10,10,50])
        self.driver.add_parameter('spar.number_of_rings[0]',low=1,high=5)
        self.driver.add_parameter('spar.number_of_rings[1]',low=1,high=10)
        self.driver.add_parameter('spar.number_of_rings[2]',low=1,high=10)
        self.driver.add_parameter('spar.number_of_rings[3]',low=1,high=50)
        
        # Constraints
        self.driver.add_constraint('spar.flange_compactness <= 1')
        self.driver.add_constraint('spar.web_compactness <= 1')

        self.driver.add_constraint('spar.VAL[0] <= 1')
        self.driver.add_constraint('spar.VAL[1] <= 1.')
        self.driver.add_constraint('spar.VAL[2] <= 1.')
        self.driver.add_constraint('spar.VAL[3] <= 1')

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


def sys_print(example):
    filteredStiffeners = filtered_stiffeners_table()
    print 'number of stiffeners: ',example.number_of_rings
    print 'wall thickness: ',example.wall_thickness
    print 'VAL: ',example.VAL
    print 'VAG: ',example.VAG
    print 'VEL: ',example.VEL
    print 'VEG: ',example.VEG
    print 'web compactness: ',example.web_compactness
    print 'flange compactness: ',example.flange_compactness
    print 'stiffener: ', filteredStiffeners[example.stiffener_index]
    print 'shell+ring+bulkhead mass: ',example.shell_ring_bulkhead_mass
    
def example_130WD_3MW():
    example = optimizationSpar()
    example.water_depth = 130.
    example.load_condition = 'N'
    example.significant_wave_height = 10.660
    example.significant_wave_period = 13.210
    example.keel_cg_mooring = 51.
    example.keel_cg_operating_system = 20.312
    example.reference_wind_speed = 11.
    example.reference_height = 75.
    example.alpha = 0.110
    example.material_density = 7850.
    example.E = 200.e9
    example.nu = 0.3
    example.yield_stress = 345000000.
    example.rotor_mass = 125000.
    example.tower_mass = 83705.
    example.free_board = 13.
    example.draft = 64.
    example.fixed_ballast_mass = 1244227.77
    example.hull_mass = 890985.086
    example.permanent_ballast_mass = 838450.256
    example.variable_ballast_mass = 418535.462
    example.number_of_sections = 4
    example.outer_diameter = [5., 6., 6., 9.]
    example.length = [6., 12., 15., 44.]
    example.end_elevation = [7., -5., -20., -64.]
    example.start_elevation = [13., 7., -5., -20.]
    example.bulk_head = ['N', 'T', 'N', 'B']
    #example.system_acceleration=1.2931
    example.run()

    yna = convert_units(example.spar.neutral_axis ,'m','inch')
    filteredStiffeners = filtered_stiffeners_table()
    for i in range (0,len(filteredStiffeners)-1):
        stiffener_bef = filteredStiffeners[i]
        stiffener_aft = filteredStiffeners[i+1]
        if yna > stiffener_bef[6] and yna<stiffener_aft[6]:
            opt_index = i+1
    second_fit = Spar()
    second_fit.wall_thickness = example.spar.wall_thickness
    second_fit.number_of_rings = example.spar.number_of_rings
    second_fit.stiffener_index = opt_index
    second_fit.initial_pass = False

    second_fit.water_depth =  example.water_depth
    second_fit.load_condition = example.load_condition
    second_fit.significant_wave_height =  example.significant_wave_height
    second_fit.significant_wave_period = example.significant_wave_period
    second_fit.keel_cg_mooring = example.keel_cg_mooring
    second_fit.keel_cg_operating_system = example.keel_cg_operating_system
    second_fit.reference_wind_speed = example.reference_wind_speed
    second_fit.reference_height = example.reference_height
    second_fit.alpha = example.alpha
    second_fit.material_density = example.material_density
    second_fit.E = example.E
    second_fit.nu =example.nu
    second_fit.yield_stress = example.yield_stress
    second_fit.rotor_mass = example.rotor_mass
    second_fit.tower_mass = example.tower_mass
    second_fit.free_board = example.free_board
    second_fit.draft = example.draft
    second_fit.fixed_ballast_mass = example.fixed_ballast_mass
    second_fit.hull_mass = example.hull_mass
    second_fit.permanent_ballast_mass = example.permanent_ballast_mass
    second_fit.variable_ballast_mass = example.variable_ballast_mass
    second_fit.number_of_sections = example.number_of_sections
    second_fit.outer_diameter = example.outer_diameter
    second_fit.length = example.length
    second_fit.end_elevation = example.end_elevation
    second_fit.start_elevation = example.start_elevation
    second_fit.bulk_head = example.bulk_head
    second_fit.system_acceleration=example.system_acceleration
    second_fit.run()
    index = opt_index
    unity = max(max(second_fit.VAL),max(second_fit.VAG),max(second_fit.VEL),max(second_fit.VEG))   
    while ((unity-1.0) > 1e-7):
        if index <124:
            index += 1
            second_fit.stiffener_index = index
            second_fit.run()
            unity = max(max(second_fit.VAL),max(second_fit.VAG),max(second_fit.VEL),max(second_fit.VEG)) 
        else:
            second_fit.stiffener_index = opt_index
            for i in range(0,second_fit.number_of_sections):
                if second_fit.VAL[i] >1. or second_fit.VAG[i]>1. or second_fit.VEL[i]>1. or second_fit.VEG[i]>1.:    
                    second_fit.number_of_rings[i] += 1 
                    second_fit.run()
                    unity = max(max(second_fit.VAL),max(second_fit.VAG),max(second_fit.VEL),max(second_fit.VEG)) 
    print '--------------example_130WD_3MW------------------'
    print "Elapsed time: ", time.time()-tt, " seconds"
    sys_print(second_fit)

def example_218WD_3MW():
    example = optimizationSpar()
    tt = time.time()
    example.water_depth = 218.
    example.load_condition = 'N'
    example.significant_wave_height = 10.820
    example.significant_wave_period = 9.800
    example.keel_cg_mooring = 35.861
    example.keel_cg_operating_system = 20.019
    example.reference_wind_speed = 11.
    example.reference_height = 75.
    example.alpha = 0.110
    example.material_density = 7850.
    example.E = 200.e9
    example.nu = 0.3
    example.yield_stress = 345000000.
    example.rotor_mass = 125000.
    example.tower_mass = 127877.
    example.free_board = 13.
    example.draft = 67.
    example.fixed_ballast_mass = 1244227.77
    example.hull_mass = 890985.086
    example.permanent_ballast_mass = 838450.256
    example.variable_ballast_mass = 484661.874
    example.number_of_sections = 4
    example.outer_diameter = [5., 6., 6., 9.]
    example.length = [6., 12., 15., 47.]
    example.end_elevation = [7., -5., -20., -67.]
    example.start_elevation = [13., 7., -5., -20.]
    example.bulk_head = ['N', 'T', 'N', 'B']
    #example.system_acceleration = 1.27894072011959
    example.gust_factor = 1.0
    example.tower_base_OD = 4.890
    example.tower_top_OD = 2.5
    example.tower_length = 60.5
    example.cut_out_speed = 25.
    example.turbine_size = '3MW'
    example.rotor_diameter = 101.0
    example.run()
    yna = convert_units(example.spar.neutral_axis ,'m','inch')
    filteredStiffeners = filtered_stiffeners_table()
    for i in range (0,len(filteredStiffeners)-1):
        stiffener_bef = filteredStiffeners[i]
        stiffener_aft = filteredStiffeners[i+1]
        if yna > stiffener_bef[6] and yna<stiffener_aft[6]:
            opt_index = i+1
    second_fit = Spar()
    second_fit.wall_thickness = example.spar.wall_thickness
    second_fit.number_of_rings = example.spar.number_of_rings
    second_fit.stiffener_index = opt_index
    second_fit.initial_pass = False

    second_fit.water_depth =  example.water_depth
    second_fit.load_condition = example.load_condition
    second_fit.significant_wave_height =  example.significant_wave_height
    second_fit.significant_wave_period = example.significant_wave_period
    second_fit.keel_cg_mooring = example.keel_cg_mooring
    second_fit.keel_cg_operating_system = example.keel_cg_operating_system
    second_fit.reference_wind_speed = example.reference_wind_speed
    second_fit.reference_height = example.reference_height
    second_fit.alpha = example.alpha
    second_fit.material_density = example.material_density
    second_fit.E = example.E
    second_fit.nu =example.nu
    second_fit.yield_stress = example.yield_stress
    second_fit.rotor_mass = example.rotor_mass
    second_fit.tower_mass = example.tower_mass
    second_fit.free_board = example.free_board
    second_fit.draft = example.draft
    second_fit.fixed_ballast_mass = example.fixed_ballast_mass
    second_fit.hull_mass = example.hull_mass
    second_fit.permanent_ballast_mass = example.permanent_ballast_mass
    second_fit.variable_ballast_mass = example.variable_ballast_mass
    second_fit.number_of_sections = example.number_of_sections
    second_fit.outer_diameter = example.outer_diameter
    second_fit.length = example.length
    second_fit.end_elevation = example.end_elevation
    second_fit.start_elevation = example.start_elevation
    second_fit.bulk_head = example.bulk_head
    #second_fit.system_acceleration=example.system_acceleration
    second_fit.gust_factor = example.gust_factor
    second_fit.tower_base_OD = example.tower_base_OD
    second_fit.tower_top_OD = example.tower_top_OD 
    second_fit.tower_length = example.tower_length 
    second_fit.cut_out_speed = example.cut_out_speed
    second_fit.turbine_size = example.turbine_size
    second_fit.rotor_diameter = example.rotor_diameter
    second_fit.run()
    index = opt_index
    unity = max(max(second_fit.VAL),max(second_fit.VAG),max(second_fit.VEL),max(second_fit.VEG))   
    while ((unity-1.0) > 1e-7):
        if index <124:
            index += 1
            second_fit.stiffener_index = index
            second_fit.run()
            unity = max(max(second_fit.VAL),max(second_fit.VAG),max(second_fit.VEL),max(second_fit.VEG)) 
        else:
            second_fit.stiffener_index = opt_index
            for i in range(0,second_fit.number_of_sections):
                if second_fit.VAL[i] >1. or second_fit.VAG[i]>1. or second_fit.VEL[i]>1. or second_fit.VEG[i]>1.:    
                    second_fit.number_of_rings[i] += 1 
                    second_fit.run()
                    unity = max(max(second_fit.VAL),max(second_fit.VAG),max(second_fit.VEL),max(second_fit.VEG)) 
    print '--------------example_218WD_3MW------------------'
    print "Elapsed time: ", time.time()-tt, " seconds"
    sys_print(second_fit)

def example_218WD_6MW():
    example = optimizationSpar()
    tt = time.time()
    example.water_depth = 218.
    example.load_condition = 'N'
    example.significant_wave_height = 10.820
    example.significant_wave_period = 9.800
    example.keel_cg_mooring = 37.177
    example.keel_cg_operating_system = 24.059
    example.reference_wind_speed = 11.
    example.reference_height = 97.
    example.alpha = 0.110
    example.material_density = 7850.
    example.E = 200.e9
    example.nu = 0.3
    example.yield_stress = 345000000.
    example.rotor_mass = 365500.000
    example.tower_mass = 366952.000
    example.free_board = 13.
    example.draft = 72.
    example.fixed_ballast_mass = 3659547.034
    example.hull_mass = 1593822.041
    example.permanent_ballast_mass = 1761475.914
    example.variable_ballast_mass = 820790.246
    example.number_of_sections = 4
    example.outer_diameter = [7., 8., 8., 13.]
    example.length = [6., 12., 15., 52.]
    example.end_elevation = [7., -5., -20., -72.]
    example.start_elevation = [13., 7., -5., -20.]
    example.bulk_head = ['N', 'T', 'N', 'B']
    #example.system_acceleration = 1.12124749328663
    example.gust_factor = 1.0
    example.tower_base_OD = 6.0
    example.tower_top_OD = 3.51
    example.tower_length = 80.5
    example.cut_out_speed = 25.
    example.turbine_size = '6MW'
    example.rotor_diameter = 154.0

    example.run()
    yna = convert_units(example.spar.neutral_axis ,'m','inch')
    filteredStiffeners = filtered_stiffeners_table()
    for i in range (0,len(filteredStiffeners)-1):
        stiffener_bef = filteredStiffeners[i]
        stiffener_aft = filteredStiffeners[i+1]
        if yna > stiffener_bef[6] and yna<stiffener_aft[6]:
            opt_index = i+1
    second_fit = Spar()
    second_fit.wall_thickness = example.spar.wall_thickness
    second_fit.number_of_rings = example.spar.number_of_rings
    second_fit.stiffener_index = opt_index
    second_fit.initial_pass = False

    second_fit.water_depth =  example.water_depth
    second_fit.load_condition = example.load_condition
    second_fit.significant_wave_height =  example.significant_wave_height
    second_fit.significant_wave_period = example.significant_wave_period
    second_fit.keel_cg_mooring = example.keel_cg_mooring
    second_fit.keel_cg_operating_system = example.keel_cg_operating_system
    second_fit.reference_wind_speed = example.reference_wind_speed
    second_fit.reference_height = example.reference_height
    second_fit.alpha = example.alpha
    second_fit.material_density = example.material_density
    second_fit.E = example.E
    second_fit.nu =example.nu
    second_fit.yield_stress = example.yield_stress
    second_fit.rotor_mass = example.rotor_mass
    second_fit.tower_mass = example.tower_mass
    second_fit.free_board = example.free_board
    second_fit.draft = example.draft
    second_fit.fixed_ballast_mass = example.fixed_ballast_mass
    second_fit.hull_mass = example.hull_mass
    second_fit.permanent_ballast_mass = example.permanent_ballast_mass
    second_fit.variable_ballast_mass = example.variable_ballast_mass
    second_fit.number_of_sections = example.number_of_sections
    second_fit.outer_diameter = example.outer_diameter
    second_fit.length = example.length
    second_fit.end_elevation = example.end_elevation
    second_fit.start_elevation = example.start_elevation
    second_fit.bulk_head = example.bulk_head
    second_fit.gust_factor = example.gust_factor
    second_fit.tower_base_OD = example.tower_base_OD
    second_fit.tower_top_OD = example.tower_top_OD 
    second_fit.tower_length = example.tower_length 
    second_fit.cut_out_speed = example.cut_out_speed
    second_fit.turbine_size = example.turbine_size
    second_fit.rotor_diameter = example.rotor_diameter
    #second_fit.system_acceleration=example.system_acceleration
    second_fit.run()
    index = opt_index
    unity = max(max(second_fit.VAL),max(second_fit.VAG),max(second_fit.VEL),max(second_fit.VEG))   
    while ((unity-1.0) > 1e-7):
        if index <124:
            index += 1
            second_fit.stiffener_index = index
            second_fit.run()
            unity = max(max(second_fit.VAL),max(second_fit.VAG),max(second_fit.VEL),max(second_fit.VEG)) 
        else:
            second_fit.stiffener_index = opt_index
            for i in range(0,second_fit.number_of_sections):
                if second_fit.VAL[i] >1. or second_fit.VAG[i]>1. or second_fit.VEL[i]>1. or second_fit.VEG[i]>1.:    
                    second_fit.number_of_rings[i] += 1 
                    second_fit.run()
                    unity = max(max(second_fit.VAL),max(second_fit.VAG),max(second_fit.VEL),max(second_fit.VEG)) 
    print '--------------example_218WD_6MW------------------'
    print "Elapsed time: ", time.time()-tt, " seconds"
    sys_print(second_fit)



def example_218WD_10MW():
    example = optimizationSpar()
    tt = time.time()
    example.water_depth = 218.
    example.load_condition = 'N'
    example.significant_wave_height = 10.820
    example.significant_wave_period = 9.800
    example.keel_cg_mooring = 45.
    example.keel_cg_operating_system = 30.730
    example.reference_wind_speed = 11.
    example.reference_height = 119.
    example.alpha = 0.110
    example.material_density = 7850.
    example.E = 200.e9
    example.nu = 0.3
    example.yield_stress = 345000000.
    example.rotor_mass = 677000.000
    example.tower_mass = 698235.000
    example.free_board = 13.
    example.draft = 92.
    example.fixed_ballast_mass = 6276669.794
    example.hull_mass = 2816863.293
    example.permanent_ballast_mass = 3133090.391
    example.variable_ballast_mass = 1420179.260
    example.number_of_sections = 4
    example.outer_diameter = [8.,9.,9.,15.]
    example.length = [6., 12., 15., 72.]
    example.end_elevation = [7., -5., -20., -92.]
    example.start_elevation = [13., 7., -5., -20.]
    example.bulk_head = ['N', 'T', 'N', 'B']
    #example.system_acceleration = 0.856545480516845
    example.gust_factor = 1.0
    example.tower_base_OD = 7.720
    example.tower_top_OD = 4.050
    example.tower_length = 102.63
    example.cut_out_speed = 25.
    example.turbine_size = '10MW'
    example.rotor_diameter = 194.0
    example.run()
    yna = convert_units(example.spar.neutral_axis ,'m','inch')
    filteredStiffeners = filtered_stiffeners_table()
    for i in range (0,len(filteredStiffeners)-1):
        stiffener_bef = filteredStiffeners[i]
        stiffener_aft = filteredStiffeners[i+1]
        if yna > stiffener_bef[6] and yna<stiffener_aft[6]:
            opt_index = i+1
    second_fit = Spar()
    second_fit.wall_thickness = example.spar.wall_thickness
    second_fit.number_of_rings = example.spar.number_of_rings
    second_fit.stiffener_index = opt_index
    second_fit.initial_pass = False

    second_fit.water_depth =  example.water_depth
    second_fit.load_condition = example.load_condition
    second_fit.significant_wave_height =  example.significant_wave_height
    second_fit.significant_wave_period = example.significant_wave_period
    second_fit.keel_cg_mooring = example.keel_cg_mooring
    second_fit.keel_cg_operating_system = example.keel_cg_operating_system
    second_fit.reference_wind_speed = example.reference_wind_speed
    second_fit.reference_height = example.reference_height
    second_fit.alpha = example.alpha
    second_fit.material_density = example.material_density
    second_fit.E = example.E
    second_fit.nu =example.nu
    second_fit.yield_stress = example.yield_stress
    second_fit.rotor_mass = example.rotor_mass
    second_fit.tower_mass = example.tower_mass
    second_fit.free_board = example.free_board
    second_fit.draft = example.draft
    second_fit.fixed_ballast_mass = example.fixed_ballast_mass
    second_fit.hull_mass = example.hull_mass
    second_fit.permanent_ballast_mass = example.permanent_ballast_mass
    second_fit.variable_ballast_mass = example.variable_ballast_mass
    second_fit.number_of_sections = example.number_of_sections
    second_fit.outer_diameter = example.outer_diameter
    second_fit.length = example.length
    second_fit.end_elevation = example.end_elevation
    second_fit.start_elevation = example.start_elevation
    second_fit.bulk_head = example.bulk_head
    #second_fit.system_acceleration=example.system_acceleration
    second_fit.gust_factor = example.gust_factor
    second_fit.tower_base_OD = example.tower_base_OD
    second_fit.tower_top_OD = example.tower_top_OD 
    second_fit.tower_length = example.tower_length 
    second_fit.cut_out_speed = example.cut_out_speed
    second_fit.turbine_size = example.turbine_size
    second_fit.rotor_diameter = example.rotor_diameter
    second_fit.run()
    index = opt_index
    unity = max(max(second_fit.VAL),max(second_fit.VAG),max(second_fit.VEL),max(second_fit.VEG))   
    while ((unity-1.0) > 1e-7):
        if index <124:
            index += 1
            second_fit.stiffener_index = index
            second_fit.run()
            unity = max(max(second_fit.VAL),max(second_fit.VAG),max(second_fit.VEL),max(second_fit.VEG)) 
            #print unity-1.0
        else:
            second_fit.stiffener_index = opt_index
            for i in range(0,second_fit.number_of_sections):
                if second_fit.VAL[i] >1. or second_fit.VAG[i]>1. or second_fit.VEL[i]>1. or second_fit.VEG[i]>1.:    
                    second_fit.number_of_rings[i] += 1 
                    second_fit.run()
                    unity = max(max(second_fit.VAL),max(second_fit.VAG),max(second_fit.VEL),max(second_fit.VEG)) 
                    #print second_fit.number_of_rings
    print '--------------example_218WD_10MW------------------'
    print "Elapsed time: ", time.time()-tt, " seconds"
    sys_print(second_fit)

if __name__ == "__main__":
    #example_130WD_3MW()
    #example_218WD_3MW()
    #example_218WD_6MW()
    example_218WD_10MW()