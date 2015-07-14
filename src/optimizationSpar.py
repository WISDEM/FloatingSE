from openmdao.main.api import Component, Assembly, convert_units
from openmdao.main.datatypes.api import Float, Array, Enum, Str, Int, Bool
from openmdao.lib.drivers.api import COBYLAdriver, SLSQPdriver
from spar import Spar
#from spar_discrete import spar_discrete
import numpy as np
import time
from spar_utils import filtered_stiffeners_table, full_stiffeners_table

class optimizationSpar(Assembly):
    # variables 
    first_fit = Bool(True,iotype='in',desc = 'wall thickness of each section')

    def configure(self):
        self.add('driver',COBYLAdriver())
        self.driver.maxfun = 100000
        self.add('spar',Spar())
        self.driver.workflow.add('spar')
        
        # objective
        self.driver.add_objective('spar.shell_ring_bulkhead_mass')
       
        # design variables
        self.driver.add_parameter('spar.neutral_axis',low=100.,high=419.,scaler=0.001)
        self.driver.add_parameter('spar.number_of_rings[0]',low=1,high=5)
        self.driver.add_parameter('spar.number_of_rings[1]',low=1,high=10)
        self.driver.add_parameter('spar.number_of_rings[2]',low=1,high=10)
        self.driver.add_parameter('spar.number_of_rings[3]',low=1,high=50)
        self.driver.add_parameter('spar.wall_thickness[0]',low=100.,high=1000.,scaler=0.0001)
        self.driver.add_parameter('spar.wall_thickness[1]',low=100.,high=1000.,scaler=0.0001)
        self.driver.add_parameter('spar.wall_thickness[2]',low=100.,high=1000.,scaler=0.0001)
        self.driver.add_parameter('spar.wall_thickness[3]',low=100.,high=1000.,scaler=0.0001)
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


def sys_print(example):
    fullStiffeners = full_stiffeners_table()
    print 'number of stiffeners: ',example.number_of_rings
    print 'wall thickness: ',example.wall_thickness
    print 'VAL: ',example.VAL
    print 'VAG: ',example.VAG
    print 'VEL: ',example.VEL
    print 'VEG: ',example.VEG
    print 'web compactness: ',example.web_compactness
    print 'flange compactness: ',example.flange_compactness
    print 'stiffener: ', fullStiffeners[example.stiffener_index]
    print 'shell+ring+bulkhead mass: ',example.shell_ring_bulkhead_mass
    
def example_218WD_3MW():
    example = optimizationSpar()
    tt = time.time()
    example.spar.water_depth = 218.
    example.spar.load_condition = 'N'
    example.spar.significant_wave_height = 10.820
    example.spar.significant_wave_period = 9.800
    example.spar.keel_cg_mooring = 35.861
    example.spar.keel_cg_operating_system = 20.019
    example.spar.reference_wind_speed = 11.
    example.spar.reference_height = 75.
    example.spar.alpha = 0.110
    example.spar.material_density = 7850.
    example.spar.E = 200.e9
    example.spar.nu = 0.3
    example.spar.yield_stress = 345000000.
    example.spar.rotor_mass = 125000.
    example.spar.tower_mass = 127877.
    example.spar.free_board = 13.
    example.spar.draft = 67.
    example.spar.fixed_ballast_mass = 1244227.77
    example.spar.hull_mass = 890985.086
    example.spar.permanent_ballast_mass = 838450.256
    example.spar.variable_ballast_mass = 484661.874
    example.spar.number_of_sections = 4
    example.spar.outer_diameter = [5., 6., 6., 9.]
    example.spar.length = [6., 12., 15., 47.]
    example.spar.end_elevation = [7., -5., -20., -67.]
    example.spar.start_elevation = [13., 7., -5., -20.]
    example.spar.bulk_head = ['N', 'T', 'N', 'B']
    #example.system_acceleration = 1.27894072011959
    example.spar.gust_factor = 1.0
    example.spar.tower_base_OD = 4.890
    example.spar.tower_top_OD = 2.5
    example.spar.tower_length = 60.5
    example.spar.cut_out_speed = 25.
    example.spar.turbine_size = '3MW'
    example.spar.rotor_diameter = 101.0
    example.run()
    yna = convert_units(example.spar.neutral_axis ,'m','inch')
    fullStiffeners = full_stiffeners_table()
    for i in range (0,len(fullStiffeners)-1):
        stiffener_bef = fullStiffeners[i]
        stiffener_aft = fullStiffeners[i+1]
        if yna > stiffener_bef[6] and yna<stiffener_aft[6]:
            opt_index = i+1
    second_fit = Spar()
    second_fit.wall_thickness = example.spar.wall_thickness
    second_fit.number_of_rings = example.spar.number_of_rings
    second_fit.stiffener_index = opt_index
    second_fit.initial_pass = False
    second_fit.water_depth =  example.spar.water_depth
    second_fit.load_condition = example.spar.load_condition
    second_fit.significant_wave_height =  example.spar.significant_wave_height
    second_fit.significant_wave_period = example.spar.significant_wave_period
    second_fit.keel_cg_mooring = example.spar.keel_cg_mooring
    second_fit.keel_cg_operating_system = example.spar.keel_cg_operating_system
    second_fit.reference_wind_speed = example.spar.reference_wind_speed
    second_fit.reference_height = example.spar.reference_height
    second_fit.alpha = example.spar.alpha
    second_fit.material_density = example.spar.material_density
    second_fit.E = example.spar.E
    second_fit.nu =example.spar.nu
    second_fit.yield_stress = example.spar.yield_stress
    second_fit.rotor_mass = example.spar.rotor_mass
    second_fit.tower_mass = example.spar.tower_mass
    second_fit.free_board = example.spar.free_board
    second_fit.draft = example.spar.draft
    second_fit.fixed_ballast_mass = example.spar.fixed_ballast_mass
    second_fit.hull_mass = example.spar.hull_mass
    second_fit.permanent_ballast_mass = example.spar.permanent_ballast_mass
    second_fit.variable_ballast_mass = example.spar.variable_ballast_mass
    second_fit.number_of_sections = example.spar.number_of_sections
    second_fit.outer_diameter = example.spar.outer_diameter
    second_fit.length = example.spar.length
    second_fit.end_elevation = example.spar.end_elevation
    second_fit.start_elevation = example.spar.start_elevation
    second_fit.bulk_head = example.spar.bulk_head
    #second_fit.system_acceleration=example.system_acceleration
    second_fit.gust_factor = example.spar.gust_factor
    second_fit.tower_base_OD = example.spar.tower_base_OD
    second_fit.tower_top_OD = example.spar.tower_top_OD 
    second_fit.tower_length = example.spar.tower_length 
    second_fit.cut_out_speed = example.spar.cut_out_speed
    second_fit.turbine_size = example.spar.turbine_size
    second_fit.rotor_diameter = example.spar.rotor_diameter
    second_fit.run()
    mass = second_fit.shell_ring_bulkhead_mass
    index = opt_index
    best_index = opt_index
    unity = max(second_fit.web_compactness,second_fit.flange_compactness,max(second_fit.VAL),max(second_fit.VAG),max(second_fit.VEL),max(second_fit.VEG))   
    for i in range(opt_index,326):
        index += 1
        second_fit.stiffener_index = index
        second_fit.run()
        unity = max(second_fit.web_compactness,second_fit.flange_compactness,max(second_fit.VAL),max(second_fit.VAG),max(second_fit.VEL),max(second_fit.VEG)) 
        if unity < 1.0:
            if second_fit.shell_ring_bulkhead_mass < mass : 
                mass=second_fit.shell_ring_bulkhead_mass 
                best_index = i
                second_fit.stiffener_index = best_index
                second_fit.run()
    
    print '--------------example_218WD_3MW------------------'
    print "Elapsed time: ", time.time()-tt, " seconds"
    sys_print(second_fit)

def example_218WD_6MW():
    example = optimizationSpar()
    tt = time.time()
    example.spar.water_depth = 218.
    example.spar.load_condition = 'N'
    example.spar.significant_wave_height = 10.820
    example.spar.significant_wave_period = 9.800
    example.spar.keel_cg_mooring = 37.177
    example.spar.keel_cg_operating_system = 24.059
    example.spar.reference_wind_speed = 11.
    example.spar.reference_height = 97.
    example.spar.alpha = 0.110
    example.spar.material_density = 7850.
    example.spar.E = 200.e9
    example.spar.nu = 0.3
    example.spar.yield_stress = 345000000.
    example.spar.rotor_mass = 365500.000
    example.spar.tower_mass = 366952.000
    example.spar.free_board = 13.
    example.spar.draft = 72.
    example.spar.fixed_ballast_mass = 3659547.034
    example.spar.hull_mass = 1593822.041
    example.spar.permanent_ballast_mass = 1761475.914
    example.spar.variable_ballast_mass = 820790.246
    example.spar.number_of_sections = 4
    example.spar.outer_diameter = [7., 8., 8., 13.]
    example.spar.length = [6., 12., 15., 52.]
    example.spar.end_elevation = [7., -5., -20., -72.]
    example.spar.start_elevation = [13., 7., -5., -20.]
    example.spar.bulk_head = ['N', 'T', 'N', 'B']
    #example.spar.system_acceleration = 1.12124749328663
    example.spar.gust_factor = 1.0
    example.spar.tower_base_OD = 6.0
    example.spar.tower_top_OD = 3.51
    example.spar.tower_length = 80.5
    example.spar.cut_out_speed = 25.
    example.spar.turbine_size = '6MW'
    example.spar.rotor_diameter = 154.0
    example.run()
    
    yna = convert_units(example.spar.neutral_axis ,'m','inch')
    fullStiffeners = full_stiffeners_table()
    for i in range (0,len(fullStiffeners)-1):
        stiffener_bef = fullStiffeners[i]
        stiffener_aft = fullStiffeners[i+1]
        if yna > stiffener_bef[6] and yna<stiffener_aft[6]:
            opt_index = i+1
    second_fit = Spar()
    second_fit.wall_thickness = example.spar.wall_thickness
    second_fit.number_of_rings = example.spar.number_of_rings
    second_fit.stiffener_index = opt_index
    second_fit.initial_pass = False
    second_fit.water_depth =  example.spar.water_depth
    second_fit.load_condition = example.spar.load_condition
    second_fit.significant_wave_height =  example.spar.significant_wave_height
    second_fit.significant_wave_period = example.spar.significant_wave_period
    second_fit.keel_cg_mooring = example.spar.keel_cg_mooring
    second_fit.keel_cg_operating_system = example.spar.keel_cg_operating_system
    second_fit.reference_wind_speed = example.spar.reference_wind_speed
    second_fit.reference_height = example.spar.reference_height
    second_fit.alpha = example.spar.alpha
    second_fit.material_density = example.spar.material_density
    second_fit.E = example.spar.E
    second_fit.nu =example.spar.nu
    second_fit.yield_stress = example.spar.yield_stress
    second_fit.rotor_mass = example.spar.rotor_mass
    second_fit.tower_mass = example.spar.tower_mass
    second_fit.free_board = example.spar.free_board
    second_fit.draft = example.spar.draft
    second_fit.fixed_ballast_mass = example.spar.fixed_ballast_mass
    second_fit.hull_mass = example.spar.hull_mass
    second_fit.permanent_ballast_mass = example.spar.permanent_ballast_mass
    second_fit.variable_ballast_mass = example.spar.variable_ballast_mass
    second_fit.number_of_sections = example.spar.number_of_sections
    second_fit.outer_diameter = example.spar.outer_diameter
    second_fit.length = example.spar.length
    second_fit.end_elevation = example.spar.end_elevation
    second_fit.start_elevation = example.spar.start_elevation
    second_fit.bulk_head = example.spar.bulk_head
    #second_fit.system_acceleration=example.system_acceleration
    second_fit.gust_factor = example.spar.gust_factor
    second_fit.tower_base_OD = example.spar.tower_base_OD
    second_fit.tower_top_OD = example.spar.tower_top_OD 
    second_fit.tower_length = example.spar.tower_length 
    second_fit.cut_out_speed = example.spar.cut_out_speed
    second_fit.turbine_size = example.spar.turbine_size
    second_fit.rotor_diameter = example.spar.rotor_diameter
    #second_fit.system_acceleration=example.system_acceleration
    second_fit.run()
    mass = second_fit.shell_ring_bulkhead_mass
    index = opt_index
    best_index = opt_index
    unity = max(second_fit.web_compactness,second_fit.flange_compactness,max(second_fit.VAL),max(second_fit.VAG),max(second_fit.VEL),max(second_fit.VEG))   
    for i in range(opt_index,326):
        index += 1
        second_fit.stiffener_index = index
        second_fit.run()
        unity = max(second_fit.web_compactness,second_fit.flange_compactness,max(second_fit.VAL),max(second_fit.VAG),max(second_fit.VEL),max(second_fit.VEG)) 
        if unity < 1.0:
            if second_fit.shell_ring_bulkhead_mass < mass : 
                mass=second_fit.shell_ring_bulkhead_mass 
                best_index = i
                second_fit.stiffener_index = best_index
                second_fit.run()
    print '--------------example_218WD_6MW------------------'
    print "Elapsed time: ", time.time()-tt, " seconds"
    sys_print(second_fit)

def example_218WD_10MW():
    example = optimizationSpar()
    tt = time.time()
    example.first_fit=True
    example.spar.water_depth = 218.
    example.spar.load_condition = 'N'
    example.spar.significant_wave_height = 10.820
    example.spar.significant_wave_period = 9.800
    example.spar.keel_cg_mooring = 45.
    example.spar.keel_cg_operating_system = 30.730
    example.spar.reference_wind_speed = 11.
    example.spar.reference_height = 119.
    example.spar.alpha = 0.110
    example.spar.material_density = 7850.
    example.spar.E = 200.e9
    example.spar.nu = 0.3
    example.spar.yield_stress = 345000000.
    example.spar.rotor_mass = 677000.000
    example.spar.tower_mass = 698235.000
    example.spar.free_board = 13.
    example.spar.draft = 92.
    example.spar.fixed_ballast_mass = 6276669.794
    example.spar.hull_mass = 2816863.293
    example.spar.permanent_ballast_mass = 3133090.391
    example.spar.variable_ballast_mass = 1420179.260
    example.spar.number_of_sections = 4
    example.spar.outer_diameter = [8.,9.,9.,15.]
    example.spar.length = [6., 12., 15., 72.]
    example.spar.end_elevation = [7., -5., -20., -92.]
    example.spar.start_elevation = [13., 7., -5., -20.]
    example.spar.bulk_head = ['N', 'T', 'N', 'B']
    #example.system_acceleration = 0.856545480516845
    example.spar.gust_factor = 1.0
    example.spar.tower_base_OD = 7.720
    example.spar.tower_top_OD = 4.050
    example.spar.tower_length = 102.63
    example.spar.cut_out_speed = 25.
    example.spar.turbine_size = '10MW'
    example.spar.rotor_diameter = 194.0
    example.run()
    
    yna = convert_units(example.spar.neutral_axis ,'m','inch')
    fullStiffeners = full_stiffeners_table()
    for i in range (0,len(fullStiffeners)-1):
        stiffener_bef = fullStiffeners[i]
        stiffener_aft = fullStiffeners[i+1]
        if yna > stiffener_bef[6] and yna<stiffener_aft[6]:
            opt_index = i+1
    second_fit = Spar()
    second_fit.wall_thickness = example.spar.wall_thickness
    second_fit.number_of_rings = example.spar.number_of_rings
    second_fit.stiffener_index = opt_index
    second_fit.initial_pass = False
    second_fit.water_depth =  example.spar.water_depth
    second_fit.load_condition = example.spar.load_condition
    second_fit.significant_wave_height =  example.spar.significant_wave_height
    second_fit.significant_wave_period = example.spar.significant_wave_period
    second_fit.keel_cg_mooring = example.spar.keel_cg_mooring
    second_fit.keel_cg_operating_system = example.spar.keel_cg_operating_system
    second_fit.reference_wind_speed = example.spar.reference_wind_speed
    second_fit.reference_height = example.spar.reference_height
    second_fit.alpha = example.spar.alpha
    second_fit.material_density = example.spar.material_density
    second_fit.E = example.spar.E
    second_fit.nu =example.spar.nu
    second_fit.yield_stress = example.spar.yield_stress
    second_fit.rotor_mass = example.spar.rotor_mass
    second_fit.tower_mass = example.spar.tower_mass
    second_fit.free_board = example.spar.free_board
    second_fit.draft = example.spar.draft
    second_fit.fixed_ballast_mass = example.spar.fixed_ballast_mass
    second_fit.hull_mass = example.spar.hull_mass
    second_fit.permanent_ballast_mass = example.spar.permanent_ballast_mass
    second_fit.variable_ballast_mass = example.spar.variable_ballast_mass
    second_fit.number_of_sections = example.spar.number_of_sections
    second_fit.outer_diameter = example.spar.outer_diameter
    second_fit.length = example.spar.length
    second_fit.end_elevation = example.spar.end_elevation
    second_fit.start_elevation = example.spar.start_elevation
    second_fit.bulk_head = example.spar.bulk_head
    #second_fit.system_acceleration=example.system_acceleration
    second_fit.gust_factor = example.spar.gust_factor
    second_fit.tower_base_OD = example.spar.tower_base_OD
    second_fit.tower_top_OD = example.spar.tower_top_OD 
    second_fit.tower_length = example.spar.tower_length 
    second_fit.cut_out_speed = example.spar.cut_out_speed
    second_fit.turbine_size = example.spar.turbine_size
    second_fit.rotor_diameter = example.spar.rotor_diameter
    #second_fit.system_acceleration=example.system_acceleration
    second_fit.run()
    mass = second_fit.shell_ring_bulkhead_mass
    index = opt_index
    best_index = opt_index
    unity = max(second_fit.web_compactness,second_fit.flange_compactness,max(second_fit.VAL),max(second_fit.VAG),max(second_fit.VEL),max(second_fit.VEG))   
    for i in range(opt_index,326):
        index += 1
        second_fit.stiffener_index = index
        second_fit.run()
        unity = max(second_fit.web_compactness,second_fit.flange_compactness,max(second_fit.VAL),max(second_fit.VAG),max(second_fit.VEL),max(second_fit.VEG)) 
        if unity < 1.0:
            if second_fit.shell_ring_bulkhead_mass < mass : 
                mass=second_fit.shell_ring_bulkhead_mass 
                best_index = i
                second_fit.stiffener_index = best_index
                second_fit.run()
    print '--------------example_218WD_10MW------------------'
    print "Elapsed time: ", time.time()-tt, " seconds"
    
    sys_print(second_fit)
    
    

if __name__ == "__main__":
    #example_218WD_3MW()
    #example_218WD_6MW()
    example_218WD_10MW()