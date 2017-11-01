from openmdao.main.api import Component, Assembly, convert_units
from openmdao.main.datatypes.api import Float, Array, Enum, Str, Int, Bool
from openmdao.lib.drivers.api import COBYLAdriver, SLSQPdriver
from spar import Spar
#from spar_discrete import spar_discrete
import numpy as np
import time
from utils import filtered_stiffeners_table, full_stiffeners_table

class optimizationSpar(Assembly):
    
    def configure(self):
        self.add('driver',COBYLAdriver())
        self.driver.maxfun = 100000
        self.add('spar',Spar())
        self.driver.workflow.add('spar')
        
        # objective
        self.driver.add_objective('spar.total_cost')
       
        # design variables
        self.driver.add_parameter('spar.neutral_axis',low=100.,high=419.,scaler=0.001)
        #self.driver.add_parameter('spar.number_of_rings[0]',low=10,high=50,scaler=0.1)
        self.driver.add_parameter('spar.number_of_rings[1]',low=10,high=100,scaler=0.1)
        self.driver.add_parameter('spar.number_of_rings[2]',low=10,high=100,scaler=0.1)
        self.driver.add_parameter('spar.number_of_rings[3]',low=1,high=80)
        self.driver.add_parameter('spar.wall_thickness[0]',low=100.,high=1000.,scaler=0.0001)
        self.driver.add_parameter('spar.wall_thickness[1]',low=100.,high=1000.,scaler=0.0001)
        self.driver.add_parameter('spar.wall_thickness[2]',low=100.,high=1000.,scaler=0.0001)
        self.driver.add_parameter('spar.wall_thickness[3]',low=100.,high=1000.,scaler=0.0001)
        # Constraints
        self.driver.add_constraint('spar.flange_compactness <= 1.')
        self.driver.add_constraint('spar.web_compactness <= 1.')
        self.driver.add_constraint('spar.platform_stability_check <= 1.')
        #self.driver.add_constraint('spar.heel_angle <= 6.')
        self.driver.add_constraint('spar.min_offset_unity <= 1.')
        self.driver.add_constraint('spar.max_offset_unity <= 1.')
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
    print 'cost: ', example.total_cost

def example_218WD_3MW():
    example = optimizationSpar()
    tt = time.time()
    example.spar.number_of_sections = 4
    example.spar.outer_diameter = [5., 6., 6., 9.]
    example.spar.length = [6., 12., 15., 47.]
    example.spar.end_elevation = [7., -5., -20., -67.]
    example.spar.start_elevation = [13., 7., -5., -20.]
    example.spar.bulk_head = ['N', 'T', 'N', 'B']
    example.spar.water_depth = 218.
    example.spar.load_condition = 'N'
    example.spar.significant_wave_height = 10.820
    example.spar.significant_wave_period = 9.800
    example.spar.wind_reference_speed = 11.
    example.spar.wind_reference_height = 75.
    example.spar.alpha = 0.110
    example.spar.RNA_keel_to_CG = 142.
    example.spar.RNA_mass = 125000.
    example.spar.tower_mass = 127877.
    example.spar.tower_center_of_gravity =  23.948
    example.spar.tower_wind_force = 19950.529
    example.spar.RNA_wind_force = 391966.178
    example.spar.RNA_center_of_gravity_x = 4.1
    example.spar.mooring_total_cost = 810424.053596  
    example.spar.mooring_keel_to_CG = 54.000
    example.spar.mooring_vertical_load = 1182948.791
    example.spar.mooring_horizontal_stiffness = 3145.200
    example.spar.mooring_vertical_stiffness =7111.072
    example.spar.sum_forces_x = [7886.848,437.844,253.782,174.651,124.198,85.424,50.689,17.999,-13.222,-43.894,-74.245,-105.117,-136.959,-170.729,-207.969,-251.793,-307.638,-386.514,-518.734,-859.583,-11091.252]
    example.spar.offset_x = [-40.362,-35.033,-29.703,-24.373,-19.044,-13.714,-8.385,-3.055,2.274,7.604,12.933,18.263,23.593,28.922,34.252,39.581,44.911,50.240,55.570,60.900,66.229]
    example.spar.damaged_mooring = [-40.079,65.824]
    example.spar.intact_mooring = [-39.782,65.399]
    example.spar.run()
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
    second_fit.stiffener_curve_fit = False
    second_fit.stiffener_index = opt_index
    second_fit.number_of_sections = example.spar.number_of_sections
    second_fit.outer_diameter = example.spar.outer_diameter 
    second_fit.length = example.spar.length 
    second_fit.end_elevation = example.spar.end_elevation 
    second_fit.start_elevation = example.spar.start_elevation 
    second_fit.bulk_head = example.spar.bulk_head 
    second_fit.water_depth = example.spar.water_depth 
    second_fit.load_condition = example.spar.load_condition 
    second_fit.significant_wave_height = example.spar.significant_wave_height
    second_fit.significant_wave_period = example.spar.significant_wave_period 
    second_fit.wind_reference_speed = example.spar.wind_reference_speed 
    second_fit.wind_reference_height = example.spar.wind_reference_height 
    second_fit.alpha = example.spar.alpha 
    second_fit.RNA_keel_to_CG = example.spar.RNA_keel_to_CG 
    second_fit.RNA_mass = example.spar.RNA_mass
    second_fit.tower_mass = example.spar.tower_mass
    second_fit.tower_center_of_gravity = example.spar.tower_center_of_gravity 
    second_fit.tower_wind_force = example.spar.tower_wind_force 
    second_fit.RNA_wind_force = example.spar.RNA_wind_force 
    second_fit.RNA_center_of_gravity_x = example.spar.RNA_center_of_gravity_x 
    second_fit.mooring_total_cost = example.spar.mooring_total_cost  
    second_fit.mooring_keel_to_CG = example.spar.mooring_keel_to_CG 
    second_fit.mooring_vertical_load = example.spar.mooring_vertical_load 
    second_fit.mooring_horizontal_stiffness = example.spar.mooring_horizontal_stiffness 
    second_fit.mooring_vertical_stiffness = example.spar.mooring_vertical_stiffness 
    second_fit.sum_forces_x = example.spar.sum_forces_x 
    second_fit.offset_x = example.spar.offset_x
    second_fit.damaged_mooring = example.spar.damaged_mooring 
    second_fit.intact_mooring = example.spar.intact_mooring 
    second_fit.run()

    cost = second_fit.total_cost
    index = opt_index
    best_index = opt_index
    unity = max(second_fit.web_compactness,second_fit.flange_compactness,max(second_fit.VAL),max(second_fit.VAG),max(second_fit.VEL),max(second_fit.VEG))   
    for i in range(opt_index,326):
        index += 1
        second_fit.stiffener_index = index
        second_fit.run()
        unity = max(second_fit.web_compactness,second_fit.flange_compactness,max(second_fit.VAL),max(second_fit.VAG),max(second_fit.VEL),max(second_fit.VEG)) 
        if unity < 1.0:
            if second_fit.total_cost < cost : 
                cost=second_fit.total_cost 
                best_index = i
                second_fit.stiffener_index = best_index
                second_fit.run()
    print '--------------example_218WD_3MW------------------'
    print "Elapsed time: ", time.time()-tt, " seconds"
    sys_print(second_fit)

def example_218WD_6MW():
    example = optimizationSpar()
    tt = time.time()
    example.spar.number_of_sections = 4
    example.spar.outer_diameter = [7., 8., 8., 13.]
    example.spar.length =  [6., 12., 15., 52.]
    example.spar.end_elevation = [7., -5., -20., -72.]
    example.spar.start_elevation = [13., 7., -5., -20.]
    example.spar.bulk_head = ['N', 'T', 'N', 'B']
    example.spar.water_depth = 218.
    example.spar.load_condition = 'N'
    example.spar.significant_wave_height = 10.820
    example.spar.significant_wave_period = 9.800
    example.spar.wind_reference_speed = 11.
    example.spar.wind_reference_height = 97.
    example.spar.alpha = 0.110
    example.spar.RNA_keel_to_CG = 169.000
    example.spar.RNA_mass = 365500.000
    example.spar.tower_mass = 366952.000
    example.spar.tower_center_of_gravity =  33.381
    example.spar.tower_wind_force = 33125.492
    example.spar.RNA_wind_force = 820818.422
    example.spar.RNA_center_of_gravity_x = 5.750
    example.spar.mooring_total_cost = 810424.054
    example.spar.mooring_keel_to_CG = 59.000
    example.spar.mooring_vertical_load = 1182948.791
    example.spar.mooring_horizontal_stiffness = 3145.200
    example.spar.mooring_vertical_stiffness = 7111.072
    example.spar.sum_forces_x = [7886.848,437.844,253.782,174.651,124.198,85.424,50.689,17.999,-13.222,-43.894,-74.245,-105.117,-136.959,-170.729,-207.969,-251.793,-307.638,-386.514,-518.734,-859.583,-11091.252]
    example.spar.offset_x = [-40.362,-35.033,-29.703,-24.373,-19.044,-13.714,-8.385,-3.055,2.274,7.604,12.933,18.263,23.593,28.922,34.252,39.581,44.911,50.240,55.570,60.900,66.229]
    example.spar.damaged_mooring = [-40.079,65.824]
    example.spar.intact_mooring = [-39.782,65.399]
    example.spar.run()
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
    second_fit.stiffener_curve_fit = False
    second_fit.stiffener_index = opt_index
    second_fit.number_of_sections = example.spar.number_of_sections
    second_fit.outer_diameter = example.spar.outer_diameter 
    second_fit.length = example.spar.length 
    second_fit.end_elevation = example.spar.end_elevation 
    second_fit.start_elevation = example.spar.start_elevation 
    second_fit.bulk_head = example.spar.bulk_head 
    second_fit.water_depth = example.spar.water_depth 
    second_fit.load_condition = example.spar.load_condition 
    second_fit.significant_wave_height = example.spar.significant_wave_height
    second_fit.significant_wave_period = example.spar.significant_wave_period 
    second_fit.wind_reference_speed = example.spar.wind_reference_speed 
    second_fit.wind_reference_height = example.spar.wind_reference_height 
    second_fit.alpha = example.spar.alpha 
    second_fit.RNA_keel_to_CG = example.spar.RNA_keel_to_CG 
    second_fit.RNA_mass = example.spar.RNA_mass
    second_fit.tower_mass = example.spar.tower_mass
    second_fit.tower_center_of_gravity = example.spar.tower_center_of_gravity 
    second_fit.tower_wind_force = example.spar.tower_wind_force 
    second_fit.RNA_wind_force = example.spar.RNA_wind_force 
    second_fit.RNA_center_of_gravity_x = example.spar.RNA_center_of_gravity_x 
    second_fit.mooring_total_cost = example.spar.mooring_total_cost  
    second_fit.mooring_keel_to_CG = example.spar.mooring_keel_to_CG 
    second_fit.mooring_vertical_load = example.spar.mooring_vertical_load 
    second_fit.mooring_horizontal_stiffness = example.spar.mooring_horizontal_stiffness 
    second_fit.mooring_vertical_stiffness = example.spar.mooring_vertical_stiffness 
    second_fit.sum_forces_x = example.spar.sum_forces_x 
    second_fit.offset_x = example.spar.offset_x
    second_fit.damaged_mooring = example.spar.damaged_mooring 
    second_fit.intact_mooring = example.spar.intact_mooring 
    second_fit.run()

    cost = second_fit.total_cost
    index = opt_index
    best_index = opt_index
    unity = max(second_fit.web_compactness,second_fit.flange_compactness,max(second_fit.VAL),max(second_fit.VAG),max(second_fit.VEL),max(second_fit.VEG))   
    for i in range(opt_index,326):
        index += 1
        second_fit.stiffener_index = index
        second_fit.run()
        unity = max(second_fit.web_compactness,second_fit.flange_compactness,max(second_fit.VAL),max(second_fit.VAG),max(second_fit.VEL),max(second_fit.VEG)) 
        if unity < 1.0:
            if second_fit.total_cost < cost : 
                cost=second_fit.total_cost 
                best_index = i
                second_fit.stiffener_index = best_index
                second_fit.run()
    print '--------------example_218WD_6MW------------------'
    print "Elapsed time: ", time.time()-tt, " seconds"
    sys_print(second_fit)

def example_218WD_10MW():
    example = optimizationSpar()
    tt = time.time()
    example.spar.number_of_sections = 4
    example.spar.outer_diameter = [8.,9.,9.,15.]
    example.spar.length =  [6., 12., 15., 72.]
    example.spar.end_elevation = [7., -5., -20., -92.]
    example.spar.start_elevation =  [13., 7., -5., -20.]
    example.spar.bulk_head = ['N', 'T', 'N', 'B']
    example.spar.water_depth = 218.
    example.spar.load_condition = 'N'
    example.spar.significant_wave_height = 10.820
    example.spar.significant_wave_period = 9.800
    example.spar.wind_reference_speed = 11.
    example.spar.wind_reference_height = 119. 
    example.spar.alpha = 0.110
    example.spar.RNA_keel_to_CG = 211.000
    example.spar.RNA_mass = 677000.000
    example.spar.tower_mass = 698235.000
    example.spar.tower_center_of_gravity =  40.983
    example.spar.tower_wind_force = 53037.111
    example.spar.RNA_wind_force = 1743933.574
    example.spar.RNA_center_of_gravity_x = 7.070
    example.spar.mooring_total_cost = 810424.054
    example.spar.mooring_keel_to_CG = 79.000
    example.spar.mooring_vertical_load = 1182948.791
    example.spar.mooring_horizontal_stiffness = 3145.200
    example.spar.mooring_vertical_stiffness = 7111.072
    example.spar.sum_forces_x = [7886.848,437.844,253.782,174.651,124.198,85.424,50.689,17.999,-13.222,-43.894,-74.245,-105.117,-136.959,-170.729,-207.969,-251.793,-307.638,-386.514,-518.734,-859.583,-11091.252]
    example.spar.offset_x = [-40.362,-35.033,-29.703,-24.373,-19.044,-13.714,-8.385,-3.055,2.274,7.604,12.933,18.263,23.593,28.922,34.252,39.581,44.911,50.240,55.570,60.900,66.229]
    example.spar.damaged_mooring = [-40.079,65.824]
    example.spar.intact_mooring = [-39.782,65.399]
    example.spar.run()
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
    second_fit.stiffener_curve_fit = False
    second_fit.stiffener_index = opt_index
    second_fit.number_of_sections = example.spar.number_of_sections
    second_fit.outer_diameter = example.spar.outer_diameter 
    second_fit.length = example.spar.length 
    second_fit.end_elevation = example.spar.end_elevation 
    second_fit.start_elevation = example.spar.start_elevation 
    second_fit.bulk_head = example.spar.bulk_head 
    second_fit.water_depth = example.spar.water_depth 
    second_fit.load_condition = example.spar.load_condition 
    second_fit.significant_wave_height = example.spar.significant_wave_height
    second_fit.significant_wave_period = example.spar.significant_wave_period 
    second_fit.wind_reference_speed = example.spar.wind_reference_speed 
    second_fit.wind_reference_height = example.spar.wind_reference_height 
    second_fit.alpha = example.spar.alpha 
    second_fit.RNA_keel_to_CG = example.spar.RNA_keel_to_CG 
    second_fit.RNA_mass = example.spar.RNA_mass
    second_fit.tower_mass = example.spar.tower_mass
    second_fit.tower_center_of_gravity = example.spar.tower_center_of_gravity 
    second_fit.tower_wind_force = example.spar.tower_wind_force 
    second_fit.RNA_wind_force = example.spar.RNA_wind_force 
    second_fit.RNA_center_of_gravity_x = example.spar.RNA_center_of_gravity_x 
    second_fit.mooring_total_cost = example.spar.mooring_total_cost  
    second_fit.mooring_keel_to_CG = example.spar.mooring_keel_to_CG 
    second_fit.mooring_vertical_load = example.spar.mooring_vertical_load 
    second_fit.mooring_horizontal_stiffness = example.spar.mooring_horizontal_stiffness 
    second_fit.mooring_vertical_stiffness = example.spar.mooring_vertical_stiffness 
    second_fit.sum_forces_x = example.spar.sum_forces_x 
    second_fit.offset_x = example.spar.offset_x
    second_fit.damaged_mooring = example.spar.damaged_mooring 
    second_fit.intact_mooring = example.spar.intact_mooring 
    second_fit.run()

    cost = second_fit.total_cost
    index = opt_index
    print 'opt:',opt_index
    best_index = opt_index
    unity = max(second_fit.web_compactness,second_fit.flange_compactness,max(second_fit.VAL),max(second_fit.VAG),max(second_fit.VEL),max(second_fit.VEG))   
    for i in range(opt_index,326):
        index += 1
        second_fit.stiffener_index = index
        second_fit.run()
        unity = max(second_fit.web_compactness,second_fit.flange_compactness,max(second_fit.VAL),max(second_fit.VAG),max(second_fit.VEL),max(second_fit.VEG)) 
        if unity-1.0  < 1.e-6:
            if second_fit.total_cost < cost : 
                cost=second_fit.total_cost 
                best_index = i
                print 'best: ',best_index
                #second_fit.stiffener_index = best_index
                #second_fit.run()
    print '--------------example_218WD_10MW------------------'
    print "Elapsed time: ", time.time()-tt, " seconds"
    sys_print(second_fit)

if __name__ == "__main__":
    #example_218WD_3MW()
    #example_218WD_6MW()
    example_218WD_10MW()