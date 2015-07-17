from openmdao.main.api import Component, Assembly, convert_units
from openmdao.main.datatypes.api import Float, Array, Enum, Str, Int, Bool
from openmdao.lib.drivers.api import COBYLAdriver, SLSQPdriver
from mooring import Mooring
import time
import numpy as np

class optimizationMooring(Assembly):
    # variables 
    def configure(self):
        self.add('driver',COBYLAdriver())
        self.add('mooring',Mooring())
        self.driver.workflow.add('mooring')

        self.driver.add_objective('mooring.mooring_total_cost')

        self.driver.add_parameter('mooring.scope_ratio',low=15.,high=50.,scaler=0.1)
        self.driver.add_parameter('mooring.pretension_percent',low=2.5,high=20.)
        self.driver.add_parameter('mooring.mooring_diameter',low=3.,high=10.,scaler=0.01)

        self.driver.add_constraint('mooring.heel_angle <= 6.')
        self.driver.add_constraint('mooring.min_offset_unity < 1.0')
        self.driver.add_constraint('mooring.max_offset_unity < 1.0')

def sys_print(example):
    print 'scope ratio: ',example.scope_ratio
    print 'pretension percent: ',example.pretension_percent
    print 'mooring diameter: ',example.mooring_diameter
    print 'heel angle: ',example.heel_angle
    print 'min offset unity: ',example.min_offset_unity
    print 'max offset unity: ',example.max_offset_unity
    print 'total mooring cost: ',example.mooring_total_cost

def example_218WD_3MW():
    tt = time.time()
    example = optimizationMooring()
    # Mooring,settings
    example.mooring.fairlead_depth = 13.
    example.mooring.scope_ratio = 1.5
    example.mooring.pretension_percent = 5.0
    example.mooring.mooring_diameter = 0.090
    example.mooring.number_of_mooring_lines = 3
    example.mooring.permanent_ballast_height = 3.
    example.mooring.fixed_ballast_height = 5.
    example.mooring.permanent_ballast_density = 4492.
    example.mooring.fixed_ballast_density = 4000.
    example.mooring.water_depth = 218.
    example.mooring.mooring_type = 'CHAIN'
    example.mooring.anchor_type = 'PILE'
    example.mooring.fairlead_offset_from_shell = 0.5
    # from,spar.py
    example.mooring.shell_buoyancy = [0.000,144905.961,688303.315,3064761.078]
    example.mooring.shell_mass = [40321.556,88041.563,137796.144,518693.048]
    example.mooring.bulkhead_mass = [0.000,10730.836,0.000,24417.970]
    example.mooring.ring_mass = [1245.878,5444.950,6829.259,28747.490]
    example.mooring.spar_start_elevation = [13., 7., -5., -20.]
    example.mooring.spar_end_elevation = [7., -5., -20., -67.]
    example.mooring.spar_keel_to_CG = 35.861 
    example.mooring.spar_keel_to_CB = 30.324
    example.mooring.spar_outer_diameter = [5.000,6.000,6.000,9.000]
    example.mooring.spar_wind_force = [1842.442,1861.334,0.000,0.000]
    example.mooring.spar_wind_moment = [100965.564,85586.296,0.000,0.000]
    example.mooring.spar_current_force = [0.000,449016.587,896445.823,49077.906]
    example.mooring.spar_current_moment = [0.000,19074749.640,28232958.052,72692.688]
    example.mooring.wall_thickness = [0.05,0.05,0.05,0.05]
    example.mooring.load_condition = 'N'
    # from,tower_RNA.py
    example.mooring.RNA_mass = 125000.000
    example.mooring.tower_mass = 127877.000
    example.mooring.tower_center_of_gravity = 23.948
    example.mooring.RNA_keel_to_CG = 142.000
    example.mooring.tower_wind_force = 19950.529
    example.mooring.tower_wind_moment = 1634522.835
    example.mooring.RNA_wind_force = 391966.178 
    example.mooring.RNA_wind_moment = 47028560.389
    example.mooring.RNA_center_of_gravity_x = 4.1

    example.run()
    print '--------------example_218WD_3MW------------------'
    print "Elapsed time: ", time.time()-tt, " seconds"
    sys_print(example.mooring)

def example_218WD_6MW():
    tt = time.time()
    example = optimizationMooring()
    example.mooring.fairlead_depth = 13.
    example.mooring.scope_ratio = 1.5
    example.mooring.pretension_percent = 5.0
    example.mooring.mooring_diameter = 0.090
    example.mooring.number_of_mooring_lines = 3
    example.mooring.permanent_ballast_height = 3.
    example.mooring.fixed_ballast_height = 7.
    example.mooring.permanent_ballast_density = 4492.
    example.mooring.fixed_ballast_density = 4000.
    example.mooring.water_depth = 218.
    example.mooring.mooring_type = 'CHAIN'
    example.mooring.anchor_type = 'PILE'
    example.mooring.fairlead_offset_from_shell = 0.5
    
    example.mooring.shell_buoyancy = [0.000,257610.598,1356480.803,7074631.036]
    example.mooring.shell_mass = [55118.458,117635.366,193284.525,830352.783]
    example.mooring.bulkhead_mass = [0.000,19239.055,0.000,51299.008]
    example.mooring.ring_mass = [3838.515,16391.495,21578.677,127137.831]
    example.mooring.spar_start_elevation = [13., 7., -5., -20.]
    example.mooring.spar_end_elevation = [7., -5., -20., -72.]
    example.mooring.spar_keel_to_CG = 37.177
    example.mooring.spar_keel_to_CB = 32.337
    example.mooring.spar_outer_diameter = [7.,8.,8.,13.]
    example.mooring.spar_wind_force = [2374.194,2345.237,0.000,0.000]
    example.mooring.spar_wind_moment = [137246.585,114777.740,0.000,0.0000]
    example.mooring.spar_current_force = [0.000,824040.566,1968613.701,182335.850]
    example.mooring.spar_current_moment = [0.000,37445057.967,67469109.912,353876.402]
    example.mooring.wall_thickness = [0.05,0.05,0.05,0.05]
    example.mooring.load_condition = 'N'
    
    example.mooring.RNA_mass = 365500.000
    example.mooring.tower_mass = 366952.000
    example.mooring.tower_center_of_gravity = 33.381
    example.mooring.RNA_keel_to_CG = 169.000
    example.mooring.tower_wind_force = 33125.492
    example.mooring.tower_wind_moment = 3124462.452
    example.mooring.RNA_wind_force = 820818.422
    example.mooring.RNA_wind_moment = 118970074.187
    example.mooring.RNA_center_of_gravity_x = 5.750

    example.run()
    print '--------------example_218WD_6MW------------------'
    print "Elapsed time: ", time.time()-tt, " seconds"
    sys_print(example.mooring)

def example_218WD_10MW():
    tt = time.time()
    example = optimizationMooring()
    example.mooring.fairlead_depth = 13.
    example.mooring.scope_ratio = 1.5
    example.mooring.pretension_percent = 5.0
    example.mooring.mooring_diameter = 0.090
    example.mooring.number_of_mooring_lines = 3
    example.mooring.permanent_ballast_height = 4.
    example.mooring.fixed_ballast_height = 9.
    example.mooring.permanent_ballast_density = 4492.
    example.mooring.fixed_ballast_density = 4000.
    example.mooring.water_depth = 218.
    example.mooring.mooring_type = 'CHAIN'
    example.mooring.anchor_type = 'PILE'
    example.mooring.fairlead_offset_from_shell = 0.5
    
    example.mooring.shell_buoyancy = [0.000,326038.413,1775098.024,13041536.503]
    example.mooring.shell_mass = [62516.908,132432.268,221028.715,1335368.667]
    example.mooring.bulkhead_mass = [0.000,24417.970,0.000,68438.752]
    example.mooring.ring_mass = [6963.553,29512.202,39460.135,617575.510]
    example.mooring.spar_start_elevation = [13., 7., -5., -20.]
    example.mooring.spar_end_elevation = [7., -5., -20., -92.]
    example.mooring.spar_keel_to_CG = 45.
    example.mooring.spar_keel_to_CB = 42.108
    example.mooring.spar_outer_diameter = [8.,9.,9.,15.]
    example.mooring.spar_wind_force = [2572.428,2522.369,0.000,0.000]
    example.mooring.spar_wind_moment = [183034.454,157067.701,0.000,0.000]
    example.mooring.spar_current_force = [0.000,1125719.734,3051908.296,425853.543]
    example.mooring.spar_current_moment = [0.000,66158450.987,145104271.963,2244211.189]
    example.mooring.wall_thickness = [0.050,0.050,0.050,0.050]
    example.mooring.load_condition = 'N'
    
    example.mooring.RNA_mass = 677000.000
    example.mooring.tower_mass = 698235.000
    example.mooring.tower_center_of_gravity = 40.983
    example.mooring.RNA_keel_to_CG = 211.000
    example.mooring.tower_wind_force = 53037.111
    example.mooring.tower_wind_moment = 6112673.024
    example.mooring.RNA_wind_force = 1743933.574
    example.mooring.RNA_wind_moment = 314378753.986
    example.mooring.RNA_center_of_gravity_x = 7.070

    example.run()
    print '--------------example_218WD_10MW------------------'
    print "Elapsed time: ", time.time()-tt, " seconds"
    sys_print(example.mooring)

if __name__ == "__main__":
    #example_218WD_3MW()
    #example_218WD_6MW()
    example_218WD_10MW()