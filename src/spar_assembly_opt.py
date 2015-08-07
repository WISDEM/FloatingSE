from openmdao.main.api import Component, Assembly, convert_units
from openmdao.main.datatypes.api import Float, Array, Enum, Str, Int, Bool
from openmdao.lib.drivers.api import COBYLAdriver,SLSQPdriver
from spar import Spar
from mooring import Mooring
from tower_RNA import Tower_RNA
from spar_assembly import Spar_Assembly
#from spar_discrete import spar_discrete
import numpy as np
import time

class sparAssembly(Assembly):

    myvar = Float(iotype='in')

    def configure(self):
        self.add('driver',COBYLAdriver())
        self.driver.maxfun = 100000
        # select components
        self.add('spar_assembly',Spar_Assembly())
        
        # workflow
        self.driver.workflow.add(['spar_assembly'])

        # objective
        self.driver.add_objective('spar_assembly.spar.spar_mass')
       
        # design variables
        self.driver.add_parameter('spar_assembly.neutral_axis',low=10.,high=41.9,scaler=0.01)
        #self.driver.add_parameter('number_of_rings[0]',low=1,high=5)
        self.driver.add_parameter('spar_assembly.number_of_rings[1]',low=1,high=10)
        self.driver.add_parameter('spar_assembly.number_of_rings[2]',low=1,high=10)
        self.driver.add_parameter('spar_assembly.number_of_rings[3]',low=1,high=50)
        self.driver.add_parameter('spar_assembly.wall_thickness[0]',low=1.,high=10.,scaler=0.01)
        self.driver.add_parameter('spar_assembly.wall_thickness[1]',low=1.,high=10.,scaler=0.01)
        self.driver.add_parameter('spar_assembly.wall_thickness[2]',low=1.,high=10.,scaler=0.01)
        self.driver.add_parameter('spar_assembly.wall_thickness[3]',low=10.,high=100.,scaler=0.001)
        self.driver.add_parameter('spar_assembly.scope_ratio',low=15.,high=45.,scaler=0.1)
        self.driver.add_parameter('spar_assembly.pretension_percent',low=2.5,high=10.)
        self.driver.add_parameter('spar_assembly.mooring_diameter',low=30.,high=100.,scaler=0.001)
        self.driver.add_parameter('spar_assembly.fixed_ballast_height',low=30.,high=100.,scaler=0.1)
        self.driver.add_parameter('spar_assembly.permanent_ballast_height',low=30.,high=100.,scaler=0.1)

        # Constraints

        self.driver.add_constraint('spar_assembly.spar.water_ballast_height < 7.5')
        self.driver.add_constraint('spar_assembly.spar.water_ballast_height > 5.5')
        self.driver.add_constraint('spar_assembly.spar.flange_compactness < 1.')
        self.driver.add_constraint('spar_assembly.spar.web_compactness < 1.')

        self.driver.add_constraint('spar_assembly.spar.VAL[0] < 0.99')
        self.driver.add_constraint('spar_assembly.spar.VAL[1] < 0.99')
        self.driver.add_constraint('spar_assembly.spar.VAL[2] < 0.99')
        self.driver.add_constraint('spar_assembly.spar.VAL[3] < 0.99')

        self.driver.add_constraint('spar_assembly.spar.VAG[0] < 0.99')
        self.driver.add_constraint('spar_assembly.spar.VAG[1] < 0.99')
        self.driver.add_constraint('spar_assembly.spar.VAG[2] < 0.99')
        self.driver.add_constraint('spar_assembly.spar.VAG[3] < 0.99')

        self.driver.add_constraint('spar_assembly.spar.VEL[0] < 0.99')
        self.driver.add_constraint('spar_assembly.spar.VEL[1] < 0.99')
        self.driver.add_constraint('spar_assembly.spar.VEL[2] < 0.99')
        self.driver.add_constraint('spar_assembly.spar.VEL[3] < 0.99')

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
    print 'shell mass: ', example.spar.shell_mass
    print 'bulkhead mass: ', example.spar.bulkhead_mass
    print 'stiffener mass: ', example.spar.stiffener_mass
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
    example.spar.stiffener_curve_fit = True
    example.neutral_axis = 0.21
    #example.stiffener_index = 232
    example.permanent_ballast_height = 3.
    example.fixed_ballast_height = 5.
    example.wall_thickness=[0.05,0.05,0.05,0.05]
    example.number_of_rings = [1,4,4,14]
    example.number_of_sections = 4
    example.bulk_head = ['N', 'T', 'N', 'B']
    example.load_condition = 'N'
    example.significant_wave_height = 10.820*1.5
    example.significant_wave_period = 9.800
    example.run()
    print '----------218WD_3MW------------'
    sys_print(example)

if __name__ == "__main__":
    example_218WD_3MW()
    #example_218WD_6MW()
    #example_218WD_10MW()
    