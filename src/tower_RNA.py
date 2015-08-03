from openmdao.main.api import Component, Assembly,convert_units
from openmdao.lib.datatypes.api import Float, Array, Str, Int, Bool
import numpy as np
from scipy.optimize import fmin, minimize
from sympy.solvers import solve
from sympy import Symbol
import math
from spar_utils import windPowerLaw,dragForce,CD,windDrag,thrust_table
pi=np.pi

class Tower_RNA(Component):
    # environmental inputs
    air_density = Float(1.198,iotype='in', units='kg/m**3', desc='density of air') 
    wind_reference_speed = Float(iotype='in', units='m/s', desc='reference wind speed')
    wind_reference_height = Float(iotype='in', units='m', desc='reference height')
    gust_factor = Float(1.0,iotype='in', desc='gust factor')
    alpha = Float(iotype='in', desc='power law exponent')
    # inputs
    base_outer_diameter = Float(iotype='in',units='m',desc='outer diameter of tower base')
    top_outer_diameter = Float(iotype='in',units='m',desc='outer diameter of tower top')
    length = Float(iotype='in',units='m',desc='tower length')
    spar_elevations = Array(iotype='in', units='m',desc = 'elevations of each section')
    example_turbine_size = Str(iotype='in',desc='for example cases, 3MW, 6MW, or 10 MW')
    rotor_diameter = Float(iotype='in', units='m',desc='rotor diameter')
    RNA_center_of_gravity_x = Float(iotype='in', units='m',desc='rotor center of gravity') 
    RNA_center_of_gravity_y = Float(iotype='in', units='m',desc='rotor center of gravity') 
    cut_out_speed = Float(25.,iotype='in', units='m/s',desc='cut-out speed of turbine') 
    tower_mass = Float(iotype='in', units='kg',desc='tower mass') 
    RNA_mass = Float(iotype='in', units='kg',desc='RNA mass') 
    # outputs
    tower_center_of_gravity = Float(iotype='out',units='m',desc='tower center of gravity')
    tower_keel_to_CG = Float(iotype='out',units='m',desc='keel to tower center of gravity')
    tower_wind_force = Float(iotype='out',units='N',desc='wind force on tower')
    RNA_keel_to_CG = Float(iotype='out',units='m',desc='keel to RNA center of gravity')
    RNA_wind_force = Float(iotype='out',units='N',desc='wind force on RNA')
    
    def __init__(self):
        super(Tower_RNA,self).__init__()
    def execute(self):
       
        # tower
        FB = self.spar_elevations[0]
        TBOD = self.base_outer_diameter
        TTOD = self.top_outer_diameter
        TLEN = self.length
        GF = self.gust_factor
        ELE = self.spar_elevations[1:]
        DRAFT = abs(min(ELE))
        WREFS = self.wind_reference_speed
        WREFH = self.wind_reference_height
        ALPHA = self.alpha
        ADEN = self.air_density
        TCG,TWF = windDrag(TLEN,TBOD,TTOD,WREFS,WREFH,ALPHA,FB,ADEN,GF)
        KGT = TCG+FB+DRAFT
        self.tower_center_of_gravity=TCG 
        self.tower_keel_to_CG = KGT 
        self.tower_wind_force = TWF 

        # RNA
        RDIA = self.rotor_diameter
        RCGX = self.RNA_center_of_gravity_x
        RCGY = self.RNA_center_of_gravity_y
        VOUT = self.cut_out_speed
        KGR = DRAFT+FB+TLEN+RCGY
        RWA = (pi/4.)*RDIA**2
        wind,Ct,thrust = thrust_table(self.example_turbine_size,ADEN,RWA)
        thrust = map(lambda x: (x), thrust);
        Ct = map(lambda x: (x), Ct);
        max_thrust = max(thrust)
        max_index = thrust.index(max_thrust)
        CT = Ct[max_index]
        HH = RCGY+TLEN+FB
        WSPEED = windPowerLaw(WREFS,WREFH,ALPHA,HH)
        if WSPEED < VOUT:
            RWF = 0.5*ADEN*(WSPEED*GF)**2*RWA*CT
        else: 
            RWF = max_thrust*1000*GF**2*0.75
        self.RNA_keel_to_CG = KGR
        self.RNA_wind_force = RWF 