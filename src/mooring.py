from openmdao.main.api import Component, Assembly,convert_units
from openmdao.lib.datatypes.api import Float, Array, Str, Int, Bool
import numpy as np
from scipy.optimize import fmin, minimize
from sympy.solvers import solve
from sympy import Symbol
import math
from spar_utils import ref_table, fairlead_anchor_table
pi=np.pi

class Mooring(Component):
    #design variables
    fairlead_depth = Float(13.0,iotype='in',units='m',desc = 'fairlead depth')
    scope_ratio = Float(1.5,iotype='in',units='m',desc = 'scope to fairlead height ratio')
    pretension_percent = Float(5.0,iotype='in',desc='Pre-Tension Percentage of MBL (match PreTension)')
    mooring_diameter = Float(0.09,iotype='in',units='m',desc='diameter of mooring chain')
    # inputs 
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
    spar_start_elevation = Array(iotype='in', units='m',desc = 'start elevation of each section')
    spar_end_elevation = Array(iotype='in', units='m',desc = 'end elevation of each section')
    shell_buoyancy = Array(iotype='in', units='kg',desc = 'shell buoyancy by section')
    shell_mass = Array(iotype='in', units='kg',desc = 'shell mass by section')
    bulkhead_mass = Array(iotype='in', units='kg',desc = 'bulkhead mass by section')
    ring_mass = Array(iotype='in', units='kg',desc = 'ring mass by section')
    shell_mass_factor = Float(1.0,iotype='in',desc='shell mass factor')
    bulkhead_mass_factor = Float(1.0,iotype='in',desc='bulkhead mass factor')
    ring_mass_factor = Float(1.0,iotype='in',desc='ring mass factor')
    outfitting_factor = Float(0.06,iotype='in',desc='outfitting factor')
    spar_mass_factor = Float(1.05,iotype='in',desc='spar mass factor')
    spar_keel_to_CG = Float(iotype='in', units='m',desc = 'KCG by section from spar')
    spar_keel_to_CB = Float(iotype='in', units='m',desc = 'KCB by section from spar')
    water_density = Float(1025,iotype='in',units='kg/m**3',desc='density of water')
    spar_outer_diameter = Array(iotype='in',units='m',desc='top outer diameter')
    spar_wind_force = Array(iotype='in', units='N',desc = 'SWF by section from spar')
    spar_wind_moment = Array(iotype='in', units='N*m',desc = 'SWM by section from spar')
    spar_current_force = Array(iotype='in', units='N',desc = 'SCF by section from spar')
    spar_current_moment = Array(iotype='in', units='N*m',desc = 'SCM by section from spar')
    wall_thickness = Array(iotype='in', units='m',desc = 'wall thickness by section from spar')
    permanent_ballast_height = Float(iotype='in',units='m',desc='height of permanent ballast')
    fixed_ballast_height = Float(iotype='in',units='m',desc='height of fixed ballast')
    permanent_ballast_density = Float(iotype='in',units='kg/m**3',desc='density of permanent ballast')
    fixed_ballast_density = Float(iotype='in',units='kg/m**3',desc='density of fixed ballast')
    RNA_mass = Float(iotype='in',units='kg',desc='RNA mass')
    tower_mass = Float(iotype='in',units='kg',desc='tower mass')
    gravity = Float(9.806,iotype='in', units='m/s**2', desc='gravity')
    tower_center_of_gravity = Float(iotype='in',units='m',desc='tower center of gravity')
    RNA_keel_to_CG = Float(iotype='in',units='m',desc='RNA keel to center of gravity')
    tower_wind_force = Float(iotype='in',units='N',desc='wind force on tower')
    tower_wind_moment =Float(iotype='in',units='N*m',desc='wind moment on tower')
    RNA_wind_force = Float(iotype='in',units='N',desc='wind force on RNA')
    RNA_wind_moment = Float(iotype='in',units='N*m',desc='wind moment on tower')
    RNA_center_of_gravity_x = Float(iotype='in', units='m',desc='rotor center of gravity') 
    amplification_factor = Float(1.0,iotype='in',desc='amplification factor for offsets') 
    allowable_heel_angle = Float(6.0,iotype='deg',desc='amplification factor for offsets') 
    load_condition =  Str(iotype='in',desc='Load condition - N for normal or E for extreme')
    # outputs 
    mooring_total_cost = Float(iotype='out',units='USD',desc='total cost for anchor + legs + miscellaneous costs')
    heel_angle_unity = Float(iotype='out',desc='heel angle unity check')
    min_offset_unity = Float(iotype='out',desc='minimum offset unity check')
    max_offset_unity = Float(iotype='out',desc='maximum offset unity check')
    keel_to_CG_operating_system = Float(iotype='out',desc='keel tp venter of gravity of whole system')
    fixed_ballast_mass = Float(iotype='out',units='kg',desc='fixed ballast mass')
    permanen_ballast_mass = Float(iotype='out',units='kg',desc='permanenet ballast mass')
    variable_ballast_mass = Float(iotype='out',units='kg',desc='variable ballast mass')

    def __init__(self):
        super(Mooring,self).__init__()
    def execute(self):
        ##### MOORING TAB ##### 
        WD = self.water_depth
        FD = self.fairlead_depth
        MDIA = self.mooring_diameter
        SR = self.scope_ratio
        PPER = self.pretension_percent
        MTYPE = self.mooring_type
        NM = self.number_of_mooring_lines
        FOFF = self.fairlead_offset_from_shell
        ODB = self.spar_outer_diameter[-1]
        NDIS= self.number_of_discretizations
        WDEN = self.water_density
        DRAFT = abs(min(self.spar_end_elevation))
        RMASS = self.RNA_mass
        SWF = self.spar_wind_force
        SWM = self.spar_wind_moment
        SCF = self.spar_current_force
        SCM = self.spar_current_moment
        FH = WD-FD 
        S = FH*SR
        if MTYPE == 'CHAIN':    
            MBL = 27600.*MDIA**2*(44.-80.*MDIA)*10**3
            WML = 18070.*MDIA**2
            AE_storm = (1.3788*MDIA**2-4.93*MDIA**3)*10**11
            AREA = 2.64*MDIA**2
            MCPL = 0.58*(MBL/1000./9.806)-87.6
        elif MTYPE == 'STRAND':
            MBL = (937600*MDIA**2-1408.3*MDIA)*10**3
            WML = 4110*MDIA**2
            AE_storm = 9.28*MDIA**2*10**10
            AREA = 0.58*MDIA**2
            MCPL = 0.42059603*(MBL/1000./9.806)+109.5
        elif MTYPE == 'IWRC':
            MBL = 648000*MDIA**2*10**3
            WML = 3670*MDIA**2
            AE_storm = 6.01*MDIA**2*10**10
            AREA = 0.54*MDIA**2
            MCPL = 0.33*(MBL/1000./9.806)+139.5
        elif MTYPE == 'FIBER': 
            MBL = (274700*MDIA**2+7953.9*MDIA-879.24)*10**3
            WML = 160.9*MDIA**2+5.522*MDIA-0.04798
            AE_storm = (10120*MDIA**2+320.7*MDIA-35.47)*10**6
            AE_drift = (5156*MDIA**2+142.7*MDIA-16)*10**6
            AREA = (np.pi/4)*MDIA**2
            MCPL = 0.53676471*(MBL/1000./9.806)
        else: 
            print "PLEASE PICK AVAILABLE MOORIN' TYPE M8"
        # if user defined values available
        if self.user_MBL != 0.0:
            MBL = self.user_MBL
        if self.user_WML != 0.0: 
            WML = self.user_WML
        if self.user_AE_storm != 0.0: 
            AE_storm = self.user_AE_storm 
        if self.user_MCPL != 0.0: 
            MCPL = self.user_MCPL
        PTEN = MBL*PPER/100.
        MWPL = WML*9.806
        x0,a,x,H,sp,yp,s,Ttop,Vtop,Tbot,Vbot,Tave,stretch,ang,offset,mkh,mkv,INC= ref_table(PTEN,MWPL,S,FH,AE_storm,MBL)
        if np.interp(PTEN,Ttop,sp)>0.:
            cat_type = 'semi-taut'
        else: 
            cat_type = 'catenary'
        XANG = np.interp(PTEN,Ttop,ang)
        XMAX = max(x)
        TALL = 0.6*MBL 
        XALL = np.interp(TALL,Ttop,x)
        TEXT = 0.8*MBL 
        XEXT = np.interp(TEXT,Ttop,x)
        # OFFSETS 
        direction =  np.array(np.linspace(0.0,360-360/NM,num=NM))
        intact_mooring = [-(XALL-x0), XALL/np.sin(direction[1]/180.*np.pi)*np.sin(np.pi-direction[1]/180*np.pi-np.arcsin(x0/XALL*np.sin(direction[1]/180.*np.pi)))]
        damaged_mooring = [-(XEXT-x0), XEXT/np.sin(direction[1]/180.*np.pi)*np.sin(np.pi-direction[1]/180*np.pi-np.arcsin(x0/XEXT*np.sin(direction[1]/180.*np.pi)))]
        survival_mooring = [-(XMAX-x0), XMAX/np.sin(direction[1]/180.*np.pi)*np.sin(np.pi-direction[1]/180*np.pi-np.arcsin(x0/XMAX*np.sin(direction[1]/180.*np.pi)))]
        fairlead_loc,anchor_loc,X_Offset,X_Fairlead,anchor_distance,Ttop_tension,H_Force,FX,sum_FX,stiffness,FY,FR = fairlead_anchor_table(NM,direction,FOFF,FD,WD,ODB,x0,NDIS,survival_mooring,Ttop,x,H)
        # COST
        each_leg = MCPL*S
        legs_total = each_leg*NM
        if self.anchor_type =='DRAG':
            each_anchor = MBL/1000./9.806/20*2000
        elif self.anchor_type == 'PILE':
            each_anchor = 150000.*(MBL/1000./9.806/1250.)**0.5
        if self.user_anchor_cost != 0.0: 
            each_anchor = self.user_anchor_cost
        anchor_total = each_anchor*NM
        misc_cost = (anchor_total+legs_total)*self.misc_cost_factor/100.
        self.mooring_total_cost = legs_total+anchor_total+misc_cost 
        # initial conditions
        KGM = DRAFT - FD 
        VTOP =  np.interp(PTEN,Ttop,Vtop)*NM
        MHK = np.interp(PTEN,Ttop,mkh)
        MVK = np.interp(PTEN,Ttop,mkv)*NM
        TMM = (WML+np.pi*MDIA**2/4*WDEN)*S*NM

        ###### PLATFORM TAB #####
        for i in range(0,len(self.spar_outer_diameter)):
            if  self.spar_end_elevation[i] >0:
                ODT = self.spar_outer_diameter[i+1]
        T = self.wall_thickness
        FB = self.spar_start_elevation[0]
        SHBUOY = sum(self.shell_buoyancy)
        SHMASS = sum(self.shell_mass)*self.shell_mass_factor
        BHMASS = sum(self.bulkhead_mass)*self.bulkhead_mass_factor
        RGMASS = sum(self.ring_mass)*self.ring_mass_factor
        SHRMASS = SHMASS + BHMASS + RGMASS
        SHRM = np.array(self.shell_mass)+np.array(self.bulkhead_mass)+np.array(self.ring_mass)
        percent_shell_mass = RMASS/SHMASS *100. 
        outfitting_mass = SHRMASS*self.outfitting_factor
        SMASS = SHRMASS*self.spar_mass_factor + outfitting_mass
        KCG = sum(np.dot(SHRM,np.array(self.spar_keel_to_CG)))/SHRMASS
        KB = sum(np.dot(np.array(self.shell_buoyancy),np.array(self.spar_keel_to_CB)))/SHBUOY
        BM = ((np.pi/64)*ODT**4)/(SHBUOY/WDEN)
        SWFORCE = sum(self.spar_wind_force)
        SWMOM = sum(self.spar_wind_moment)
        SCFORCE = sum(self.spar_current_force)
        SCMOM = sum(self.spar_current_moment)
        BVL = np.pi/4.*(ODB-2*T[-1])**2.  # ballast volume per length
        PBH = self.permanent_ballast_height 
        PBDEN = self.permanent_ballast_density
        KGPB = (PBH/2.)+T[-1] 
        PBM = BVL*PBH*PBDEN
        FBH = self.fixed_ballast_height
        FBDEN = self.fixed_ballast_density
        KGFB = (FBH/2.)+PBH+T[-1] 
        FBM = BVL*FBH*FBDEN
        WPA = np.pi/4*(ODT)**2

        ##### SIZING TAB #####
        G = self.gravity
        KGR = self.RNA_keel_to_CG
        TMASS = self.tower_mass
        TCG = self.tower_center_of_gravity
        TWF = self.tower_wind_force
        TWM = self.tower_wind_moment
        RWF = self.RNA_wind_force
        RWM = self.RNA_wind_moment
        RCGX = self.RNA_center_of_gravity_x
        KGT = TCG+FB+DRAFT
        WBM = SHBUOY-SMASS-RMASS-TMASS-VTOP/G-FBM-PBM
        WBH = WBM/(WDEN*BVL)
        KGWB = WBH/2.+PBH+FBH+T[-1]
        KGB = (SMASS*KCG+WBM*KGWB+FBM*KGFB+PBM*KGPB+TMASS*KGT+RMASS*KGR)/(SMASS+WBM+FBM+PBM+TMASS+RMASS)
        KG = (SMASS*KCG+WBM*KGWB+FBM*KGFB+PBM*KGPB+TMASS*KGT+RMASS*KGR+VTOP/G*KGM)/SHBUOY
        GM = KB+BM-KG
        if KG <KB: 
            stability_check = 'pass'
        weight = SMASS+RMASS+TMASS+WBM+FBM+PBM
        # [TOP MASS(RNA+TOWER)]
        top_mass = RMASS+TMASS 
        KG_top = (RMASS*KGR+TMASS*KGT)
        # [INERTIA PROPERTIES - LOCAL]
        I_top_loc = (1./12.)*top_mass*KG_top**2
        I_hull_loc = (1./12.)*SMASS*(DRAFT+FB)**2
        I_WB_loc = (1./12.)*WBM*WBH**2
        I_FB_loc = (1./12.)*FBM*FBH**2
        I_PB_loc = (1./12.)*PBM*PBH**2
        # [INERTIA PROPERTIES - SYSTEM]
        I_top_sys = I_top_loc + top_mass*(KG_top-KG)**2
        I_hull_sys = I_hull_loc + SMASS*(KCG-KG)**2
        I_WB_sys = I_WB_loc + WBM*(KGWB-KGB)**2 
        I_FB_sys = I_FB_loc + FBM*(KGFB-KGB)**2
        I_PB_sys = I_PB_loc + PBM*(KGPB-KGB)**2
        I_total = I_top_sys + I_hull_sys + I_WB_sys + I_FB_sys + I_PB_sys
        I_yaw =  weight*(ODB/2.)**2
        # [ADDED MASS]
        surge = (pi/4.)*ODB**2*DRAFT*WDEN
        heave = (1/6.)*WDEN*ODB**3
        pitch = (surge*((KG-DRAFT)-(KB-DRAFT))**2+surge*DRAFT**2/12.)*I_total
        # [OTHER SYSTEM PROPERTIES]
        r_gyration = (I_total/weight)**0.5
        CM = (SMASS*KCG+WBM*KGWB+FBM*KGFB+PBM*KGPB)/(SMASS+WBM+FBM+PBM)
        surge_period = 2*pi*((weight+surge)/MHK)**0.5
        # [PLATFORM STIFFNESS]
        K33 = WDEN*G*(pi/4.)**ODT**2+MVK  #heave
        K44 = abs(WDEN*G*((pi/4.)*(ODT/2.)**4-(KB-KG)*SHBUOY/WDEN)) #roll
        K55 = abs(WDEN*G*((pi/4.)*(ODT/2.)**4-(KB-KG)*SHBUOY/WDEN)) #pitch
        # 
        T_surge = 2*pi*((weight+surge)/MHK)**0.5
        T_heave = 2*pi*((weight+heave)/K33)**0.5
        K_pitch = GM*SHBUOY*G
        T_pitch = 2*pi*(pitch/K_pitch)**0.5
        F_total = RWF+TWF+sum(SWF)+sum(SCF)
        for j in range(1,len(sum_FX)): 
            if sum_FX[j]< (-F_total/1000.): 
                X2 = sum_FX[j]
                X1 = sum_FX[j-1]
                Y2 = X_Offset[j]
                Y1 = X_Offset[j-1]
        max_offset = (Y1+(-F_total/1000.-X1)*(Y2-Y1)/(X2-X1))*self.amplification_factor
        i=0
        while sum_FX[i]> (F_total/1000.):
                i+=1
        min_offset = (X_Offset[i-1]+(F_total/1000.-sum_FX[i-1])*(X_Offset[i]-X_Offset[i-1])/(sum_FX[i]-sum_FX[i-1]))*self.amplification_factor
        M_total = RWM+TWM+sum(SWM)+sum(SCM)+(-F_total*(KGM-KG))+(RMASS*G*-RCGX)
        heel_angle = (M_total/K_pitch)*180./pi
        # unity checks! 
        self.heel_angle_unity = heel_angle/self.allowable_heel_angle
        if self.load_condition == 'E': 
            self.max_offset_unity = max_offset/damaged_mooring[1]
            self.min_offset_unity = min_offset/damaged_mooring[0]
        elif self.load_condition == 'N': 
            self.max_offset_unity = max_offset/intact_mooring[1]
            self.min_offset_unity = min_offset/intact_mooring[0]
        self.keel_to_CG_operating_system = KG
        self.fixed_ballast_mass = FBM
        self.permanen_ballast_mass = PBM
        self.variable_ballast_mass = WBM



        

        


        







 
