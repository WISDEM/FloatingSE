from openmdao.main.api import Component, Assembly,convert_units
from openmdao.lib.datatypes.api import Float, Array, Str, Int, Bool
from openmdao.lib.drivers.api import SLSQPdriver
import numpy as np
from scipy.optimize import fmin, minimize
from sympy.solvers import solve
from sympy import Symbol
import math
from spar_utils import full_stiffeners_table,thrust_table,plasticityRF,frustumVol,frustumCG,ID,waveProperties,waveU,waveUdot,CD,inertialForce,windPowerLaw,pipeBuoyancy,currentSpeed,rootsearch,bisect,roots,calcPsi,dragForce,curWaveDrag,windDrag,calculateWindCurrentForces
pi=np.pi

class Spar(Component):
    #initial_pass = Bool(True, iotype='in', desc='initial run sets VD to 0 to get other property values')
    #system_acceleration = Float(0.0,iotype='in',units='m',desc = 'system acceleration')
    wall_thickness = Array([0.027,0.025,0.027,0.047],iotype='in', units='m',desc = 'wall thickness of each section')
    number_of_rings = Array([1,4,4,23],iotype='in',desc = 'number of stiffeners in each section')
    neutral_axis = Float(0.285,iotype='in',units='m',desc = 'neutral axis location')
    stiffener_curve_fit = Bool(True, iotype='in', desc='flag for using optimized stiffener dimensions or discrete stiffeners')
    stiffener_index = Int(iotype='in',desc='index of stiffener from filtered table')
    number_of_sections = Int(iotype='in',desc='number of sections in the spar')
    outer_diameter = Array(iotype='in', units='m',desc = 'outer diameter of each section')
    length = Array(iotype='in', units='m',desc = 'wlength of each section')
    end_elevation = Array(iotype='in', units='m',desc = 'end elevation of each section')
    start_elevation = Array(iotype='in', units='m',desc = 'start elevation of each section')
    bulk_head = Array(iotype='in',desc = 'N for none, T for top, B for bottom') 
    # enviroentnment inputs
    gust_factor = Float(1.0,iotype='in', desc='gust factor')
    straight_col_cost = Float(3490.,iotype='in',units='USD',desc='cost of straight columns in $/ton')
    tapered_col_cost = Float(4720.,iotype='in',units='USD',desc='cost of tapered columns in $/ton')
    gravity = Float(9.806,iotype='in', units='m/s**2', desc='gravity')
    air_density = Float(1.198,iotype='in', units='kg/m**3', desc='density of air')
    water_density = Float(1025,iotype='in',units='kg/m**3',desc='density of water')
    water_depth = Float(iotype='in', units='m', desc='water depth')
    load_condition =  Str(iotype='in',desc='Load condition - N for normal or E for extreme')
    significant_wave_height = Float(iotype='in', units='m', desc='significant wave height')
    significant_wave_period = Float(iotype='in', units='m', desc='significant wave period')
    wind_reference_speed = Float(iotype='in', units='m/s', desc='reference wind speed')
    wind_reference_height = Float(iotype='in', units='m', desc='reference height')
    alpha = Float(iotype='in', desc='power law exponent')
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
    


    # from tower_RNA.py
    RNA_keel_to_CG = Float(iotype='in',units='m',desc='RNA keel to center of gravity')
    RNA_mass = Float(iotype='in',units='kg',desc='RNA mass')
    tower_mass = Float(iotype='in',units='kg',desc='tower mass')
    tower_center_of_gravity = Float(iotype='in',units='m',desc='tower center of gravity')
    tower_wind_force = Float(iotype='in',units='N',desc='wind force on tower')
    RNA_wind_force = Float(iotype='in',units='N',desc='wind force on RNA')
    RNA_center_of_gravity_x = Float(iotype='in',units='m',desc='RNA center of gravity in x-direction')

    # from mooring 
    mooring_total_cost = Float(iotype='in',units='USD',desc='total cost for anchor + legs + miscellaneous costs')
    mooring_keel_to_CG = Float(iotype='in',units='N',desc='KGM used in spar.py')
    mooring_vertical_load = Float(iotype='in',units='N',desc='mooring vertical load in all mooring lines')
    mooring_horizontal_stiffness = Float(iotype='in',units='N/m',desc='horizontal stiffness of one single mooring line')
    mooring_vertical_stiffness = Float(iotype='in',units='N/m',desc='vertical stiffness of all mooring lines')
    sum_forces_x = Array(iotype='in',units='N',desc='sume of forces in x direction')
    offset_x = Array(iotype='in',units='m',desc='X offsets in discretization')
    damaged_mooring = Array(iotype='in',units='m',desc='range of damaged mooring')
    intact_mooring = Array(iotype='in',units='m',desc='range of intact mooring')
    # outputs
    flange_compactness = Float(iotype='out',desc = 'check for flange compactness')
    web_compactness = Float(iotype='out',desc = 'check for web compactness')
    VAL = Array(iotype='out',desc = 'unity check for axial load - local buckling')
    VAG = Array(iotype='out',desc = 'unity check for axial load - genenral instability')
    VEL = Array(iotype='out',desc = 'unity check for external pressure - local buckling')
    VEG = Array(iotype='out',desc = 'unity check for external pressure - general instability')
    platform_stability_check = Float(iotype='out',desc = 'check for platform stability')
    heel_angle = Float(iotype='out',desc='heel angle unity check')
    min_offset_unity = Float(iotype='out',desc='minimum offset unity check')
    max_offset_unity = Float(iotype='out',desc='maximum offset unity check')
    spar_and_mooring_cost = Float(iotype='out',units='USD',desc='cost of mooring and spar')
   
    #spar_current_force = Array(iotype='out', units='N',desc = 'SCF by section')
    

    def __init__(self):
        super(Spar,self).__init__()
    def execute(self):
        ''' 
        '''
    

        # assign all varibles so its easier to read later
        G = self.gravity
        ADEN = self.air_density 
        WDEN = self.water_density
        WD = self.water_depth
        LOADC = self.load_condition
        Hs = self.significant_wave_height
        Ts = self.significant_wave_period
        if Hs!= 0: 
            WAVEH = 1.86*Hs
            WAVEP = 0.71*Ts
            WAVEL = G*WAVEP**2/(2*pi)
            WAVEN = 2*pi/WAVEL  
        WREFS = self.wind_reference_speed
        WREFH = self.wind_reference_height
        ALPHA = self.alpha 
        MDEN = self.material_density
        E = self.E
        PR = self.nu
        FY = self.yield_stress
        PBH = self.permanent_ballast_height 
        PBDEN = self.permanent_ballast_density
        FBH = self.fixed_ballast_height
        FBDEN = self.fixed_ballast_density
        RMASS = self.RNA_mass
        KGR = self.RNA_keel_to_CG
        TMASS = self.tower_mass
        TCG = self.tower_center_of_gravity
        TWF = self.tower_wind_force
        RWF = self.RNA_wind_force
        RCGX = self.RNA_center_of_gravity_x
        VTOP = self.mooring_vertical_load
        MHK = self.mooring_horizontal_stiffness
        MVK = self.mooring_vertical_stiffness
        KGM = self.mooring_keel_to_CG
        DRAFT = abs(min(self.end_elevation))
        #FBM = self.fixed_ballast_mass
        #PBM = self.permanent_ballast_mass
        #WBM = self.variable_ballast_mass
        OD = np.array(self.outer_diameter)
        ODB = OD[-1]
        for i in range(0,len(OD)):
            if  self.end_elevation[i] >0:
                ODTW = OD[i+1]
        T = np.array(self.wall_thickness)
        LB = np.array(self.length)
        ELE = np.array(self.end_elevation)
        ELS = np.array(self.start_elevation)
        FB = ELS [0]
        BH = self.bulk_head
        N = np.array(self.number_of_rings)
        NSEC = self.number_of_sections
        if self.stiffener_curve_fit: # curve fits
            YNA=self.neutral_axis
            D = 0.0029+1.3345977*YNA
            IR =0.051*YNA**3.7452
            TW =np.exp(0.88132868+1.0261134*np.log(IR)-3.117086*np.log(YNA))
            AR =np.exp(4.6980391+0.36049717*YNA**0.5-2.2503113/(TW**0.5))
            TFM =1.2122528*YNA**0.13430232*YNA**1.069737
            BF = (0.96105249*TW**-0.59795001*AR**0.73163096)
            IR = 0.47602202*TW**0.99500847*YNA**2.9938134    
        else: # discrete, actual stiffener 
            allStiffeners = full_stiffeners_table()
            stiffener = allStiffeners[self.stiffener_index]
            stiffenerName = stiffener[0]
            AR = convert_units( stiffener[1],'inch**2','m**2')
            D = convert_units(stiffener[2],'inch','m')
            TW = convert_units(stiffener[3],'inch','m')
            BF=  convert_units(stiffener[4],'inch','m')
            TFM = convert_units(stiffener[5],'inch','m')
            YNA = convert_units(stiffener[6],'inch','m')
            self.neutral_axis=YNA
            IR = convert_units(stiffener[7],'inch**4','m**4')
        HW = D - TFM
        SHM,RGM,BHM,SHB,SWF,SWM,SCF,SCM,KCG,KCB=calculateWindCurrentForces(0.,0.,N,AR,BH,OD,NSEC,T,LB,MDEN,DRAFT,ELE,ELS,WDEN,ADEN,G,Hs,Ts,WD,WREFS,WREFH,ALPHA)     
        SHBUOY = sum(SHB)
        SHMASS = sum(SHM)*self.shell_mass_factor
        BHMASS = sum(BHM)*self.bulkhead_mass_factor
        RGMASS = sum(RGM)*self.ring_mass_factor
        SHRMASS = SHMASS + BHMASS + RGMASS
        SHRM = np.array(SHM)+np.array(BHM)+np.array(RGM)
        percent_shell_mass = RGMASS/SHMASS *100. 
        outfitting_mass = SHRMASS*self.outfitting_factor
        SMASS = SHRMASS*self.spar_mass_factor + outfitting_mass
        KCG = np.dot(SHRM,np.array(KCG))/SHRMASS
        KB = np.dot(np.array(SHB),np.array(KCB))/SHBUOY
        BM = ((np.pi/64)*ODTW**4)/(SHBUOY/WDEN)
        SWFORCE = sum(SWF)
        SCFORCE = sum(SCF) # NOTE: inaccurate; reruns later
        BVL = np.pi/4.*(ODB-2*T[-1])**2.  # ballast volume per length
        KGPB = (PBH/2.)+T[-1] 
        PBM = BVL*PBH*PBDEN
        KGFB = (FBH/2.)+PBH+T[-1] 
        FBM = BVL*FBH*FBDEN
        WPA = np.pi/4*(ODTW)**2
        KGT = TCG+FB+DRAFT
        WBM = SHBUOY-SMASS-RMASS-TMASS-VTOP/G-FBM-PBM
        WBH = WBM/(WDEN*BVL)
        KGWB = WBH/2.+PBH+FBH+T[-1]
        KGB = (SMASS*KCG+WBM*KGWB+FBM*KGFB+PBM*KGPB+TMASS*KGT+RMASS*KGR)/(SMASS+WBM+FBM+PBM+TMASS+RMASS)
        KG = (SMASS*KCG+WBM*KGWB+FBM*KGFB+PBM*KGPB+TMASS*KGT+RMASS*KGR+VTOP/G*KGM)/SHBUOY
        GM = KB+BM-KG
        self.platform_stability_check = KG/KB 
        total_mass = SMASS+RMASS+TMASS+WBM+FBM+PBM
        VD = (RWF+TWF+SWFORCE+SCFORCE)/(SMASS+RMASS+TMASS+FBM+PBM+WBM)
        print VD
        SHM,RGM,BHM,SHB,SWF,SWM,SCF,SCM,KCG,KCB=calculateWindCurrentForces(KG,VD,N,AR,BH,OD,NSEC,T,LB,MDEN,DRAFT,ELE,ELS,WDEN,ADEN,G,Hs,Ts,WD,WREFS,WREFH,ALPHA)     
        SCFORCE = sum(SCF)
        # calculate moments 
        RWM = RWF*(KGR-KG)
        TWM = TWF*(KGT-KG)
        columns_mass = sum(SHM[1::2])+sum(RGM[1::2])+sum(BHM[1::2])
        tapered_mass = sum(SHM[0::2])+sum(RGM[0::2])+sum(BHM[0::2])
        COSTCOL = self.straight_col_cost
        COSTTAP = self.tapered_col_cost
        spar_total_cost = COSTCOL*columns_mass/1000. + COSTTAP*tapered_mass/1000.
        self.spar_and_mooring_cost = self.mooring_total_cost + spar_total_cost




        ##### SIZING TAB #####    
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
        I_yaw =  total_mass*(ODB/2.)**2
        # [ADDED MASS]
        surge = (pi/4.)*ODB**2*DRAFT*WDEN
        heave = (1/6.)*WDEN*ODB**3
        pitch = (surge*((KG-DRAFT)-(KB-DRAFT))**2+surge*DRAFT**2/12.)*I_total
        # [OTHER SYSTEM PROPERTIES]
        r_gyration = (I_total/total_mass)**0.5
        CM = (SMASS*KCG+WBM*KGWB+FBM*KGFB+PBM*KGPB)/(SMASS+WBM+FBM+PBM)
        surge_period = 2*pi*((total_mass+surge)/MHK)**0.5
        # [PLATFORM STIFFNESS]
        K33 = WDEN*G*(pi/4.)**ODTW**2+MVK  #heave
        K44 = abs(WDEN*G*((pi/4.)*(ODTW/2.)**4-(KB-KG)*SHBUOY/WDEN)) #roll
        K55 = abs(WDEN*G*((pi/4.)*(ODTW/2.)**4-(KB-KG)*SHBUOY/WDEN)) #pitch
        # 
        T_surge = 2*pi*((total_mass+surge)/MHK)**0.5
        T_heave = 2*pi*((total_mass+heave)/K33)**0.5
        K_pitch = GM*SHBUOY*G
        T_pitch = 2*pi*(pitch/K_pitch)**0.5
        F_total = RWF+TWF+sum(SWF)+sum(SCF)
        #print sum_FX
        sum_FX = self.sum_forces_x
        X_Offset = self.offset_x
        for j in range(1,len(sum_FX)): 
            if sum_FX[j]< (-F_total/1000.): 
                X2 = sum_FX[j]
                X1 = sum_FX[j-1]
                Y2 = X_Offset[j]
                Y1 = X_Offset[j-1]
        max_offset = (Y1+(-F_total/1000.-X1)*(Y2-Y1)/(X2-X1))*self.offset_amplification_factor
        i=0
        while sum_FX[i]> (F_total/1000.):
                i+=1
        min_offset = (X_Offset[i-1]+(F_total/1000.-sum_FX[i-1])*(X_Offset[i]-X_Offset[i-1])/(sum_FX[i]-sum_FX[i-1]))*self.offset_amplification_factor
        M_total = RWM+TWM+sum(SWM)+sum(SCM)+(-F_total*(KGM-KG))+(RMASS*G*-RCGX)
        self.heel_angle = (M_total/K_pitch)*180./pi
        # unity checks! 
        if self.load_condition == 'E': 
            self.max_offset_unity = max_offset/self.damaged_mooring[1]
            self.min_offset_unity = min_offset/self.damaged_mooring[0]
        elif self.load_condition == 'N': 
            self.max_offset_unity = max_offset/self.intact_mooring[1]
            self.min_offset_unity = min_offset/self.intact_mooring[0]

        # shell data
        RO = OD/2.  # outer radius 
        R = RO-T/2. # radius to centerline of wall/mid fiber radius 
        # ring data 
        LR = LB/(N+1.) # number of ring spacing
        #shell and ring data
        RF = RO - HW  # radius to flange
        MX = LR/(R*T)**0.5  # geometry parameter
        # effective width of shell plate in longitudinal direction 
        LE=np.array([0.]*NSEC)
        for i in range(0,NSEC):
            if MX[i] <= 1.56: 
                LE[i]=LR[i]
            else: 
                LE = 1.1*(2*R*T)**0.5+TW 
        # ring properties with effective shell plate
        AER = AR+LE*T  # cross sectional area with effective shell 
        YENA = (LE*T*T/2 + HW*TW*(HW/2+T) + TFM*BF*(TFM/2+HW+T))/AER 
        IER = IR+AR*(YNA+T/2.)**2*LE*T/AER+LE*T**3/12. # moment of inertia
        RC = RO-YENA-T/2. # radius to centroid of ring stiffener 
        # set loads (0 mass loads for external pressure) 
        MBALLAST = PBM + FBM + WBM # sum of all ballast masses
        W = (RMASS + TMASS + MBALLAST + SMASS) * G
        P = WDEN * G* abs(ELE)  # hydrostatic pressure at depth of section bottom 
        if Hs != 0: # dynamic head 
            DH = WAVEH/2*(np.cosh(WAVEN*(WD-abs(ELE)))/np.cosh(WAVEN*WD)) 
        else: 
            DH = 0 
        P = P + WDEN*G*DH # hydrostatic pressure + dynamic head
        GF = self.gust_factor

        
        #-----RING SECTION COMPACTNESS (SECTION 7)-----#
        self.flange_compactness = (0.5*BF/TFM)/(0.375*(E/FY)**0.5)
        self.web_compactness = (HW/TW)/((E/FY)**0.5)
        #-----PLATE AND RING STRESS (SECTION 11)-----#
        # shell hoop stress at ring midway 
        Dc = E*T**3/(12*(1-PR**2))  # parameter D 
        BETAc = (E*T/(4*RO**2*Dc))**0.25 # parameter beta 
        TWS = AR/HW
        dum1 = BETAc*LR
        KT = 8*BETAc**3 * Dc * (np.cosh(dum1) - np.cos(dum1))/ (np.sinh(dum1) + np.sin(dum1))
        KD = E * TWS * (RO**2 - RF**2)/(RO * ((1+PR) * RO**2 + (1-PR) * RF**2))
        dum = dum1/2. 
        PSIK = 2*(np.sin(dum) * np.cosh(dum) + np.cos(dum) * np.sinh(dum)) / (np.sinh(dum1) + np.sin(dum1))
        PSIK = PSIK.clip(min=0) # psik >= 0; set all negative values of psik to zero
        SIGMAXA = -W/(2*pi*R*T)
        PSIGMA = P + (PR*SIGMAXA*T)/RO
        PSIGMA = np.minimum(PSIGMA,P) # PSIGMA has to be <= P
        dum = KD/(KD+KT)
        KTHETAL = 1 - PSIK*PSIGMA/P*dum
        FTHETAS = KTHETAL*P*RO/T
        # shell hoop stress at ring 
        KTHETAG = 1 - (PSIGMA/P*dum)
        FTHETAR = KTHETAG*P*RO/T
        #-----LOCAL BUCKLING (SECTION 4)-----# 
        # axial compression and bending 
        ALPHAXL = 9/(300+(2*R)/T)**0.4
        CXL = (1+(150/((2*R)/T))*(ALPHAXL**2)*(MX**4))**0.5
        FXEL = CXL * (pi**2 * E / (12 * (1 - PR**2))) * (T/LR)**2 # elastic 
        FXCL=np.array(NSEC*[0.])
        for i in range(0,len(FXEL)):
            FXCL[i] = plasticityRF(FXEL[i],FY) # inelastic 
        # external pressure
        BETA = np.array([0.]*NSEC)
        ALPHATHETAL = np.array([0.]*NSEC)
        global ZM
        ZM = 12*(MX**2 * (1-PR**2)**.5)**2/pi**4
        for i in range(0,NSEC):
            f=lambda x:x**2*(1+x**2)**4/(2+3*x**2)-ZM[i]
            ans = roots(f, 0.,15.)
            ans_array = np.asarray(ans)
            is_scalar = False if ans_array.ndim>0 else True
            if is_scalar: 
                BETA[i] = ans 
            else: 
                BETA[i] = float(min(ans_array))
            if MX[i] < 5:
                ALPHATHETAL[i] = 1
            elif MX[i] >= 5:
                ALPHATHETAL[i] = 0.8  
        n = np.round(BETA*pi*R/LR) # solve for smallest whole number n 
        BETA = LR/(pi*R/n)
        left = (1+BETA**2)**2/(0.5+BETA**2)
        right = 0.112*MX**4 / ((1+BETA**2)**2*(0.5+BETA**2))
        CTHETAL = (left + right)*ALPHATHETAL 
        FREL = CTHETAL * pi**2 * E * (T/LR)**2 / (12*(1-PR**2)) # elastic
        FRCL=np.array(NSEC*[0.])
        for i in range(0,len(FREL)):
            FRCL[i] = plasticityRF(FREL[i],FY) # inelastic 
        #-----GENERAL INSTABILITY (SECTION 4)-----# 
        # axial compression and bending 
        AC = AR/(LR*T) # Ar bar 
        ALPHAX = 0.85/(1+0.0025*(OD/T))
        ALPHAXG = np.array([0.]*NSEC)
        for i in range(0,NSEC):
            if AC[i] >= 0.2 :
                ALPHAXG[i] = 0.72
            elif AC[i] > 0.06 and AC[i] <0.2:
                ALPHAXG[i] = (3.6-0.5*ALPHAX[i])*AC[i]+ALPHAX[i]
            else: 
                ALPHAXG[i] = ALPHAX[i]
        FXEG = ALPHAXG * 0.605 * E * T / R * (1 + AC)**0.5 # elastic
        FXCG = np.array(NSEC*[0.])
        for i in range(0,len(FXEG)):
            FXCG[i] = plasticityRF(FXEG[i],FY) # inelastic  
        # external pressure 
        ALPHATHETAG = 0.8
        LAMBDAG = pi * R / LB 
        k = 0.5 
        PEG = np.array([0.]*NSEC)
        for i in range(0,NSEC):
            t = T[i]
            r = R[i]
            lambdag = LAMBDAG[i]
            ier = IER[i]
            rc = RC[i]
            ro = RO[i]
            lr = LR[i]
            def f(x,E,t,r,lambdag,k,ier,rc,ro,lr):
                return E*(t/r)*lambdag**4/((x**2+k*lambdag**2-1)*(x**2+lambdag**2)**2) + E*ier*(x**2-1)/(lr*rc**2*ro)   
            x0 = [2]
            m = float(fmin(f, x0, xtol=1e-3, args=(E,t,r,lambdag,k,ier,rc,ro,lr))) # solve for n that gives minimum P_eG
            PEG[i] = f(m,E,t,r,lambdag,k,ier,rc,ro,lr)
        ALPHATHETAG = 0.8 #adequate for ring stiffeners 
        FREG = ALPHATHETAG*PEG*RO*KTHETAG/T # elastic 
        FRCG = np.array(NSEC*[0.])
        for i in range(0,len(FREG)):
            FRCG[i] = plasticityRF(FREG[i],FY) # inelastic  
        # General Load Case
        NPHI = W/(2*pi*R)
        NTHETA = P * RO 
        K = NPHI/NTHETA 
        #-----Local Buckling (SECTION 6) - Axial Compression and bending-----# 
        C = (FXCL + FRCL)/FY -1
        KPHIL = 1
        CST = K * KPHIL /KTHETAL 
        FTHETACL = np.array([0.]*NSEC)
        bnds = (0,None)
        #print FRCL
        for i in range(0,NSEC):
            cst = CST[i]
            fxcl = FXCL[i]
            frcl = FRCL[i]
            c = C[i]
            x = Symbol('x')
            ans = solve((cst*x/fxcl)**2 - c*(cst*x/fxcl)*(x/frcl) + (x/frcl)**2 - 1, x)
            FTHETACL[i] =  float(min([a for a in ans if a>0]))
        FPHICL = CST*FTHETACL
        #-----General Instability (SECTION 6) - Axial Compression and bending-----# 
        C = (FXCG + FRCG)/FY -1
        KPHIG = 1
        CST = K * KPHIG /KTHETAG 
        FTHETACG = np.array([0.]*NSEC)
        for i in range(0,NSEC):
            cst = CST[i]
            fxcg = FXCG[i]
            frcg = FRCG[i]
            c = C[i]
            x = Symbol('x',real=True)
            ans = solve((cst*x/fxcg)**2 - c*(cst*x/fxcg)*(x/frcg) + (x/frcg)**2 - 1, x)
            FTHETACG[i] =  float(min([a for a in ans if a>0]))
        FPHICG = CST*FTHETACG
        #-----Allowable Stresses-----# 
        # factor of safety
        FOS = 1.25
        if LOADC == 'N' or LOADC == 'n': 
            FOS = 1.65
        FAL = np.array([0.]*NSEC)
        FAG = np.array([0.]*NSEC)
        FEL = np.array([0.]*NSEC)
        FEG = np.array([0.]*NSEC)
        for i in range(0,NSEC):
            # axial load    
            FAL[i] = FPHICL[i]/(FOS*calcPsi(FPHICL[i],FY))
            FAG[i] = FPHICG[i]/(FOS*calcPsi(FPHICG[i],FY))
            # external pressure
            FEL[i] = FTHETACL[i]/(FOS*calcPsi(FTHETACL[i],FY))
            FEG[i] = FTHETACG[i]/(FOS*calcPsi(FTHETACG[i],FY))
        # unity check 
        self.VAL = abs(SIGMAXA / FAL)
        self.VAG = abs(SIGMAXA / FAG)
        self.VEL = abs(FTHETAS / FEL)
        self.VEG = abs(FTHETAS / FEG)
#------------------------------------------------------------------
