# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 16:27:49 2015

@author: yhuang1
"""

from openmdao.main.api import Component, Assembly,convert_units
from openmdao.lib.datatypes.api import Float, Array, Str, Int, Bool
from openmdao.lib.drivers.api import SLSQPdriver
import numpy as np
from scipy.optimize import fmin, minimize
from sympy.solvers import solve
from sympy import Symbol
import math
from spar_utils import full_stiffeners_table,thrust_table,plasticityRF,frustumVol,frustumCG,ID,waveProperties,waveU,waveUdot,CD,inertialForce,windPowerLaw,pipeBuoyancy,currentSpeed,rootsearch,bisect,roots,calcPsi,dragForce,curWaveDrag,windDrag
pi=np.pi

class Spar(Component):
    # design variables 
    wall_thickness = Array([0.027,0.025,0.027,0.047],iotype='in', units='m',desc = 'wall thickness of each section')
    number_of_rings = Array([1,4,4,23],iotype='in',desc = 'number of stiffeners in each section')
    neutral_axis = Float(0.285,iotype='in',units='m',desc = 'neutral axis location')
    # inputs 
    initial_pass = Bool(True, iotype='in', desc='flag for using optimized stiffener dimensions or discrete stiffeners')
    stiffener_index = Int(iotype='in',desc='index of stiffener from filtered table')
    gravity = Float(9.806,iotype='in', units='m/s**2', desc='gravity')
    air_density = Float(1.198,iotype='in', units='kg/m**3', desc='density of air')
    water_density = Float(1025,iotype='in',units='kg/m**3',desc='density of water')
    water_depth = Float(iotype='in', units='m', desc='water depth')
    load_condition =  Str(iotype='in',desc='Load condition - N for normal or E for extreme')
    significant_wave_height = Float(iotype='in', units='m', desc='significant wave height')
    significant_wave_period = Float(iotype='in', units='m', desc='significant wave period')
    #keel_cg_mooring = Float(iotype='in', units='m', desc='center of gravity above keel of mooring line')
    keel_cg_operating_system = Float(iotype='in', units='m', desc='center of gravity above keel of operating system')
    wind_reference_speed = Float(iotype='in', units='m/s', desc='reference wind speed')
    wind_reference_height = Float(iotype='in', units='m', desc='reference height')
    alpha = Float(iotype='in', desc='power law exponent')
    material_density = Float(7850.,iotype='in', units='kg/m**3', desc='density of spar material')
    E = Float(200.e9,iotype='in', units='Pa', desc='young"s modulus of spar material')
    nu = Float(0.3,iotype='in', desc='poisson"s ratio of spar material')
    yield_stress = Float(345000000.,iotype='in', units='Pa', desc='yield stress of spar material')
    RNA_mass = Float(iotype='in', units='kg', desc='rotor mass')
    tower_mass = Float(iotype='in', units='kg', desc='tower mass')
    #draft = Float(iotype='in', units='m', desc='draft length')
    fixed_ballast_mass = Float(iotype='in', units='kg', desc='fixed ballast mass')
    hull_mass = Float(iotype='in', units='kg', desc='hull mass')
    permanent_ballast_mass = Float(iotype='in', units='kg', desc='permanent ballast mass')
    variable_ballast_mass = Float(iotype='in', units='kg', desc='variable ballast mass')
    number_of_sections = Int(iotype='in',desc='number of sections in the spar')
    outer_diameter = Array(iotype='in', units='m',desc = 'outer diameter of each section')
    length = Array(iotype='in', units='m',desc = 'wlength of each section')
    end_elevation = Array(iotype='in', units='m',desc = 'end elevation of each section')
    start_elevation = Array(iotype='in', units='m',desc = 'start elevation of each section')
    bulk_head = Array(iotype='in',desc = 'N for none, T for top, B for bottom')
    tower_wind_force = Float(iotype='in',units='N',desc='wind force on tower')
    RNA_wind_force = Float(iotype='in',units='N',desc='wind force on RNA')
    #system_acceleration = Float(iotype='in', units='m/s**2', desc='acceleration')
    #tower_base_OD = Float(iotype='in', units='m', desc='outer diameter of tower base')
    #tower_top_OD = Float(iotype='in', units='m', desc='outer diameter of tower top')
    #tower_length = Float(iotype='in', units='m', desc='length of tower')
    gust_factor = Float(iotype='in', desc='gust factor')
    cut_out_speed = Float(iotype='in', units='m/s',desc='cut out speed of turbine')
    
    #rotor_diameter = Float(iotype='in', units='m',desc='rotor diameter')
    # outputs
    flange_compactness = Float(iotype='out',desc = 'check for flange compactness')
    web_compactness = Float(iotype='out',desc = 'check for web compactness')
    VAL = Array(iotype='out',desc = 'unity check for axial load - local buckling')
    VAG = Array(iotype='out',desc = 'unity check for axial load - genenral instability')
    VEL = Array(iotype='out',desc = 'unity check for external pressure - local buckling')
    VEG = Array(iotype='out',desc = 'unity check for external pressure - general instability')
    columns_mass = Float(iotype='out', units='kg',desc='total mass of straight columns/sections')
    tapered_mass = Float(iotype='out', units='kg',desc='total mass od tapered columns/sections')
    shell_buoyancy = Array(iotype='out',units='kg',desc = 'shell shell buoyancy')
    # shell_mass = Float(iotype='out',desc = 'mass of shell')   
    shell_ring_bulkhead_mass = Float(iotype='out',desc = 'mass to be minimized')
    shell_mass = Array(iotype='out', units='kg',desc = 'shell mass by section')
    bulkhead_mass = Array(iotype='out', units='kg',desc = 'bulkhead mass by section')
    ring_mass = Array(iotype='out', units='kg',desc = 'ring mass by section')
    keel_cg_shell = Array(iotype='out', units='m',desc = 'KCG by section')
    keel_cb_shell = Array(iotype='out', units='m',desc = 'KCB by section')
    spar_wind_force = Array(iotype='out', units='N',desc = 'SWF by section')
    spar_wind_moment = Array(iotype='out', units='N*m',desc = 'SWM by section')
    spar_current_force = Array(iotype='out', units='N',desc = 'SCF by section')
    spar_current_moment = Array(iotype='out', units='N*m',desc = 'SCM by section')

    def __init__(self):
        super(Spar,self).__init__()
    def execute(self):
        ''' 
        '''
        def calculateWindCurrentForces (VD): 
            ODT = OD # outer diameter - top
            ODB = np.append(OD[1:NSEC],OD[-1]) # outer diameter - bottom
            WOD = (ODT+ODB)/2 # average outer diameter 
            COD = WOD # center outer diameter
            IDT = ID(ODT,T)
            IDB = ID(ODB,T)
            OV = frustumVol(ODT,ODB,LB)
            IV = frustumVol(IDT,IDB,LB)
            MV = OV - IV # shell volume 
            SHM = MV * MDEN # shell mass 
            SCG = frustumCG(ODB,ODT,LB) # shell center of gravity
            KCG = DRAFT + ELE + SCG # keel to shell center of gravity
            KCB = DRAFT + ELE + SCG
            SHB = OV*WDEN #  outer volume --> displaced water mass
            coneH = np.array([0.]*NSEC)
            for i in range(0,len(LB)):
                if ODB[i]==ODT[i]:
                    coneH[i]=LB[i]
                else:
                    coneH[i] = -LB[i] * (ODB[i]/ODT[i]) / (1 - ODB[i]/ODT[i]) # cone height 
            # initialize arrays 
            SCF = np.array([0.]*NSEC)
            KCS = np.array([0.]*NSEC)
            SCM = np.array([0.]*NSEC)
            SWF = np.array([0.]*NSEC)
            KWS = np.array([0.]*NSEC)
            SWM = np.array([0.]*NSEC)
            BHM = np.array([0.]*NSEC)
            RGM = np.array([0.]*NSEC)
            for i in range (0,NSEC): 
            #for i in range(0,NSEC) :  # iterate through the sections
                if i <(NSEC-1) : # everything but the last section 
                    if ELE[i] <0 : # if bottom end underwater
                        HWL = abs(ELE[i]) 
                        if HWL >= LB[i]: # COMPLETELY UNDERWATER 
                            
                            SCF[i] = curWaveDrag(Hs,Ts,WD,OD[i],OD[i+1],ELS[i],LB[i],SCG[i],VD,G,WDEN)
                            KCS[i] = KCB[i]
                            if SCF[i] == 0: 
                                KCS[i] = 0.
                            SCM[i] = SCF[i] * (KCS[i]-KCGO)
                            SWF[i] = 0.
                            KWS[i] = 0.
                            SWM[i] = 0.
                        else: # PARTIALLY UNDER WATER 
                            if ODT[i] == ODB[i]:  # cylinder
                                SHB[i] = pipeBuoyancy(ODT[i],WDEN)*HWL # redefine
                                SCG[i] = HWL/2 # redefine
                                KCB[i] = DRAFT + ELE[i] + SCG[i] # redefine 
                                ODW = ODT[i] # assign single variable
                            else: # frustum
                                ODW = ODB[i]*(coneH[i]-HWL)/coneH[i] # assign single variable 
                                WOD[i] = (ODT[i]+ODW[i])/2  # redefine 
                                COD[i] = (ODW[i]+ODB[i])/2 # redefine 
                                SHB[i] = frustumVol(ODW[i],ODB[i],HWL) # redefine 
                                SCG[i] = frustumCG(ODW,ODB[i],HWL) # redefine 
                                KCB[i] = DRAFT + ELE[i] + SCG[i] # redefine 
                            
                            SCF[i] = curWaveDrag(Hs,Ts,WD,ODW,ODB[i],0.,HWL,SCG[i],VD,G,WDEN)
                            KCS[i] = KCB[i]
                            if SCF[i] == 0 : 
                                KCS[i] = 0.
                            SCM[i] = SCF[i]*(KCS[i]-KCGO)
                            if WREFS != 0: # if there is wind 
                                WSPEED = windPowerLaw(WREFS,WREFH,ALPHA,(LB[i]-HWL)/2) # assign single variable 
                                CDW = CD(WSPEED,WOD[i],ADEN) # assign single variable 
                                SWF[i] = dragForce(WOD[i],CDW,LB[i]-HWL,WSPEED,ADEN)
                                KWS[i]= KCG[i]
                                SWM[i] = SWF[i]*(KWS[i]-KCGO) 
                            else: # no wind 
                                SWF[i] = 0.
                                KWS[i] = 0.
                                SWM[i] = 0.
                    else: # FULLY ABOVE WATER 
                        SHB[i] = 0. # redefines 
                        KCB[i] = 0.
                        SCF[i] = 0.
                        KCS[i] = 0.
                        SCM[i] = 0.
                        if WREFS != 0: # if there is wind 
                            WSPEED = windPowerLaw(WREFS,WREFH,ALPHA,ELE[i]+LB[i]/2) # assign single variable 
                            CDW = CD(WSPEED,WOD[i],ADEN) # assign single variable 
                            SWF[i] = dragForce(WOD[i],CDW,LB[i],WSPEED,ADEN)
                            KWS[i]= KCG[i]
                            SWM[i] = SWF[i]*(KWS[i]-KCGO) 
                        else: # no wind 
                            SWF[i] = 0.
                            KWS[i] = 0.
                            SWM[i] = 0.
                    RGM[i] = N[i]*(pi*ID(WOD[i],T[i])*AR)*MDEN # ring mass
                else: # last section 
                    # SHM unchanged 
                    KCG[i] = DRAFT + ELE[i] + LB[i]/2  #redefine
                    # SHB already calculated 
                    # KCB already calculated 
                   
                    SCF[i] = curWaveDrag(Hs,Ts,WD,OD[i],OD[i],ELS[i],LB[i],KCG[i],VD,G,WDEN)
                    KCS[i] = KCG [i] # redefines
                    if SCF[i] ==0: 
                        KCS[i] = 0.
                    SCM[i] = SCF[i]*(KCS[i]-KCGO)
                    SWF[i] = 0.
                    KWS[i] = 0.
                    SWM[i] = 0.
                    RGM[i] = N[i]*(pi*ID(OD[i],T[i])*AR)*MDEN # ring mass
                if BH[i] == 'T':
                    BHM[i] = pi / 4 * IDT[i]**2 * T[i] * MDEN 
                    KCG[i] = (KCG[i] * SHM[i] + (DRAFT + ELS[i] - 0.5 * T[i]) * BHM[i]) / (SHM[i] + BHM[i])
                elif BH[i] == 'B' :
                    BHM[i] = pi / 4 * IDB[i]**2 * T[i] * MDEN
                    KCG[i] = (KCG[i] * SHM[i] + (DRAFT + ELE[i] + 0.5 * T[i]) * BHM[i]) / (SHM[i] + BHM[i])
                else: 
                    KCG[i] = KCG[i]
            #total_mass = sum(SHM)+sum(RGM)+sum(BHM)
            #TCG,TWF = windDrag(TLEN,TBOD,TTOD,WREFS,WREFH,ALPHA,FB,ADEN,GF)
            TWF = self.tower_wind_force
            RWF = self.RNA_wind_force
            VD = (RWF+TWF+sum(SWF)+sum(SCF))/(SMASS+RMASS+TMASS+FBM+PBM+WBM)
            return (VD,SHM,RGM,BHM,SHB,SWF,SWM,SCF,SCM,KCG,KCB)
        
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
        #KCG = self.keel_cg_mooring
        KCGO = self.keel_cg_operating_system
        WREFS = self.wind_reference_speed
        WREFH = self.wind_reference_height
        ALPHA = self.alpha 
        MDEN = self.material_density
        E = self.E
        PR = self.nu
        FY = self.yield_stress
        RMASS = self.RNA_mass
        TMASS = self.tower_mass
        DRAFT = abs(min(self.end_elevation))
        FBM = self.fixed_ballast_mass
        PBM = self.permanent_ballast_mass
        OD = np.array(self.outer_diameter)
        T = np.array(self.wall_thickness)
        LB = np.array(self.length)
        ELE = np.array(self.end_elevation)
        ELS = np.array(self.start_elevation)
        FB = ELS [0]
        BH = self.bulk_head
        N = np.array(self.number_of_rings)
        NSEC = self.number_of_sections
        if self.initial_pass: # curve fits
            YNA=self.neutral_axis
            D = 0.0029+1.3345977*YNA
            IR =0.051*YNA**3.7452
            TW =np.exp(0.88132868+1.0261134*np.log(IR)-3.117086*np.log(YNA))
            AR =np.exp(4.6980391+0.36049717*YNA**0.5-2.2503113/(TW**0.5))
            TFM =1.2122528*YNA**0.13430232*YNA**1.069737
            BF = (0.96105249*TW**-0.59795001*AR**0.73163096)
            IR = 0.47602202*TW**0.99500847*YNA**2.9938134
            SMASS = self.hull_mass
        else: # discrete, actual stiffener
            SMASS = 1.11*self.shell_ring_bulkhead_mass
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
        WBM = self.variable_ballast_mass
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
        MTURBINE = RMASS 
        MTOWER = TMASS 
        MBALLAST = PBM + FBM + WBM # sum of all ballast masses
        MHULL = SMASS 
        W = (MTURBINE + MTOWER + MBALLAST + MHULL) * G
        P = WDEN * G* abs(ELE)  # hydrostatic pressure at depth of section bottom 
        if Hs != 0: # dynamic head 
            DH = WAVEH/2*(np.cosh(WAVEN*(WD-abs(ELE)))/np.cosh(WAVEN*WD)) 
        else: 
            DH = 0 
        P = P + WDEN*G*DH # hydrostatic pressure + dynamic head
        #TBOD = self.tower_base_OD
        #TTOD = self.tower_top_OD
        #TLEN = self.tower_length
        GF = self.gust_factor
        #TCG = (TLEN/4.)*(((TBOD/2.)**2+2.*(TBOD/2.)*(TTOD/2.)+3.*(TTOD/2.)**2.)/((TBOD/2.)**2+(TBOD/2.)*(TTOD/2.)+(TTOD/2.)**2))
        VOUT = self.cut_out_speed
        #TSIZE = self.turbine_size
        
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
       

        VD,SHM,RGM,BHM,SHB,SWF,SWM,SCF,SCM,KCG,KCB=calculateWindCurrentForces(0.)
        VD_unused,SHM,RGM,BHM,SHB,SWF,SWM,SCF,SCM,KCG,KCB=calculateWindCurrentForces(VD)
        

        self.shell_mass = SHM 
        self.ring_mass = RGM 
        self.bulkhead_mass = BHM 
        self.shell_buoyancy = SHB 
        self.spar_wind_force = SWF
        self.spar_wind_moment = SWM
        self.spar_current_force = SCF 
        self.spar_current_moment = SCM 
        self.keel_cg_shell = KCG 
        self.keel_cb_shell = KCB
        self.shell_ring_bulkhead_mass = sum(SHM)+sum(RGM)+sum(BHM)
        self.columns_mass = sum(SHM[1::2])+sum(RGM[1::2])+sum(BHM[1::2])
        self.tapered_mass = sum(SHM[0::2])+sum(RGM[0::2])+sum(BHM[0::2])
        
        print self.wall_thickness
        print self.number_of_rings
        print self.neutral_axis
        print self.VAL
        print self.VAG
        print self.VEL
        print self.VEG
        print self.web_compactness
        print self.flange_compactness
        print self.shell_ring_bulkhead_mass
        print self.columns_mass
        print self.tapered_mass
#------------------------------------------------------------------
