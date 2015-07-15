import numpy as np
from math import pi, cos, sqrt, radians, sin, exp, log10, log, floor, ceil

import scipy as scp

def ref_table(PTEN,MWPL,S,FH,AE,MBL):
    n = 754.
    max_a =(MBL-MWPL*FH)/MWPL
    INC = max_a/n
    a = np.array(np.linspace(0.0,INC*n,num=n+1))
    H = MWPL*a
    Ttop = H + MWPL*FH
    Vtop = (Ttop**2-H**2)**0.5
    ang = 90. - np.arcsin(H/Ttop)*180/np.pi
    sp_temp = -(S/2.)+(FH/2.)*(1.+(4*a**2/(S**2.-FH**2)))**0.5 
    sp_temp = [0 if i < 0 else i for i in sp_temp]
    # INITIALIZE ARRanchor_yS
    Vbot = np.array([0.]*len(a))
    Tbot = np.array([0.]*len(a))
    Tave = np.array([0.]*len(a))
    stretch = np.array([0.]*len(a))
    x = np.array([0.]*len(a))
    s = np.array([0.]*len(a))
    yp = np.array([0.]*len(a))
    sp = np.array([0.]*len(a))
    for i in range (0,len(a)):
        if sp_temp[i] == 0.: 
            Vbot[i] = 0.
            Tbot[i] = H[i]
            Tave[i] = 0.5*(Ttop[i]+Tbot[i])
            stretch[i] = 1.+Tave[i]/AE
            if i == 0:
                x[i] = S*stretch[i] -FH
            else: 
                x[i] = S*stretch[i-1] -FH*(1.+2.*a[i]/FH)**0.5+a[i]*np.arccosh(1.+FH/a[i])
            s[i] = (FH**2+2.*FH*a[i])**0.5
            sp [i] = 0.
        else: 
            s[i] = S*stretch[i-1]
            Vbot[i] = Vtop[i]-MWPL*s[i]
            Tbot[i] = (H[i]**2+Vbot[i]**2)**0.5
            Tave[i] =  0.5*(Ttop[i]+Tbot[i])
            stretch[i] = 1.+Tave[i]/AE
            sp[i] = sp_temp[i]*stretch[i-1]
            x[i] = a[i]*(np.arcsinh((S+sp[i])/a[i])-np.arcsinh(sp[i]/a[i]))*stretch[i-1]
            
        if i == 0: 
            yp[i] = -a[i]+(a[i]**2+sp[i]**2)**0.5
        else: 
            yp[i] = (-a[i]+(a[i]**2+sp[i]**2)**0.5)*stretch[i-1]
    x0 = np.interp(PTEN,Ttop,x)
    offset = x - x0
    MKH = np.array([0.]*len(a))
    MKV = np.array([0.]*len(a))
    for i in range(1,len(a)):
        MKH[i] = (H[i]-H[i-1])/(offset[i]-offset[i-1])
        MKV[i] = (Vtop[i]-Vtop[i-1])/(offset[i]-offset[i-1])
    return x0,a,x,H,sp,yp,s,Ttop,Vtop,Tbot,Vbot,Tave,stretch,ang,offset,MKH,MKV,INC

def fairlead_anchor_table(NM,direction,FOFF,FD,WD,ODB,x0,NDIS,survival_mooring,Ttop,x,H):
    # fairlead
    fairlead_x = np.array(((ODB/2)+FOFF)*np.cos(direction*np.pi/180.))
    fairlead_y = np.array(((ODB/2)+FOFF)*np.sin(direction*np.pi/180.))
    fairlead_z = np.array([-FD]*len(fairlead_x))
    # anchor 
    anchor_x = np.array(fairlead_x+x0*np.cos(direction*np.pi/180.))
    anchor_y = np.array(fairlead_y+x0*np.sin(direction*np.pi/180.))
    anchor_z = np.array([-WD]*len(anchor_x))
    # delta 
    delta = (survival_mooring[1]-survival_mooring[0])/NDIS
    X_Offset = np.array(np.linspace(survival_mooring[0],survival_mooring[1],num=NDIS+1))
    # initialize arrays
    X_Fairlead = np.empty((1,NDIS+1),dtype=np.object)
    anchor_distance = np.empty((1,NDIS+1),dtype=np.object)
    Ttop_tension = np.empty((1,NDIS+1),dtype=np.object)
    H_Force = np.empty((1,NDIS+1),dtype=np.object)
    FX = np.empty((1,NDIS+1),dtype=np.object)
    FY = np.empty((1,NDIS+1),dtype=np.object)
    sum_FX = np.array([0.]*(NDIS+1))
    stiffness = np.array([0.]*(NDIS+1))
    # fill arrays
    for i in range(0,NDIS+1):
        X_Fairlead[0,i] = np.array(X_Offset[i]+fairlead_x)
        anchor_distance[0,i] = np.array((((X_Fairlead[0,i])-anchor_x)**2+(fairlead_y-anchor_y)**2)**0.5-0.00001)
        Ttop_vect = np.array([0.]*NM)
        H_vect = np.array([0.]*NM)
        for j in range(0,NM):
            Ttop_vect[j]=np.interp(anchor_distance[0,i][j],x,Ttop)/1000.
            H_vect[j]=np.interp(anchor_distance[0,i][j],x,H)/1000.
        Ttop_tension[0,i] = Ttop_vect
        H_Force[0,i] = H_vect
        FX[0,i] = np.array((anchor_x-X_Fairlead[0,i])/anchor_distance[0,i]*H_Force[0,i])
        FY[0,i] = np.array((anchor_y-fairlead_y)/anchor_distance[0,i]*H_Force[0,i])
        sum_FX[i] = sum(FX[0,i])
    for i in range(0,NDIS+1):    
        if i == NDIS:
            stiffness[i] = abs(sum_FX[i]/X_Offset[i])
        else: 
            stiffness[i] = abs((sum_FX[i]-sum_FX[i+1])/(X_Offset[i]-X_Offset[i+1]))    
    FR = np.power((np.power(FY,2)+np.power(FX,2)),0.5)
    # pack some things 
    fairlead_loc = [fairlead_x,fairlead_y,fairlead_z]
    anchor_loc = [anchor_x,anchor_y,anchor_z]
    return fairlead_loc,anchor_loc,X_Offset,X_Fairlead,anchor_distance,Ttop_tension,H_Force,FX,sum_FX,stiffness,FY,FR


    
    
    
