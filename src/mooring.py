from openmdao.main.api import Component, Assembly,convert_units
from openmdao.lib.datatypes.api import Float, Array, Str, Int, Bool
import numpy as np
from scipy.optimize import fmin, minimize
from sympy.solvers import solve
from sympy import Symbol
import math
from mooring_utils import ref_table, fairlead_anchor_table
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
	anchor_type = Str('PILE',,iotype='in',desc='PILE or DRAG')
	fairlead_offset_from_shell = Float(0.5,iotype='in',units='m',desc='fairlead offset from shell')
	outer_diameter_base = Float(iotype='in',units='m',desc='base outer diameter')
	user_MBL = Float(0.0,iotype='in',units='N',desc='user defined minimum breaking load ')
	user_WML = Float(0.0,iotype='in',units='kg/m',desc='user defined wet mass/length')
	user_AE_storm = Float(0.0,iotype='in',units='Pa',desc='user defined E modulus')
	user_MCPL = Float(0.0,iotype='in',units='USD/m',desc='user defined mooring cost per length')
	user_anchor_cost = Float(0.0,iotype='in',units='USD',desc='user defined cost per anchor')
	misc_cost_factor = Float(10.0,iotype='in',desc='miscellaneous cost factor in percent')
	number_of_segments = Int(20,iotype='in',desc='number of segments for mooring discretization')
	# outputs 
	total_cost = Float(iotype='in',units='USD',desc='total cost for anchor + legs + miscellaneous costs')
	def __init__(self):
        super(Mooring,self).__init__()
    def execute(self):
    	WD = self.water_depth
    	FD = self.fairlead_depth
    	MDIA = self.mooring_diameter
    	SR = self.scope_ratio
		PPER = self.pretension_percent
		MTYPE = self.mooring_type
		NM = self.number_of_mooring_lines
		FOFF = self.fairlead_offset_from_shell
		ODB = self.outer_diameter_base
		NSEG= self.number_of_segments
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
		x0,a,x,H,sp,yp,s,Ttop,Vtop,Tbot,Vbot,Tave,stretch,ang,offset,MKH,MKV = ref_table(PTEN,MWPL,S,FH,AE_storm,MBL)
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
		direction, FX,FY,FZ, AX,AY,AZ= fairlead_anchor_table(NM,FOFF,FD,WD,ODB,x0)
		# OFFSETS 
		intact_mooring = [-(XALL-x0), XALL/np.sin(direction[1]/180.*np.pi)*np.sin(np.pi-direction[1]/180*np.pi-np.arcsin(x0/XALL*np.sin(direction[1]/180.*np.pi)))]
		damaged_mooring = [-(XEXT-x0), XEXT/np.sin(direction[1]/180.*np.pi)*np.sin(np.pi-direction[1]/180*np.pi-np.arcsin(x0/XEXT*np.sin(direction[1]/180.*np.pi)))]
		survival_mooring = [-(XMAX-x0), XMAX/np.sin(direction[1]/180.*np.pi)*np.sin(np.pi-direction[1]/180*np.pi-np.arcsin(x0/XMAX*np.sin(direction[1]/180.*np.pi)))]
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
		self.total_cost = legs_total+anchor_total+misc_cost 
		# INITIAL CONDITION 
		KGM = DRAFT - FD 
		VTOP 
		MHK 
		MVK 
		TMM 

		

		


		







 
	