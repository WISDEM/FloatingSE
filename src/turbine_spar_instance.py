from floatingse.floatingInstance import NSECTIONS, NPTS, vecOption
from floatingse.sparInstance import SparInstance
from floating_turbine_assembly import FloatingTurbine
from commonse import eps
from rotorse import TURBULENCE_CLASS, TURBINE_CLASS, DRIVETRAIN_TYPE, r_aero
import numpy as np
import offshorebos.wind_obos as wind_obos
import time

NDEL = 0

class FloatingTurbineInstance(SparInstance):
    def __init__(self):
        super(FloatingTurbineInstance, self).__init__()
        
        # Remove what we don't need from Spar
        self.params.pop('rna_cg', None)
        self.params.pop('rna_mass', None)
        self.params.pop('rna_I', None)
        self.params.pop('rna_moment', None)
        self.params.pop('rna_force', None)

        # Environmental parameters
        self.params['air_density'] = self.params['base.windLoads.rho']
        self.params.pop('base.windLoads.rho')
        
        self.params['air_viscosity'] = self.params['base.windLoads.mu']
        self.params.pop('base.windLoads.mu', None)
        
        self.params['water_viscosity'] = self.params['base.waveLoads.mu']
        self.params.pop('base.waveLoads.mu')
        
        self.params['wave_height'] = self.params['hmax']
        self.params.pop('hmax')
        
        self.params['wave_period'] = self.params['T']
        self.params.pop('T', None)
        
        self.params['mean_current_speed'] = self.params['Uc']
        self.params.pop('Uc', None)

        self.params['wind_reference_speed'] = self.params['Uref']
        self.params.pop('Uref', None)
        
        self.params['wind_reference_height'] = self.params['zref']
        self.params.pop('zref')
        
        self.params['shearExp'] = 0.11

        self.params['morison_mass_coefficient'] = self.params['cm']
        self.params.pop('cm', None)
        
        self.params['wind_bottom_height'] = self.params['z0']
        self.params.pop('z0', None)

        self.params['wind_beta'] = self.params['beta']
        self.params.pop('beta', None)
        
        #self.params['wave_beta']            = 0.0

        self.params['hub_height']           = 90.0
        
        self.params['safety_factor_frequency']   = 1.1
        self.params['safety_factor_stress']      = 1.35
        self.params['safety_factor_materials']   = 1.3
        self.params['safety_factor_buckling']    = 1.1
        self.params['safety_factor_fatigue']     = 1.35*1.3*1.0
        self.params['safety_factor_consequence'] = 1.0
        self.params.pop('gamma_f', None)
        self.params.pop('gamma_m', None)
        self.params.pop('gamma_n', None)
        self.params.pop('gamma_b', None)
        self.params.pop('gamma_fatigue', None)
      
        self.params['project_lifetime']   = 20.0
        self.params['number_of_turbines'] = 20
        self.params['annual_opex']        = 7e5
        self.params['fixed_charge_rate']  = 0.12
        self.params['discount_rate']      = 0.07

        # For RotorSE
        self.params['number_of_modes'] = 5
        self.params['hubFraction'] = 0.025
        self.params['bladeLength'] =  61.5
        self.params['r_max_chord'] = 0.23577
        self.params['chord_sub'] = np.array([3.2612, 4.5709, 3.3178, 1.4621])
        self.params['theta_sub'] = np.array([13.2783, 7.46036, 2.89317, -0.0878099])
        self.params['precone'] = 2.5
        self.params['tilt'] = 5.0
        self.params['control:Vin'] = 3.0
        self.params['control:Vout'] = 25.0
        self.params['machine_rating'] = 5e6
        self.params['control:minOmega'] = 0.0
        self.params['control:maxOmega'] = 12.0
        self.params['control:tsr'] = 7.55
        self.params['sparT'] = np.array([0.05, 0.047754, 0.045376, 0.031085, 0.0061398])
        self.params['teT'] = np.array([0.1, 0.09569, 0.06569, 0.02569, 0.00569])

        self.params['idx_cylinder_aero'] = 3
        self.params['idx_cylinder_str'] = 14
        self.params['precurve_sub'] = np.array([0.0, 0.0, 0.0])
        self.params['yaw'] = 0.0
        self.params['nBlades'] = 3
        self.params['turbine_class'] = TURBINE_CLASS['I']
        self.params['turbulence_class'] = TURBULENCE_CLASS['B']
        self.params['drivetrainType'] = DRIVETRAIN_TYPE['GEARED']
        self.params['gust_stddev'] = 3
        self.params['control:pitch'] = 0.0
        self.params['VfactorPC'] = 0.7
        self.params['pitch_extreme'] =  0.0
        self.params['azimuth_extreme'] = 0.0
        self.params['rstar_damage'] = np.linspace(0.0, 1.0, len(r_aero)+1) #np.zeros(len(r_aero)+1)
        self.params['Mxb_damage'] = eps * np.ones(len(r_aero)+1)
        self.params['Myb_damage'] = eps * np.ones(len(r_aero)+1)
        self.params['strain_ult_spar'] = 1e-2
        self.params['strain_ult_te'] = 2*2500*1e-6
        self.params['m_damage'] = 10.0
        self.params['nSector'] = 4
        self.params['tiploss'] = True
        self.params['hubloss'] = True
        self.params['wakerotation'] = True 
        self.params['usecd'] = True
        self.params['AEP_loss_factor'] = 1.0
        self.params['dynamic_amplication_tip_deflection'] = 1.35
        self.params['shape_parameter'] = 0.0

        # For RNA
        self.params['hub_mass'] = 0.1*285599.0
        self.params['nac_mass'] = 0.5*285599.0
        self.params['hub_cm']   = np.array([-1.13197635, 0.0, 0.50875268])
        self.params['nac_cm']   = np.array([-1.13197635, 0.0, 0.50875268])
        self.params['hub_I']    = 0.1*np.array([1.14930678e+08, 2.20354030e+07, 1.87597425e+07, 0.0, 0.0, 5.03710467e+05])
        self.params['nac_I']    = 0.1*np.array([1.14930678e+08, 2.20354030e+07, 1.87597425e+07, 0.0, 0.0, 5.03710467e+05])
        self.params['downwind'] = False
        self.params['rna_weightM'] = True

        # For turbine costs
        self.params['blade_mass_cost_coeff'] = 13.08
        self.params['hub_mass_cost_coeff'] = 3.8
        self.params['pitch_system_mass_cost_coeff'] = 22.91
        self.params['spinner_mass_cost_coeff'] = 23.0
        self.params['lss_mass_cost_coeff'] = 12.6
        self.params['bearings_mass_cost_coeff'] = 6.35
        self.params['gearbox_mass_cost_coeff'] = 17.4
        self.params['high_speed_side_mass_cost_coeff'] = 8.25
        self.params['generator_mass_cost_coeff'] = 17.43
        self.params['bedplate_mass_cost_coeff'] = 4.5
        self.params['yaw_system_mass_cost_coeff'] = 11.01
        self.params['variable_speed_elec_mass_cost_coeff'] = 26.5
        self.params['hydraulic_cooling_mass_cost_coeff'] = 163.95
        self.params['nacelle_cover_mass_cost_coeff'] = 7.61
        self.params['elec_connec_machine_rating_cost_coeff'] = 40.0
        self.params['nacelle_platforms_mass_cost_coeff'] = 8.7
        self.params['base_hardware_cost_coeff'] = 0.7
        self.params['transformer_mass_cost_coeff'] = 26.5
        self.params['tower_mass_cost_coeff'] = 3.20
        self.params['hub_assemblyCostMultiplier'] = 0.0
        self.params['hub_overheadCostMultiplier'] = 0.0
        self.params['nacelle_assemblyCostMultiplier'] = 0.0
        self.params['nacelle_overheadCostMultiplier'] = 0.0
        self.params['tower_assemblyCostMultiplier'] = 0.0
        self.params['tower_overheadCostMultiplier'] = 0.0
        self.params['turbine_assemblyCostMultiplier'] = 0.0
        self.params['turbine_overheadCostMultiplier'] = 0.0
        self.params['hub_profitMultiplier'] = 0.0
        self.params['nacelle_profitMultiplier'] = 0.0
        self.params['tower_profitMultiplier'] = 0.0
        self.params['turbine_profitMultiplier'] = 0.0
        self.params['hub_transportMultiplier'] = 0.0
        self.params['nacelle_transportMultiplier'] = 0.0
        self.params['tower_transportMultiplier'] = 0.0
        self.params['turbine_transportMultiplier'] = 0.0
        self.params['offshore'] = True
        self.params['crane'] = False
        self.params['bearing_number'] = 2
        self.params['bedplate_mass'] = 93090.6
        self.params['controls_cost_base'] = np.array([35000.0,55900.0])
        self.params['controls_esc'] = 1.5
        self.params['crane_cost'] = 0.0
        self.params['elec_connec_cost_esc'] = 1.5
        self.params['gearbox_mass'] = 30237.60
        self.params['generator_mass'] = 16699.85
        self.params['high_speed_side_mass'] = 1492.45
        self.params['hydraulic_cooling_mass'] = 1e3
        self.params['low_speed_shaft_mass'] = 31257.3
        self.params['main_bearing_mass'] = 9731.41 / 2
        self.params['nacelle_cover_mass'] = 1e3
        self.params['nacelle_platforms_mass'] = 1e3
        self.params['pitch_system_mass'] = 17004.0
        self.params['spinner_mass'] = 1810.5
        self.params['transformer_mass'] = 1e3
        self.params['variable_speed_elec_mass'] = 1e3
        self.params['yaw_system_mass'] = 11878.24
        
        # Offshore BOS
        # Turbine / Plant parameters
        self.params['nacelleL'] =                     -np.inf
        self.params['nacelleW'] =                     -np.inf
        self.params['distShore'] =                    90.0
        self.params['distPort'] =                     90.0
        self.params['distPtoA'] =                     90.0
        self.params['distAtoS'] =                     90.0
        self.params['substructure'] =                 wind_obos.Substructure.SPAR
        self.params['anchor'] =                       wind_obos.Anchor.DRAGEMBEDMENT
        self.params['turbInstallMethod'] =            wind_obos.TurbineInstall.INDIVIDUAL
        self.params['towerInstallMethod'] =           wind_obos.TowerInstall.ONEPIECE
        self.params['installStrategy'] =              wind_obos.InstallStrategy.PRIMARYVESSEL
        self.params['cableOptimizer'] =               False
        self.params['buryDepth'] =                    2.0
        self.params['arrayY'] =                       9.0
        self.params['arrayX'] =                       9.0
        self.params['substructCont'] =                0.30
        self.params['turbCont'] =                     0.30
        self.params['elecCont'] =                     0.30
        self.params['interConVolt'] =                 345.0
        self.params['distInterCon'] =                 3.0
        self.params['scrapVal'] =                     0.0
        #General'] = , 
        self.params['inspectClear'] =                 2.0
        self.params['plantComm'] =                    0.01
        self.params['procurement_contingency'] =      0.05
        self.params['install_contingency'] =          0.30
        self.params['construction_insurance'] =       0.01
        self.params['capital_cost_year_0'] =          0.20
        self.params['capital_cost_year_1'] =          0.60
        self.params['capital_cost_year_2'] =          0.10
        self.params['capital_cost_year_3'] =          0.10
        self.params['capital_cost_year_4'] =          0.0
        self.params['capital_cost_year_5'] =          0.0
        self.params['tax_rate'] =                     0.40
        self.params['interest_during_construction'] = 0.08
        #Substructure & Foundation'] = , 
        self.params['mpileCR'] =                      2250.0
        self.params['mtransCR'] =                     3230.0
        self.params['mpileD'] =                       0.0
        self.params['mpileL'] =                       0.0
        self.params['mpEmbedL'] =                     30.0
        self.params['jlatticeCR'] =                   4680.0
        self.params['jtransCR'] =                     4500.0
        self.params['jpileCR'] =                      2250.0
        self.params['jlatticeA'] =                    26.0
        self.params['jpileL'] =                       47.50
        self.params['jpileD'] =                       1.60
        self.params['ssHeaveCR'] =                    6250.0
        self.params['scourMat'] =                     250000.0
        self.params['number_install_seasons'] =       1.0
        #Electrical Infrastructure'] = , 
        self.params['pwrFac'] =                       0.95
        self.params['buryFac'] =                      0.10
        self.params['catLengFac'] =                   0.04
        self.params['exCabFac'] =                     0.10
        self.params['subsTopFab'] =                   14500.0
        self.params['subsTopDes'] =                   4500000.0
        self.params['topAssemblyFac'] =               0.075
        self.params['subsJackCR'] =                   6250.0
        self.params['subsPileCR'] =                   2250.0
        self.params['dynCabFac'] =                    2.0
        self.params['shuntCR'] =                      35000.0
        self.params['highVoltSG'] =                   950000.0
        self.params['medVoltSG'] =                    500000.0
        self.params['backUpGen'] =                    1000000.0
        self.params['workSpace'] =                    2000000.0
        self.params['otherAncillary'] =               3000000.0
        self.params['mptCR'] =                        12500.0
        self.params['arrVoltage'] =                   33.0
        self.params['cab1CR'] =                       185.889
        self.params['cab2CR'] =                       202.788
        self.params['cab1CurrRating'] =               300.0
        self.params['cab2CurrRating'] =               340.0
        self.params['arrCab1Mass'] =                  20.384
        self.params['arrCab2Mass'] =                  21.854
        self.params['cab1TurbInterCR'] =              8410.0
        self.params['cab2TurbInterCR'] =              8615.0
        self.params['cab2SubsInterCR'] =              19815.0
        self.params['expVoltage'] =                   220.0
        self.params['expCurrRating'] =                530.0
        self.params['expCabMass'] =                   71.90
        self.params['expCabCR'] =                     495.411
        self.params['expSubsInterCR'] =               57500.0
        # Vector inputs
        #self.params['arrayCables'] =                  [33, 66]
        #self.params['exportCables'] =                 [132, 220]
        #Assembly & Installation',
        self.params['moorTimeFac'] =                  0.005
        self.params['moorLoadout'] =                  5.0
        self.params['moorSurvey'] =                   4.0
        self.params['prepAA'] =                       168.0
        self.params['prepSpar'] =                     18.0
        self.params['upendSpar'] =                    36.0
        self.params['prepSemi'] =                     12.0
        self.params['turbFasten'] =                   8.0
        self.params['boltTower'] =                    7.0
        self.params['boltNacelle1'] =                 7.0
        self.params['boltNacelle2'] =                 7.0
        self.params['boltNacelle3'] =                 7.0
        self.params['boltBlade1'] =                   3.50
        self.params['boltBlade2'] =                   3.50
        self.params['boltRotor'] =                    7.0
        self.params['vesselPosTurb'] =                2.0
        self.params['vesselPosJack'] =                8.0
        self.params['vesselPosMono'] =                3.0
        self.params['subsVessPos'] =                  6.0
        self.params['monoFasten'] =                   12.0
        self.params['jackFasten'] =                   20.0
        self.params['prepGripperMono'] =              1.50
        self.params['prepGripperJack'] =              8.0
        self.params['placePiles'] =                   12.0
        self.params['prepHamMono'] =                  2.0
        self.params['prepHamJack'] =                  2.0
        self.params['removeHamMono'] =                2.0
        self.params['removeHamJack'] =                4.0
        self.params['placeTemplate'] =                4.0
        self.params['placeJack'] =                    12.0
        self.params['levJack'] =                      24.0
        self.params['hamRate'] =                      20.0
        self.params['placeMP'] =                      3.0
        self.params['instScour'] =                    6.0
        self.params['placeTP'] =                      3.0
        self.params['groutTP'] =                      8.0
        self.params['tpCover'] =                      1.50
        self.params['prepTow'] =                      12.0
        self.params['spMoorCon'] =                    20.0
        self.params['ssMoorCon'] =                    22.0
        self.params['spMoorCheck'] =                  16.0
        self.params['ssMoorCheck'] =                  12.0
        self.params['ssBall'] =                       6.0
        self.params['surfLayRate'] =                  375.0
        self.params['cabPullIn'] =                    5.50
        self.params['cabTerm'] =                      5.50
        self.params['cabLoadout'] =                   14.0
        self.params['buryRate'] =                     125.0
        self.params['subsPullIn'] =                   48.0
        self.params['shorePullIn'] =                  96.0
        self.params['landConstruct'] =                7.0
        self.params['expCabLoad'] =                   24.0
        self.params['subsLoad'] =                     60.0
        self.params['placeTop'] =                     24.0
        self.params['pileSpreadDR'] =                 2500.0
        self.params['pileSpreadMob'] =                750000.0
        self.params['groutSpreadDR'] =                3000.0
        self.params['groutSpreadMob'] =               1000000.0
        self.params['seaSpreadDR'] =                  165000.0
        self.params['seaSpreadMob'] =                 4500000.0
        self.params['compRacks'] =                    1000000.0
        self.params['cabSurveyCR'] =                  240.0
        self.params['cabDrillDist'] =                 500.0
        self.params['cabDrillCR'] =                   3200.0
        self.params['mpvRentalDR'] =                  72000.0
        self.params['diveTeamDR'] =                   3200.0
        self.params['winchDR'] =                      1000.0
        self.params['civilWork'] =                    40000.0
        self.params['elecWork'] =                     25000.0
        #Port & Staging'] = , 
        self.params['nCrane600'] =                    0.0
        self.params['nCrane1000'] =                   0.0
        self.params['crane600DR'] =                   5000.0
        self.params['crane1000DR'] =                  8000.0
        self.params['craneMobDemob'] =                150000.0
        self.params['entranceExitRate'] =             0.525
        self.params['dockRate'] =                     3000.0
        self.params['wharfRate'] =                    2.75
        self.params['laydownCR'] =                    0.25
        #Engineering & Management'] = , 
        self.params['estEnMFac'] =                    0.04
        #Development'] = , 
        self.params['preFEEDStudy'] =                 5000000.0
        self.params['feedStudy'] =                    10000000.0
        self.params['stateLease'] =                   250000.0
        self.params['outConShelfLease'] =             1000000.0
        self.params['saPlan'] =                       500000.0
        self.params['conOpPlan'] =                    1000000.0
        self.params['nepaEisMet'] =                   2000000.0
        self.params['physResStudyMet'] =              1500000.0
        self.params['bioResStudyMet'] =               1500000.0
        self.params['socEconStudyMet'] =              500000.0
        self.params['navStudyMet'] =                  500000.0
        self.params['nepaEisProj'] =                  5000000.0
        self.params['physResStudyProj'] =             500000.0
        self.params['bioResStudyProj'] =              500000.0
        self.params['socEconStudyProj'] =             200000.0
        self.params['navStudyProj'] =                 250000.0
        self.params['coastZoneManAct'] =              100000.0
        self.params['rivsnHarbsAct'] =                100000.0
        self.params['cleanWatAct402'] =               100000.0
        self.params['cleanWatAct404'] =               100000.0
        self.params['faaPlan'] =                      10000.0
        self.params['endSpecAct'] =                   500000.0
        self.params['marMamProtAct'] =                500000.0
        self.params['migBirdAct'] =                   500000.0
        self.params['natHisPresAct'] =                250000.0
        self.params['addLocPerm'] =                   200000.0
        self.params['metTowCR'] =                     11518.0
        self.params['decomDiscRate'] =                0.03

        self.params['dummy_mass'] = eps

        
    def get_assembly(self): return FloatingTurbine(NSECTIONS)
    
    def get_design_variables(self):
        desvarList = super(FloatingTurbineInstance, self).get_design_variables()
        # Make a neat list of design variables, lower bound, upper bound, scalar
        desvarList.extend( [('hub_height', 50.0, 300.0, 1.0),
                            #('hubFraction', 1e-2, 1e-1, 1e2),
                            ('bladeLength', 30.0, 120.0, 1.0),
                            #('r_max_chord', 0.1, 0.7, 10.0),
                            #('control:Vin', 1.0, 20.0, 1.0),
                            #('control:Vout', 20.0, 35.0, 1.0),
                            #('control:ratedPower', 1e6, 15e6, 1e-6),
                            #('control:maxOmega', 1.0, 30.0, 1.0),
                            #('control:tsr', 2.0, 15.0, 1.0),
                            #('sparT', 2e-3, 2e-1, 1e2),
                            #('teT', 2e-3, 2e-1, 1e2)])#,
                            ('chord_sub', 1.0, 5.0, 1.0), #transport widths?
                            ('theta_sub', -30.0, 30.0, 1.0)])
                            #('precone', 0.0, 20.0, 1.0),
                            #('tilt', 0.0, 20.0, 1.0)] )
        return desvarList

    def get_constraints(self):
        conList = super(FloatingTurbineInstance, self).get_constraints()
        for con in conList:
            con[0] = 'sm.' + con[0]

        conList.extend( [['rotor.Pn_margin', None, 1.0, None],
                         ['rotor.P1_margin', None, 1.0, None],
                         ['rotor.Pn_margin_cfem', None, 1.0, None],
                         ['rotor.P1_margin_cfem', None, 1.0, None],
                         ['rotor.rotor_strain_sparU', -1.0, None, None],
                         ['rotor.rotor_strain_sparL', None, 1.0, None],
                         ['rotor.rotor_strain_teU', -1.0, None, None],
                         ['rotor.rotor_strain_teL', None, 1.0, None],
                         ['rotor.rotor_buckling_sparU', None, 1.0, None],
                         ['rotor.rotor_buckling_sparL', None, 1.0, None],
                         ['rotor.rotor_buckling_teU', None, 1.0, None],
                         ['rotor.rotor_buckling_teL', None, 1.0, None],
                         ['rotor.rotor_damage_sparU', None, 0.0, None],
                         ['rotor.rotor_damage_sparL', None, 0.0, None],
                         ['rotor.rotor_damage_teU', None, 0.0, None],
                         ['rotor.rotor_damage_teL', None, 0.0, None],
                         ['tcons.frequency_ratio', None, 1.0, None],
                         ['tcons.tip_deflection_ratio', None, 1.0, None],
                         ['tcons.ground_clearance', 30.0, None, None],
        ])
        return conList

    def add_objective(self):
        # OBJECTIVE FUNCTION: Minimize total cost!
        self.prob.driver.add_objective('lcoe')



def example():
    myspar = FloatingTurbineInstance()
    myspar.evaluate('psqp')
    #myspar.visualize('spar-initial.jpg')
    return myspar

def optimize_spar(algo='slsqp', myspar=None):
    if myspar is None: myspar = FloatingTurbineInstance()
    myspar.run(algo)
    myspar.visualize('fowt-'+algo+'.jpg')
    return myspar

def psqp_optimal():
#OrderedDict([('lcoe', array([86.74566798]))])
    myspar = FloatingTurbineInstance()
    myspar.params['fairlead'] = 10.45427429906716
    myspar.params['base_freeboard'] = 0.012227997729194548
    myspar.params['base_section_height'] = np.array( [24.91986941308374, 28.664570049375737, 17.388103281705487, 5.355696433808805, 1.0089558548858526] )
    myspar.params['base_outer_diameter'] = np.array( [4.3785759779766815, 5.99204398617345, 4.80606316118305, 6.299171122049278, 3.682630947830492, 6.462409714961575] )
    myspar.params['base_wall_thickness'] = np.array( [0.008435355282050833, 0.006620121917047349, 0.011975410531627335, 0.023610485079362177, 0.009793303719532461, 0.014882660379687008] )
    myspar.params['tower_section_height'] = np.array( [17.803070940862685, 22.46714914047011, 12.936887331685712, 23.017810908927416, 19.05371121845179] )
    myspar.params['tower_outer_diameter'] = np.array( [6.516519072879301, 5.857623401370521, 2.348111253776492, 9.077302813606506, 6.479895926823832, 3.978195007348745] )
    myspar.params['tower_wall_thickness'] = np.array( [0.005649107581204494, 0.005000000000000039, 0.007958131761257845, 0.006357541278608118, 0.005000000000000154, 0.004999999999999883] )
    myspar.params['scope_ratio'] = 2.9634135408547366
    myspar.params['anchor_radius'] = 853.9693570359251
    myspar.params['mooring_diameter'] = 0.11731401937770455
    myspar.params['base_permanent_ballast_height'] = 0.8702945199308161
    myspar.params['base_stiffener_web_height'] = np.array( [0.03364944269627522, 0.03422965793306594, 0.0653853009421931, 0.319794684236231, 0.15539731488946043] )
    myspar.params['base_stiffener_web_thickness'] = np.array( [0.0031410970323787653, 0.001570807033502036, 0.0026945342378884725, 0.013347915361981637, 0.12602598634343917] )
    myspar.params['base_stiffener_flange_width'] = np.array( [0.010154439680712016, 0.01466737942654637, 0.014132102720848354, 0.49366066294646616, 0.012242722083984511] )
    myspar.params['base_stiffener_flange_thickness'] = np.array( [0.10512127168828525, 0.01951322988108743, 0.006090457328626164, 0.02533884180364821, 0.008008176039696592] )
    myspar.params['base_stiffener_spacing'] = np.array( [0.18328757651218572, 0.31384436541720717, 0.3244335166913816, 14.50594377648881, 17.132041605435127] )
    myspar.params['hub_height'] = 95.29085753812689
    myspar.params['bladeLength'] = 58.3208292513506
    myspar.params['chord_sub'] = np.array( [2.7676780377901298, 4.847917106306521, 3.3972456158415367, 1.006021002460202] )
    myspar.params['theta_sub'] = np.array( [13.331345934063577, 7.919583245145044, 3.5671272985502065, -0.03498290397940218] )

    myspar.evaluate('psqp')
    #myspar.visualize('fowt-psqp.jpg')
    return myspar

def deriv_check():
   # ----------------
    myspar = FloatingTurbineInstance()
    myspar.evaluate('psqp')
    f = open('deriv_total_spar.dat','w')
    out = myspar.prob.check_total_derivatives(f)#, compact_print=True)
    f.close()
    tol = 1e-4
    for comp in out.keys():
        if ( (out[comp]['rel error'][0] > tol) and (out[comp]['abs error'][0] > tol) ):
            print comp, out[comp]['magnitude'][0], out[comp]['rel error'][0], out[comp]['abs error'][0]

    
if __name__ == '__main__':
    myspar=psqp_optimal()
    myspar=optimize_spar('psqp',myspar)
    #example()
    #deriv_check()
