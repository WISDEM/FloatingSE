from floatingse.floatingInstance import NSECTIONS, NPTS, vecOption
from floatingse.semiInstance import SemiInstance
from floating_turbine_assembly import FloatingTurbine
from commonse import eps
from rotorse import TURBULENCE_CLASS, TURBINE_CLASS, DRIVETRAIN_TYPE, r_aero
import numpy as np
import offshorebos.wind_obos as wind_obos
import time

NDEL = 0

class TurbineSemiInstance(SemiInstance):
    def __init__(self):
        super(TurbineSemiInstance, self).__init__()
        
        # Remove what we don't need from Semi
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
        self.params['substructure'] =                 wind_obos.Substructure.SEMISUBMERSIBLE
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
        desvarList = super(TurbineSemiInstance, self).get_design_variables()
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
                            #('teT', 2e-3, 2e-1, 1e2),
                            ('chord_sub', 1.0, 5.0, 1.0), #transport widths?
                            ('theta_sub', -30.0, 30.0, 1.0)])
                            #('precone', 0.0, 20.0, 1.0),
                            #('tilt', 0.0, 20.0, 1.0)] )
        return desvarList

    def get_constraints(self):
        conList = super(TurbineSemiInstance, self).get_constraints()
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
    mysemi = TurbineSemiInstance()
    mysemi.evaluate('psqp')
    #mysemi.visualize('semi-initial.jpg')
    return mysemi

def optimize_semi(algo='slsqp', mysemi=None):
    if mysemi is None: mysemi = TurbineSemiInstance()
    mysemi.run(algo)
    mysemi.visualize('fowt-'+algo+'.jpg')
    return mysemi

def psqp_optimal():
    #OrderedDict([('sm.total_cost', array([0.65987536]))])
    mysemi = TurbineSemiInstance()
    mysemi.params['fairlead'] = 7.183521549430237
    mysemi.params['fairlead_offset_from_shell'] = 4.967469680533604
    mysemi.params['radius_to_auxiliary_column'] = 35.58870759240209
    mysemi.params['base_freeboard'] = 1.8556926622837724
    mysemi.params['base_section_height'] = np.array( [5.758702205499821, 5.932123715473788, 12.649914720165619, 1.1743267198999756, 2.28400750301171] )
    mysemi.params['base_outer_diameter'] = np.array( [2.682869034274697, 4.679698577662294, 2.8368461322486755, 3.9093609084596106, 8.53661518622927, 9.601420790369382] )
    mysemi.params['base_wall_thickness'] = np.array( [0.015610491104533566, 0.014264204669877033, 0.01084396143779099, 0.017048182990643174, 0.016317203123475895, 0.009999999999972966] )
    mysemi.params['auxiliary_freeboard'] = 7.743688638413e-06
    mysemi.params['auxiliary_section_height'] = np.array( [0.3584582528483153, 0.11284845512926432, 4.745281122675717, 2.6052232333567638, 0.1021088647853414] )
    mysemi.params['auxiliary_outer_diameter'] = np.array( [2.805333883415325, 9.193677611995618, 3.7323328062447083, 2.5230079309738342, 7.685288769102364, 27.944181635345167] )
    mysemi.params['auxiliary_wall_thickness'] = np.array( [0.0100000000000001, 0.0226286388652529, 0.010431737725082074, 0.012367006377755127, 0.025137023242421785, 0.033430110352010815] )
    mysemi.params['pontoon_outer_diameter'] = 0.6602057831089426
    mysemi.params['pontoon_wall_thickness'] = 0.010000000000029945
    mysemi.params['base_pontoon_attach_lower'] = -15.760226115905143
    mysemi.params['base_pontoon_attach_upper'] = 1.2519188710694733
    mysemi.params['tower_section_height'] = np.array( [17.655396366000158, 19.226433757973275, 14.23644859388797, 22.784705691594947, 20.972221763589605] )
    mysemi.params['tower_outer_diameter'] = np.array( [9.714088786763828, 7.323184441896651, 3.615403892264461, 11.16592185497034, 6.25464845911022, 9.074488902735874] )
    mysemi.params['tower_wall_thickness'] = np.array( [0.009999999999967563, 0.009999999999992463, 0.009999999999998321, 0.010000000000053395, 0.009999999999958998, 0.010000000000055418] )
    mysemi.params['scope_ratio'] = 4.307472574455454
    mysemi.params['anchor_radius'] = 837.4978564574001
    mysemi.params['mooring_diameter'] = 0.1058895307179274
    mysemi.params['base_stiffener_web_height'] = np.array( [0.27457614479043413, 0.5184059820139102, 0.08751808320279283, 0.11238220311363739, 0.26109531911027145] )
    mysemi.params['base_stiffener_web_thickness'] = np.array( [0.011404004882958604, 0.02153101863763857, 0.00887354561558569, 0.15529418628382105, 0.01736488731387063] )
    mysemi.params['base_stiffener_flange_width'] = np.array( [0.023432167681717464, 0.009999999999933432, 0.047657705072646585, 0.010000000000048408, 0.009999999999882395] )
    mysemi.params['base_stiffener_flange_thickness'] = np.array( [0.06070860998100411, 0.05602708959912581, 0.28499054362624754, 0.1215489209591134, 0.19092489430295656] )
    mysemi.params['base_stiffener_spacing'] = np.array( [4.290802879580099, 1.3325820165443916, 8.055350673864233, 17.074821647983427, 4.016377530800262] )
    mysemi.params['base_permanent_ballast_height'] = 0.7406786845363778
    mysemi.params['auxiliary_stiffener_web_height'] = np.array( [0.010821490056153862, 0.04346822107899336, 0.011388379728794092, 0.038167978545700365, 0.04420615217267322] )
    mysemi.params['auxiliary_stiffener_web_thickness'] = np.array( [0.045577133827427, 0.0016691291238774102, 0.0011970892934217978, 0.06061344218716923, 0.0010000000000044383] )
    mysemi.params['auxiliary_stiffener_flange_width'] = np.array( [0.024622749706955625, 0.010000000000003322, 0.08851884953834328, 0.01584452110848402, 0.009999999999901314] )
    mysemi.params['auxiliary_stiffener_flange_thickness'] = np.array( [0.014425061183055105, 0.02488395486326047, 0.12348274150880165, 0.26410846269870586, 0.4235681536891606] )
    mysemi.params['auxiliary_stiffener_spacing'] = np.array( [1.4261012367607533, 3.569062685370194, 6.1914071400568425, 17.09349416894326, 16.46590146309927] )
    mysemi.params['auxiliary_permanent_ballast_height'] = 0.11460333696924746
    mysemi.params['hub_height'] = 96.73089883532874
    mysemi.params['bladeLength'] = 61.73698189718739
    mysemi.params['chord_sub'] = np.array( [3.348395782771469, 5.000000000000066, 3.0705421226293144, 1.0000000000000553] )
    mysemi.params['theta_sub'] = np.array( [12.309766936805092, 7.045443238837607, -0.36282211475399956, 3.6154561854813005] )
    #OrderedDict([('lcoe', array([42.10756006]))])
    mysemi.evaluate('slsqp')
    #mysemi.visualize('fowt-psqp.jpg')
    return mysemi
    
if __name__ == '__main__':
    mysemi=optimize_semi('psqp')
    #example()
