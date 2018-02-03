from floatingse.floatingInstance import NSECTIONS, NPTS, vecOption
from floatingse.semiInstance import SemiInstance
from floating_turbine_assembly import FloatingTurbine
from commonse import eps
from rotorse import TURBULENCE_CLASS, TURBINE_CLASS, DRIVETRAIN_TYPE, r_aero
import numpy as np
import offshorebos.wind_obos as wind_obos
import time

NDEL = 0

class FloatingTurbineInstance(SemiInstance):
    def __init__(self):
        super(FloatingTurbineInstance, self).__init__()
        
        # Remove what we don't need from Semi
        self.params.pop('tower_metric', None)
        self.params.pop('min_d_to_t', None)
        self.params.pop('min_taper', None)
        self.params.pop('turbine_surge_force', None)
        self.params.pop('turbine_pitch_moment', None)
        self.params.pop('turbine_center_of_gravity', None)
        self.params.pop('turbine_mass', None)

        # For RotorSE
        self.params['hubFraction'] = 0.025
        self.params['bladeLength'] =  61.5
        self.params['r_max_chord'] = 0.23577
        self.params['chord_sub'] = np.array([3.2612, 4.5709, 3.3178, 1.4621])
        self.params['theta_sub'] = np.array([13.2783, 7.46036, 2.89317, -0.0878099])
        self.params['precone'] = 2.5
        self.params['tilt'] = 5.0
        self.params['control:Vin'] = 3.0
        self.params['control:Vout'] = 25.0
        self.params['control:ratedPower'] = 5e6
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
        #self.params['cdf_reference_height_wind_speed'] = 90.0 
        self.params['control:pitch'] = 0.0
        self.params['VfactorPC'] = 0.7
        self.params['pitch_extreme'] =  0.0
        self.params['azimuth_extreme'] = 0.0
        self.params['rstar_damage'] = np.zeros(len(r_aero)+1)
        self.params['Mxb_damage'] = np.zeros(len(r_aero)+1)
        self.params['Myb_damage'] = np.zeros(len(r_aero)+1)
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
        # TODO
        self.params['rotor.wind.z'] = np.array([90.0])

        
        # For TowerSE
        self.params['hub_height']           = 90.0
        self.params['cd_usr']               = np.inf
        self.params['wind_bottom_height']   = 0.0
        self.params['wind_beta']            = 0.0
        self.params['wave_beta']            = 0.0
        self.params['wave_velocity_z0']     = 0.0
        self.params['wave_acceleration_z0'] = 0.0 
        self.params['z_depth']              = -self.params['water_depth']

        self.params['safety_factor_stress']      = 1.35
        self.params['safety_factor_materials']   = 1.3
        self.params['safety_factor_buckling']    = 1.1
        self.params['safety_factor_fatigue']     = 1.35*1.3*1.0
        self.params['safety_factor_consequence'] = 1.0
        
        self.params['rna_mass']   = 285599.0
        self.params['rna_F']      = np.array([1284744.19620519, 0.0, -2914124.84400512])
        self.params['rna_M']      = np.array([3963732.76208099, -2275104.79420872, -346781.68192839])
        self.params['rna_Ixx']    = 1.14930678e+08
        self.params['rna_Iyy']    = 2.20354030e+07
        self.params['rna_Izz']    = 1.87597425e+07
        self.params['rna_Ixy']    = 0.0
        self.params['rna_Iyz']    = 0.0
        self.params['rna_Ixz']    = 5.03710467e+05
        self.params['rna_offset'] = np.array([-1.13197635, 0.0, 0.50875268])
        
        self.params['tower_diameter']          = vecOption(6.5, NSECTIONS+1)
        self.params['tower_section_height']    = vecOption(87.6/NSECTIONS, NSECTIONS)
        self.params['tower_wall_thickness']    = vecOption(0.05, NSECTIONS+1)
        self.params['tower_buckling_length']   = 30.0
        self.params['tower_outfitting_factor'] = 1.07
        self.params['tower_force_discretization'] = 5.0

        self.params['tower_M_DEL']                    = np.zeros(NDEL)
        self.params['tower_z_DEL']                    = np.zeros(NDEL)
        self.params['stress_standard_value']          = 80.0
        self.params['frame3dd_matrix_method']         = 1
        self.params['compute_stiffnes']               = False
        self.params['slope_SN']                       = 4
        self.params['number_of_modes']                = 5
        self.params['compute_shear']                  = True
        self.params['frame3dd_convergence_tolerance'] = 1e-9
        self.params['lumped_mass_matrix']             = 0
        self.params['shift_value']                    = 0.0
        
        
        self.params['project_lifetime']   = 20.0
        self.params['number_of_turbines'] = 20
        self.params['annual_opex']        = 7e5
        self.params['fixed_charge_rate']  = 0.12
        self.params['discount_rate']      = 0.07
        
        # Offshore BOS
        # Turbine / Plant parameters
        self.params['turbCapEx'] =                    1605.0
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

        
    def get_assembly(self): return FloatingTurbine(NSECTIONS, NPTS, NDEL)
    
    def get_design_variables(self):
        # Make a neat list of design variables, lower bound, upper bound, scalar
        desvarList = [('fairlead',0.0, 100.0, 1.0),
                      ('fairlead_offset_from_shell',0.0, 5.0, 1e2),
                      ('radius_to_ballast_cylinder',0.0, 40.0, 1.0),
                      ('freeboard_base',0.0, 50.0, 1.0),
                      ('section_height_base',1e-1, 100.0, 1e1),
                      ('outer_diameter_base',1.1, 40.0, 10.0),
                      ('wall_thickness_base',5e-3, 1.0, 1e3),
                      ('freeboard_ballast',0.0, 50.0, 1.0),
                      ('section_height_ballast',1e-1, 100.0, 1e1),
                      ('outer_diameter_ballast',1.1, 40.0, 10.0),
                      ('wall_thickness_ballast',5e-3, 1.0, 1e3),
                      ('pontoon_outer_diameter', 0.1, 10.0, 10.0),
                      ('pontoon_inner_diameter', 0.02, 9.9, 10.0),
                      ('tower_diameter', 3.0, 30.0, 1.0),
                      ('tower_wall_thickness', 0.002, 1.0, 100.0),
                      ('hub_height', 50.0, 300.0, 1.0),
                      ('scope_ratio', 1.0, 5.0, 1.0),
                      ('anchor_radius', 1.0, 1e3, 1e-2),
                      ('mooring_diameter', 0.05, 1.0, 1e1),
                      ('stiffener_web_height_base', 1e-2, 1.0, 1e2),
                      ('stiffener_web_thickness_base', 1e-3, 5e-1, 1e2),
                      ('stiffener_flange_width_base', 1e-2, 5.0, 1e2),
                      ('stiffener_flange_thickness_base', 1e-3, 5e-1, 1e2),
                      ('stiffener_spacing_base', 1e-1, 1e2, 1e1),
                      ('permanent_ballast_height_base', 1e-1, 50.0, 1.0),
                      ('stiffener_web_height_ballast', 1e-2, 1.0, 1e2),
                      ('stiffener_web_thickness_ballast', 1e-3, 5e-1, 1e2),
                      ('stiffener_flange_width_ballast', 1e-2, 5.0, 1e2),
                      ('stiffener_flange_thickness_ballast', 1e-3, 5e-1, 1e2),
                      ('stiffener_spacing_ballast', 1e-1, 1e2, 1e1),
                      ('permanent_ballast_height_ballast', 1e-1, 50.0, 1.0)]

        # TODO: Integer and Boolean design variables
        #prob.driver.add_desvar('number_of_ballast_columns', lower=1)
        #prob.driver.add_desvar('number_of_mooring_lines', lower=1)
        #prob.driver.add_desvar('mooring_type')
        #prob.driver.add_desvar('anchor_type')
        #prob.driver.add_desvar('bulkhead_nodes')
        #prob.driver.add_desvar('cross_attachment_pontoons')
        #prob.driver.add_desvar('lower_attachment_pontoons')
        #prob.driver.add_desvar('upper_attachment_pontoons')
        #prob.driver.add_desvar('lower_ring_pontoons')
        #prob.driver.add_desvar('upper_ring_pontoons')
        return desvarList

    def add_constraints_objective(self):

        # CONSTRAINTS
        # These are mostly the outputs that were not connected to another model

        # Ensure that draft is greater than 0 (spar length>0) and that less than water depth
        # Ensure that fairlead attaches to draft
        self.prob.driver.add_constraint('sm.geomBase.draft_depth_ratio',lower=0.0, upper=0.75)
        self.prob.driver.add_constraint('sm.geomBall.draft_depth_ratio',lower=0.0, upper=0.75)
        self.prob.driver.add_constraint('sm.geomBall.fairlead_draft_ratio',lower=0.0, upper=1.0)
        self.prob.driver.add_constraint('sm.sg.base_ballast_spacing',lower=0.0, upper=1.0)

        # Ensure that the radius doesn't change dramatically over a section
        self.prob.driver.add_constraint('sm.gcBase.manufacturability',upper=0.0)
        self.prob.driver.add_constraint('sm.gcBase.weldability',upper=0.0)
        self.prob.driver.add_constraint('sm.gcBall.manufacturability',upper=0.0)
        self.prob.driver.add_constraint('sm.gcBall.weldability',upper=0.0)

        # Tower related constraints
        self.prob.driver.add_constraint('tow.manufacturability',upper=0.0)
        self.prob.driver.add_constraint('tow.weldability',upper=0.0)
        self.prob.driver.add_constraint('tow.tower.stress',upper=1.0)
        self.prob.driver.add_constraint('tow.tower.global_buckling',upper=1.0)
        self.prob.driver.add_constraint('tow.tower.shell_buckling',upper=1.0)
        #self.prob.driver.add_constraint('tow.tower.damage',upper=1.0)
        
        # Ensure that the spar top matches the tower base
        self.prob.driver.add_constraint('sm.tt.transition_buffer',lower=0.0, upper=5.0)
        
        # Ensure max mooring line tension is less than X% of MBL: 60% for intact mooring, 80% for damanged
        self.prob.driver.add_constraint('sm.mm.safety_factor',lower=0.0, upper=0.8)

        # Ensure there is sufficient mooring line length, MAP doesn't give an error about this
        self.prob.driver.add_constraint('sm.mm.mooring_length_min',lower=1.0)
        self.prob.driver.add_constraint('sm.mm.mooring_length_max',upper=1.0)

        # API Bulletin 2U constraints
        self.prob.driver.add_constraint('sm.base.flange_spacing_ratio', upper=0.5)
        self.prob.driver.add_constraint('sm.base.web_radius_ratio', upper=0.5)
        self.prob.driver.add_constraint('sm.base.flange_compactness', lower=1.0)
        self.prob.driver.add_constraint('sm.base.web_compactness', lower=1.0)
        self.prob.driver.add_constraint('sm.base.axial_local_unity', upper=1.0)
        self.prob.driver.add_constraint('sm.base.axial_general_unity', upper=1.0)
        self.prob.driver.add_constraint('sm.base.external_local_unity', upper=1.0)
        self.prob.driver.add_constraint('sm.base.external_general_unity', upper=1.0)

        self.prob.driver.add_constraint('sm.ball.flange_spacing_ratio', upper=0.5)
        self.prob.driver.add_constraint('sm.ball.web_radius_ratio', upper=0.5)
        self.prob.driver.add_constraint('sm.ball.flange_compactness', lower=1.0)
        self.prob.driver.add_constraint('sm.ball.web_compactness', lower=1.0)
        self.prob.driver.add_constraint('sm.ball.axial_local_unity', upper=1.0)
        self.prob.driver.add_constraint('sm.ball.axial_general_unity', upper=1.0)
        self.prob.driver.add_constraint('sm.ball.external_local_unity', upper=1.0)
        self.prob.driver.add_constraint('sm.ball.external_general_unity', upper=1.0)

        # Pontoon tube radii
        self.prob.driver.add_constraint('sm.pon.pontoon_radii_ratio', upper=1.0)
        self.prob.driver.add_constraint('sm.pon.base_connection_ratio',upper=0.0)
        self.prob.driver.add_constraint('sm.pon.ballast_connection_ratio',upper=0.0)

        # Pontoon stress safety factor
        self.prob.driver.add_constraint('sm.pon.axial_stress_factor', upper=0.8)
        self.prob.driver.add_constraint('sm.pon.shear_stress_factor', upper=0.8)
        
        # Achieving non-zero variable ballast height means the semi can be balanced with margin as conditions change
        self.prob.driver.add_constraint('sm.sm.variable_ballast_height', lower=2.0, upper=100.0)
        self.prob.driver.add_constraint('sm.sm.variable_ballast_mass', lower=0.0)

        # Metacentric height should be positive for static stability
        self.prob.driver.add_constraint('sm.sm.metacentric_height', lower=0.1)

        # Center of buoyancy should be above CG (difference should be positive)
        self.prob.driver.add_constraint('sm.sm.static_stability', lower=0.1)

        # Surge restoring force should be greater than wave-wind forces (ratio < 1)
        self.prob.driver.add_constraint('sm.sm.offset_force_ratio',lower=0.0, upper=1.0)

        # Heel angle should be less than 6deg for ordinary operation, less than 10 for extreme conditions
        self.prob.driver.add_constraint('sm.sm.heel_angle',lower=0.0, upper=10.0)


        # OBJECTIVE FUNCTION: Minimize total cost!
        self.prob.driver.add_objective('lcoe.lcoe')


        
    def visualize(self, fname=None):
        fig = self.init_figure()

        self.draw_ocean(fig)

        mooringMat = self.prob['sm.mm.plot_matrix']
        self.draw_mooring(fig, mooringMat)

        pontoonMat = self.prob['sm.pon.plot_matrix']
        zcut = 1.0 + np.maximum( self.params['freeboard_base'], self.params['freeboard_ballast'] )
        self.draw_pontoons(fig, pontoonMat, 0.5*self.params['pontoon_outer_diameter'], zcut)

        self.draw_cylinder(fig, [0.0, 0.0], self.params['freeboard_base'], self.params['section_height_base'],
                           0.5*self.params['outer_diameter_base'], self.params['stiffener_spacing_base'])

        R_semi    = self.params['radius_to_ballast_cylinder']
        ncylinder = self.params['number_of_ballast_columns']
        angles = np.linspace(0, 2*np.pi, ncylinder+1)
        x = R_semi * np.cos( angles )
        y = R_semi * np.sin( angles )
        for k in xrange(ncylinder):
            self.draw_cylinder(fig, [x[k], y[k]], self.params['freeboard_ballast'], self.params['section_height_ballast'],
                               0.5*self.params['outer_diameter_ballast'], self.params['stiffener_spacing_ballast'])
            
        self.set_figure(fig, fname)
        


def example():
    mysemi = FloatingTurbineInstance()
    mysemi.evaluate('psqp')
    #mysemi.visualize('semi-initial.jpg')
    #mysemi.run('slsqp')
    return mysemi

def slsqp_optimal():
    #OrderedDict([('sm.total_cost', array([0.65987536]))])
    mysemi = FloatingTurbineInstance()

    fairlead=0.700118055345
    fairlead_offset_from_shell=4.99999573349
    radius_to_ballast_cylinder=24.0062943408
    freeboard_base=0.43911732433
    section_height_base=[3.22351301, 1.18643201, 0.36842117, 7.85819571, 0.10149862]
    outer_diameter_base=[1.10215765, 1.10176146, 1.12730743, 1.40557497, 1.34551741, 3.11536911]
    wall_thickness_base=[0.00888014, 0.00908301, 0.00711699, 0.0074305, 0.01036372, 0.0064007, ]
    freeboard_ballast=0.00917073242087
    section_height_ballast=[0.41568677, 0.19027904, 0.1027831, 0.10319249, 0.10167768]
    outer_diameter_ballast=[1.1, 1.10000678, 21.84555589, 15.7451946, 7.98475117, 3.57187412]
    wall_thickness_ballast=[0.00540337, 0.00512813, 0.005, 0.00504785, 0.00531883, 0.02901446]
    pontoon_outer_diameter=0.238587591233
    pontoon_inner_diameter=0.0282506439854
    tower_diameter=[3.00279735, 4.79003221, 3.78145422, 3.1666783, 3.05500872, 3.25601802]
    tower_wall_thickness=[0.00210799, 0.00200344, 0.00200777, 0.00617966, 0.01447719, 0.01161293]
    tower_section_height=[0.1, 0.1, 0.1, 0.1, 0.1]
    scope_ratio=4.29418386598
    anchor_radius=836.13551283
    mooring_diameter=0.0500184150196
    stiffener_web_height_base=[0.01232775, 0.06213763, 0.03086181, 0.51465266, 0.17599423]
    stiffener_web_thickness_base=[0.00383956, 0.0244798, 0.01232182, 0.03480631, 0.01458695]
    stiffener_flange_width_base=[0.01836843, 0.06126301, 0.01334884, 4.13546424, 0.22894627]
    stiffener_flange_thickness_base=[0.01045948, 0.01846925, 0.00807756, 0.49859327, 0.07726461]
    stiffener_spacing_base=[0.60855135, 0.78253197, 0.42576036, 8.271076, 6.61783294]
    permanent_ballast_height_base=1.6046620632
    stiffener_web_height_ballast=[0.01001693, 0.01014131, 0.01, 0.01000002, 0.08771132]
    stiffener_web_thickness_ballast=[0.16070436, 0.00309267, 0.001, 0.00100016, 0.46245178]
    stiffener_flange_width_ballast=[0.71634965, 0.01014155, 0.01, 0.01000002, 0.02378251]
    stiffener_flange_thickness_ballast=[0.11864209, 0.00309244, 0.001, 0.00100018, 0.42417502]
    stiffener_spacing_ballast=[9.066162, 0.75015028, 0.25110771, 0.39769332, 11.91746405]
    permanent_ballast_height_ballast=0.331014224803

    mysemi.evaluate('slsqp')
    mysemi.visualize('semi-psqp.jpg')
    return mysemi
    
if __name__ == '__main__':
    #slsqp_optimal()
    example()
