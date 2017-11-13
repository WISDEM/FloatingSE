from semiInstance import SemiInstance
import numpy as np
import time

def example_semi():
    mysemi = SemiInstance()
    mysemi.run('psqp')
    return mysemi

def psqp_optimal():
    #OrderedDict([('sm.total_cost', array([ 0.478222]))])
    mysemi = SemiInstance()
    mysemi.fairlead = 0.98837804
    mysemi.fairlead_offset_from_shell = 0.03492563
    mysemi.radius_to_ballast_cylinder = 12.50450881
    mysemi.freeboard_base =  4.73371486e-05
    mysemi.section_height_base = np.array([ 6.87136507,  6.73962166,  7.10951696,  7.74446946,  5.06246213])
    mysemi.outer_radius_base = np.array([ 4.22649245,  4.6491417 ,  5.11405587,  4.60265028,  4.14238525, 3.72814673])
    mysemi.wall_thickness_base = np.array([ 0.005     ,  0.005     ,  0.005     ,  0.005     ,  0.005     , 0.00803123])
    mysemi.freeboard_ballast = 1.34505626
    mysemi.section_height_ballast = np.array([ 0.46278875,  0.26941733,  0.25622358,  0.65026967,  0.69473497])
    mysemi.outer_radius_ballast = np.array([ 1.1     ,  1.21    ,  1.331   ,  1.4641  ,  1.31769 ,  1.185921])
    mysemi.wall_thickness_ballast = np.array([ 0.005,  0.005,  0.005,  0.005,  0.005,  0.005])
    mysemi.scope_ratio = 2.33930678
    mysemi.anchor_radius = 449.98914924
    mysemi.mooring_diameter = 0.43319508
    mysemi.stiffener_web_height_base = np.array([ 0.08808218,  0.08498695,  0.07744793,  0.0651362 ,  0.06189142])
    mysemi.stiffener_web_thickness_base = np.array([ 0.00365833,  0.00352977,  0.00321665,  0.00270531,  0.00257054])
    mysemi.stiffener_flange_width_base = np.array([ 0.01      ,  0.01      ,  0.01      ,  0.01      ,  0.02975293])
    mysemi.stiffener_flange_thickness_base = np.array([ 0.02582615,  0.02323277,  0.01759444,  0.0116956 ,  0.00332889])
    mysemi.stiffener_spacing_base = np.array([ 0.15536269,  0.16733385,  0.1943853 ,  0.24621259,  0.29430671])
    mysemi.permanent_ballast_height_base = 4.44060642
    mysemi.stiffener_web_height_ballast = np.array([ 0.02407717,  0.01      ,  0.01      ,  0.01      ,  0.0108659 ])
    mysemi.stiffener_web_thickness_ballast = np.array([ 0.001,  0.001,  0.001,  0.001,  0.001])
    mysemi.stiffener_flange_width_ballast = np.array([ 0.01      ,  0.01      ,  0.04717587,  0.01      ,  0.01      ])
    mysemi.stiffener_flange_thickness_ballast = np.array([ 0.001     ,  0.001     ,  0.00261248,  0.001     ,  0.0020657 ])
    mysemi.stiffener_spacing_ballast = np.array([ 0.85071975,  1.27903722,  1.2899592 ,  1.65261691,  1.68134627])
    mysemi.permanent_ballast_height_ballast = 0.1

    mysemi.evaluate('psqp')
    mysemi.visualize('semi-psqp.jpg')
    return mysemi
    
    '''
OrderedDict([('sg.base_draft_depth_ratio', array([ 0.15379536])), ('sg.ballast_draft_depth_ratio', array([ 0.00453384])), ('sg.fairlead_draft_ratio', array([ 1.])), ('sg.base_taper_ratio', array([ 0.1,  0.1,  0.1,  0.1,  0.1])), ('sg.ballast_taper_ratio', array([ 0.1,  0.1,  0.1,  0.1,  0.1])), ('mm.safety_factor', array([ 0.79999972])), ('mm.mooring_length_min', array([ 1.0417004])), ('mm.mooring_length_max', array([ 0.76110371])), ('base.flange_spacing_ratio', array([ 0.06436552,  0.05976077,  0.05144422,  0.04061531,  0.10109498])), ('base.web_radius_ratio', array([ 0.01984809,  0.01740966,  0.01594119,  0.01489673,  0.01572738])), ('base.flange_compactness', array([ 46.63655042,  41.9534534 ,  31.77183121,  21.11977822,   2.02039828])), ('base.web_compactness', array([ 1.,  1.,  1.,  1.,  1.])), ('base.axial_local_unity', array([ 1.        ,  0.99999957,  1.        ,  0.99999998,  0.4951127 ])), ('base.axial_general_unity', array([ 1.        ,  1.00000011,  0.9963726 ,  0.95711537,  0.96815567])), ('base.external_local_unity', array([ 0.92480828,  0.93984832,  0.97171596,  0.99999998,  0.4951127 ])), ('base.external_general_unity', array([ 0.95265914,  0.96616978,  1.        ,  1.00000019,  1.00000176])), ('ball.flange_spacing_ratio', array([ 0.01175475,  0.00781838,  0.0365716 ,  0.00605101,  0.00594761])), ('ball.web_radius_ratio', array([ 0.02084604,  0.00787092,  0.00715538,  0.00718962,  0.00868018])), ('ball.flange_compactness', array([ 1.8057878 ,  1.8057878 ,  1.        ,  1.8057878 ,  3.73021882])), ('ball.web_compactness', array([ 1.        ,  2.40771706,  2.40771706,  2.40771706,  2.21584734])), ('ball.axial_local_unity', array([ 0.34513666,  0.57609198,  0.65011859,  0.88163762,  0.87608159])), ('ball.axial_general_unity', array([ 0.16949242,  0.1365349 ,  0.09848718,  0.92523936,  0.95183049])), ('ball.external_local_unity', array([ 0.34513666,  0.57609198,  0.65011859,  0.88163762,  0.87608159])), ('ball.external_general_unity', array([ 0.17928123,  0.14093933,  0.11602595,  0.95372108,  1.00045621])), ('sm.variable_ballast_height', array([ 28.68929657])), ('sm.variable_ballast_mass', array([ 1929784.88830389])), ('sm.metacentric_height', array([ 1.42537821])), ('sm.static_stability', array([ 1.22470678])), ('sm.offset_force_ratio', array([ 0.17013327])), ('sm.heel_angle', array([ 10.00000501]))])
    '''
    
if __name__ == '__main__':
    psqp_optimal()
    #example_semi()
