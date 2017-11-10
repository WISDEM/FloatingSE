from semiInstance import SemiInstance
import numpy as np
import time

def example_semi():
    mysemi = SemiInstance()
    mysemi.run('psqp')

def psqp_optimal():
    #OrderedDict([('sm.total_cost', array([ 0.51482756]))])
    mysemi = SparInstance()
    mysemi.radius_to_ballast_cylinder = 10.0
    mysemi.freeboard_base = 5.0
    mysemi.freeboard_ballast = 5.0
    mysemi.fairlead = 7.57
    mysemi.fairlead_offset_from_shell = 0.05
    mysemi.base_length = 40.0
    mysemi.ballast_length = 20.0
    mysemi.outer_radius_base = 7.0
    mysemi.wall_thickness_base = 0.05
    mysemi.outer_radius_ballast = 7.0
    mysemi.wall_thickness_ballast = 0.05
    mysemi.scope_ratio = 2.41
    mysemi.anchor_radius = 450.0
    mysemi.mooring_diameter = 0.19
    mysemi.number_of_mooring_lines = 3
    mysemi.number_of_ballast_columns = 3
    mysemi.permanent_ballast_height_base = 10.0
    mysemi.stiffener_web_height_base= 0.1
    mysemi.stiffener_web_thickness_base = 0.04
    mysemi.stiffener_flange_width_base = 0.1
    mysemi.stiffener_flange_thickness_base = 0.02
    mysemi.stiffener_spacing_base = 0.4
    mysemi.permanent_ballast_height_ballast = 10.0
    mysemi.stiffener_web_height_ballast= 0.1
    mysemi.stiffener_web_thickness_ballast = 0.04
    mysemi.stiffener_flange_width_ballast = 0.1
    mysemi.stiffener_flange_thickness_ballast = 0.02
    mysemi.stiffener_spacing_ballast = 0.4

    mysemi.freeboard = 0.0
    mysemi.fairlead = 28.98778705
    mysemi.fairlead_offset_from_shell = 4.15123014
    mysemi.section_height = np.array([ 49.01203711,  52.29873288,  33.77996355,   1.98476181,  26.42465958])
    mysemi.outer_radius = np.array([ 3.76031187,  3.78714959,  4.16586455,  3.74927809,  3.37435028, 3.03691526])
    mysemi.wall_thickness = np.array([ 0.00661516,  0.01453792,  0.005     ,  0.0097314 ,  0.0059745 , 0.005     ])
    mysemi.scope_ratio = 2.60758833
    mysemi.anchor_radius = 448.0246413
    mysemi.mooring_diameter = 0.0724271
    mysemi.stiffener_web_height= np.array([ 0.12411858,  0.12090302,  0.11733809,  0.07447426,  0.09330525])
    mysemi.stiffener_web_thickness = np.array([ 0.00515503,  0.00502148,  0.00487342,  0.00309315,  0.00387526 ])
    mysemi.stiffener_flange_width = np.array([ 0.01      ,  0.01      ,  0.01      ,  0.04525972,  0.01 ])
    mysemi.stiffener_flange_thickness = np.array([ 0.27650071,  0.21402134,  0.13546971,  0.00276745,  0.03429803])
    mysemi.stiffener_spacing = np.array([ 0.21849442,  0.26591885,  0.2661029 ,  0.32705785,  0.20720817])
    mysemi.permanent_ballast_height = 30.06223745

    mysemi.evaluate('psqp')
    '''
OrderedDict([('sg.base_draft_depth_ratio', array([ 0.17275925])), ('sg.ballast_draft_depth_ratio', array([ 0.02265394])), ('sg.fairlead_draft_ratio', array([ 1.])), ('sg.base_taper_ratio', array([ 0.1       ,  0.1       ,  0.09026335,  0.1       ,  0.1       ])), ('sg.ballast_taper_ratio', array([ 0.04127914,  0.1       ,  0.1       ,  0.1       ,  0.07875056])), ('mm.safety_factor', array([ 0.80000027])), ('mm.mooring_length_min', array([ 1.04223784])), ('mm.mooring_length_max', array([ 0.76611317])), ('base.flange_spacing_ratio', array([ 0.06314612,  0.05871862,  0.05291367,  0.0438366 ,  0.10831451])), ('base.web_radius_ratio', array([ 0.02201697,  0.01999476,  0.01701493,  0.01499987,  0.01711951])), ('base.flange_compactness', array([ 47.71109593,  38.51075605,  34.44122107,  25.56695872,   2.82119422])), ('base.web_compactness', array([ 1.,  1.,  1.,  1.,  1.])), ('base.axial_local_unity', array([ 0.99999908,  0.99999967,  0.99999214,  0.99999973,  0.46746784])), ('base.axial_general_unity', array([ 1.00000014,  1.0000001 ,  0.99956729,  0.96265414,  0.96543846])), ('base.external_local_unity', array([ 0.92714649,  0.94260721,  0.96561723,  0.99999973,  0.46746784])), ('base.external_general_unity', array([ 0.96482652,  0.97769587,  1.00000901,  1.00000091,  1.00000029])), ('ball.flange_spacing_ratio', array([ 0.01346932,  0.00519513,  0.00480397,  0.0137002 ,  0.01821247])), ('ball.web_radius_ratio', array([ 0.02184455,  0.02153266,  0.02257112,  0.01616389,  0.01344929])), ('ball.flange_compactness', array([ 1.98270488,  1.8057878 ,  1.8057878 ,  1.        ,  1.        ])), ('ball.web_compactness', array([ 1.        ,  1.        ,  1.        ,  1.13133363,  1.4940478 ])), ('ball.axial_local_unity', array([ 0.3967597 ,  0.96189693,  1.        ,  0.82660218,  0.89949624])), ('ball.axial_general_unity', array([ 0.94195881,  0.94114672,  0.59457037,  0.49548101,  0.3633438 ])), ('ball.external_local_unity', array([ 0.3967597 ,  0.96189693,  1.        ,  0.82660218,  0.89949624])), ('ball.external_general_unity', array([ 1.00010512,  1.00018247,  0.63772046,  0.53578781,  0.40977581])), ('sm.variable_ballast_height', array([ 27.05155972])), ('sm.variable_ballast_mass', array([ 1701445.12558187])), ('sm.metacentric_height', array([ 2.84155882])), ('sm.static_stability', array([ 2.70671224])), ('sm.offset_force_ratio', array([ 0.17201257])), ('sm.heel_angle', array([ 10.00003879]))])

OrderedDict([('fairlead.x', array([ 4.93855938])), ('fairlead_offset_from_shell.x', array([ 0.02512148])), ('radius_to_ballast_cylinder.x', array([ 10.51684707])), ('freeboard_base.x', array([  2.24630975e-05])), ('section_height_base.x', array([ 7.66926714,  7.43839348,  7.46690128,  7.63755397,  7.44942337])), ('outer_radius_base.x', array([ 3.65153992,  4.01669392,  4.41836331,  4.81717959,  4.33546163,
        3.90191547])), ('wall_thickness_base.x', array([ 0.005     ,  0.005     ,  0.005     ,  0.005     ,  0.005     ,
        0.00822288])), ('freeboard_ballast.x', array([ 1.39099538])), ('section_height_ballast.x', array([ 2.50103369,  1.36189629,  0.98551523,  0.88036519,  0.60074436])), ('outer_radius_ballast.x', array([ 1.1       ,  1.14540706,  1.25994776,  1.38594254,  1.24734829,
        1.14911891])), ('wall_thickness_ballast.x', array([ 0.005     ,  0.005     ,  0.005     ,  0.00514155,  0.005     ,
        0.005     ])), ('scope_ratio.x', array([ 2.38418696])), ('anchor_radius.x', array([ 449.99766102])), ('mooring_diameter.x', array([ 0.44420369])), ('stiffener_web_height_base.x', array([ 0.08441564,  0.08432849,  0.07857106,  0.0686442 ,  0.07050994])), ('stiffener_web_thickness_base.x', array([ 0.00350605,  0.00350243,  0.0032633 ,  0.00285101,  0.0029285 ])), ('stiffener_flange_width_base.x', array([ 0.01      ,  0.01      ,  0.01      ,  0.01      ,  0.03038519])), ('stiffener_flange_thickness_base.x', array([ 0.02642121,  0.02132629,  0.01907268,  0.01415834,  0.0047471 ])), ('stiffener_spacing_base.x', array([ 0.15836285,  0.17030372,  0.18898707,  0.2281199 ,  0.28052747])), ('permanent_ballast_height_base.x', array([ 6.00143667])), ('stiffener_web_height_ballast.x', array([ 0.02452495,  0.02589684,  0.02986035,  0.02128211,  0.0161154 ])), ('stiffener_web_thickness_ballast.x', array([ 0.0010186 ,  0.00107558,  0.00124019,  0.001     ,  0.001     ])), ('stiffener_flange_width_ballast.x', array([ 0.01      ,  0.01      ,  0.01      ,  0.02384101,  0.03333377])), ('stiffener_flange_thickness_ballast.x', array([ 0.00109797,  0.001     ,  0.001     ,  0.00132026,  0.00184594])), ('stiffener_spacing_ballast.x', array([ 0.74242778,  1.92487935,  2.08161196,  1.74019403,  1.83027142])), ('permanent_ballast_height_ballast.x', array([ 0.1]))])

    '''
    
if __name__ == '__main__':
    example_semi()
