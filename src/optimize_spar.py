from sparInstance import SparInstance
import numpy as np
import time

def example_spar():
    myspar = SparInstance()
    myspar.run('psqp')
    
def psqp_optimal():
    #OrderedDict([('sp.total_cost', array([ 3.68125491]))])
    myspar = SparInstance()

    myspar.freeboard = 0.0
    myspar.fairlead = 28.98778705
    myspar.fairlead_offset_from_shell = 4.15123014
    myspar.section_height = np.array([ 49.01203711,  52.29873288,  33.77996355,   1.98476181,  26.42465958])
    myspar.outer_radius = np.array([ 3.76031187,  3.78714959,  4.16586455,  3.74927809,  3.37435028, 3.03691526])
    myspar.wall_thickness = np.array([ 0.00661516,  0.01453792,  0.005     ,  0.0097314 ,  0.0059745 , 0.005     ])
    myspar.scope_ratio = 2.60758833
    myspar.anchor_radius = 448.0246413
    myspar.mooring_diameter = 0.0724271
    myspar.stiffener_web_height= np.array([ 0.12411858,  0.12090302,  0.11733809,  0.07447426,  0.09330525])
    myspar.stiffener_web_thickness = np.array([ 0.00515503,  0.00502148,  0.00487342,  0.00309315,  0.00387526 ])
    myspar.stiffener_flange_width = np.array([ 0.01      ,  0.01      ,  0.01      ,  0.04525972,  0.01 ])
    myspar.stiffener_flange_thickness = np.array([ 0.27650071,  0.21402134,  0.13546971,  0.00276745,  0.03429803])
    myspar.stiffener_spacing = np.array([ 0.21849442,  0.26591885,  0.2661029 ,  0.32705785,  0.20720817])
    myspar.permanent_ballast_height = 30.06223745

    myspar.evaluate('psqp')
    '''
OrderedDict([('sg.draft_depth_ratio', array([ 0.75])), ('sg.fairlead_draft_ratio', array([ 0.17729533])), ('sg.taper_ratio', array([ 0.0071371,  0.1      ,  0.1      ,  0.1      ,  0.1      ])), ('mm.safety_factor', array([ 0.79959284])), ('mm.mooring_length_min', array([ 1.02897572])), ('mm.mooring_length_max', array([ 0.77368529])), ('cyl.flange_spacing_ratio', array([ 0.04576776,  0.03760546,  0.03757945,  0.13838446,  0.04826065])), ('cyl.web_radius_ratio', array([ 0.03289015,  0.03040433,  0.02964902,  0.02090908,  0.02910666])), ('cyl.flange_compactness', array([ 499.30160594,  386.47712046,  244.62955708,    1.10416659,
         61.93496328])), ('cyl.web_compactness', array([ 1.        ,  1.        ,  1.        ,  0.99999999,  1.        ])), ('cyl.axial_local_unity', array([ 0.99999959,  0.99999964,  0.99998764,  0.9949145 ,  0.99660552])), ('cyl.axial_general_unity', array([ 0.99999958,  0.99982145,  0.91031899,  1.2161923 ,  0.93928689])), ('cyl.external_local_unity', array([ 0.85868796,  0.89067778,  0.95522241,  0.98490783,  0.99660552])), ('cyl.external_general_unity', array([ 0.92690846,  0.99999937,  1.00243   ,  1.27160385,  1.00003484])), ('sp.variable_ballast_height', array([ 9.45559567])), ('sp.variable_ballast_mass', array([ 432098.15110704])), ('sp.metacentric_height', array([ 46.5396792])), ('sp.static_stability', array([ 46.53063839])), ('sp.offset_force_ratio', array([ 1.00056272])), ('sp.heel_angle', array([ 10.0002829]))])
    '''


def conmin_optimal():
    #OrderedDict([('sp.total_cost', array([ 8.15839897]))])
    myspar = SparInstance()
    myspar.freeboard = 5.0
    myspar.fairlead = 7.57
    myspar.fairlead_offset_from_shell = 0.05
    myspar.section_height = np.array([ 18.99987492,  18.9998873 ,  18.99990693,  18.99990914,  18.99990425])
    myspar.outer_radius = np.array([ 6.99962345,  6.99955813,  6.99973629,  6.99978022,  6.99976883, 6.99988])
    myspar.wall_thickness = np.array([ 0.03712666,  0.02787312,  0.02712097,  0.02206188,  0.02157211, 0.03579269])
    myspar.scope_ratio = 2.40997737
    myspar.anchor_radius = 450.0
    myspar.mooring_diameter = 0.1909802
    myspar.stiffener_web_height= np.array([ 0.10557588,  0.10316776,  0.09795284,  0.09743845,  0.09743956])
    myspar.stiffener_web_thickness = np.array([ 0.03599046,  0.03502903,  0.03323707,  0.03302298,  0.0330262 ])
    myspar.stiffener_flange_width = np.array([ 0.10066915,  0.10029873,  0.09894232,  0.09882406,  0.0988245 ])
    myspar.stiffener_flange_thickness = np.array([ 0.02739561,  0.02327079,  0.01406197,  0.01304515,  0.01304842])
    myspar.stiffener_spacing = np.array([ 0.40020418,  0.40036638,  0.4008825 ,  0.4009331 ,  0.40093272])
    myspar.permanent_ballast_height = 10.0

    myspar.evaluate('conmin')
    '''
OrderedDict([('sg.draft_depth_ratio', array([ 0.41284162])), ('mm.safety_factor', array([ 0.7999935])), ('mm.mooring_length_min', array([ 1.03413212])), ('mm.mooring_length_max', array([ 0.76788083])), ('sp.flange_spacing_ratio', array([ 0.25154447,  0.25051738,  0.24681128,  0.24648516,  0.2464865 ])), ('sp.web_radius_ratio', array([ 0.01508315,  0.01473899,  0.01399375,  0.01392023,  0.01392029])), ('sp.flange_compactness', array([ 4.91418164,  4.18969521,  2.56643842,  2.38370799,  2.38429482])), ('sp.web_compactness', array([ 8.20782661,  8.17503344,  8.16979471,  8.16002162,  8.160726  ])), ('sp.axial_local_unity', array([ 0.51507608,  0.46336477,  0.38573459,  0.25387818,  0.06900377])), ('sp.axial_general_unity', array([ 0.9982523 ,  0.95677559,  0.9846294 ,  0.65050248,  0.19042545])), ('sp.external_local_unity', array([ 0.43199964,  0.39057533,  0.32676712,  0.21654406,  0.05750315])), ('sp.external_general_unity', array([ 1.00397317,  0.96693407,  0.99605626,  0.66093149,  0.1917524 ])), ('sp.metacentric_height', array([ 20.55260275])), ('sp.static_stability', array([ 20.41649121])), ('sp.variable_ballast_height', array([ 28.52903672])), ('sp.variable_ballast_mass', array([ 4464694.51334896])), ('sp.offset_force_ratio', array([ 0.98529446])), ('sp.heel_angle', array([ 2.39612487]))])
    '''


def cobyla_optimal():
    #OrderedDict([('sp.total_cost', array([ 6.83851908]))])
    myspar = SparInstance()
    myspar.freeboard = 7.56789854
    myspar.fairlead = 9.41184644
    myspar.fairlead_offset_from_shell = 0.0471558864
    myspar.section_height = np.array([ 18.708914991,  18.784270853,  18.799716693,  18.648435942, 18.711380637])
    myspar.outer_radius = np.array([ 5.764219519,  5.657993694,  6.159558061,  6.125155506, 6.293851894,  6.606570305])
    myspar.wall_thickness = np.array([ 0.043758918  ,  0.03934623132,  0.04101795034,  0.03947006871, 0.03855182803,  0.04268526778])
    myspar.scope_ratio = 2.39202552
    myspar.anchor_radius = 442.036507
    myspar.mooring_diameter = 0.153629334
    myspar.stiffener_web_height= np.array([ 0.1433863028,  0.1192863504,  0.1102913546,   0.0959098443,   0.0760210847])
    myspar.stiffener_web_thickness = np.array([ 0.0059552804,  0.0049543342,  0.004580744 ,  0.003983435 ,  0.0031573928 ])
    myspar.stiffener_flange_width = np.array([ 0.0924192057,  0.0977347306,  0.0800589589,  0.0797488027,  0.0861943184 ])
    myspar.stiffener_flange_thickness = np.array([ 0.02739561,  0.02327079,  0.01406197,  0.01304515,  0.01304842])
    myspar.stiffener_spacing = np.array([ 0.937472777,   0.913804583,   0.975992681,   0.940785141,  1.077950861])
    myspar.permanent_ballast_height = 2.1531719

    myspar.evaluate('cobyla')
    '''
OrderedDict([('sg.draft_depth_ratio', array([ 0.3948845])), ('mm.safety_factor', array([ 0.79998459])), ('mm.mooring_length_min', array([ 1.03296286])), ('mm.mooring_length_max', array([ 0.76687562])), ('sp.flange_spacing_ratio', array([ 0.09858335,  0.10695364,  0.08202824,  0.08476835,  0.07996127])), ('sp.web_radius_ratio', array([ 0.02510657,  0.020188  ,  0.01795587,  0.01544565,  0.01178583])), ('sp.flange_compactness', array([ 4.34435796,  3.92278448,  4.35797943,  2.16402123,  1.        ])), ('sp.web_compactness', array([ 1.,  1.,  1.,  1.,  1.])), ('sp.axial_local_unity', array([ 0.47642979,  0.38788415,  0.28705696,  0.17106678,  0.05175507])), ('sp.axial_general_unity', array([ 0.97397846,  0.97695689,  0.98037179,  0.98931662,  0.5567153 ])), ('sp.external_local_unity', array([ 0.41057698,  0.33470586,  0.24904213,  0.14822411,  0.04496548])), ('sp.external_general_unity', array([ 1.        ,  1.        ,  1.        ,  1.        ,  0.56070475])), ('sp.metacentric_height', array([ 6.12375997])), ('sp.static_stability', array([ 5.98379185])), ('sp.variable_ballast_height', array([ 60.25812533])), ('sp.variable_ballast_mass', array([ 6785190.91932651])), ('sp.offset_force_ratio', array([ 1.0000217])), ('sp.heel_angle', array([ 9.99999845]))])
    '''
        
if __name__ == '__main__':
    example_spar()
