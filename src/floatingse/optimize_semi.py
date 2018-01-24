from semiInstance import SemiInstance
import numpy as np
import time

def example_semi():
    mysemi = SemiInstance()
    #mysemi.evaluate('psqp')
    #mysemi.visualize('semi-initial.jpg')
    mysemi.run('psqp')
    return mysemi

def psqp_optimal():
    #OrderedDict([('sm.total_cost', array([0.65987536]))])
    mysemi = SemiInstance()

    mysemi.fairlead = 22.2366002
    mysemi.fairlead_offset_from_shell = 4.99949523
    mysemi.radius_to_ballast_cylinder = 26.79698385
    mysemi.freeboard_base = 4.97159308
    mysemi.section_height_base = np.array([6.72946378, 5.97993104, 5.47072089, 5.71437475, 5.44290777])
    mysemi.outer_radius_base = np.array([2.0179943 , 2.21979373, 2.4417731 , 2.68595041, 2.95454545, 3.25 ])
    mysemi.wall_thickness_base = np.array([0.01100738, 0.00722966, 0.00910002, 0.01033024, 0.00639292, 0.00560714])
    mysemi.freeboard_ballast = -1.14370386e-20
    mysemi.section_height_ballast = np.array([1.44382195, 2.71433629, 6.1047888 , 5.14428218, 6.82937098])
    mysemi.outer_radius_ballast = np.array([2.57228724, 2.82647421, 3.10005118, 3.40594536, 3.74653989, 4.12119389])
    mysemi.wall_thickness_ballast = np.array([0.01558312, 0.005 , 0.005 , 0.005 , 0.005 , 0.005 ])
    mysemi.outer_pontoon_radius = 0.92428188
    mysemi.inner_pontoon_radius = 0.88909984
    mysemi.scope_ratio = 4.71531904
    mysemi.anchor_radius = 837.58954811
    mysemi.mooring_diameter = 0.36574595
    mysemi.stiffener_web_height_base = np.array([0.01625364, 0.04807025, 0.07466081, 0.0529478 , 0.03003529])
    mysemi.stiffener_web_thickness_base = np.array([0.00263325, 0.00191218, 0.00404707, 0.00495706, 0.00137335])
    mysemi.stiffener_flange_width_base = np.array([0.0100722 , 0.06406752, 0.01342377, 0.07119415, 0.01102604])
    mysemi.stiffener_flange_thickness_base = np.array([0.06126737, 0.00481305, 0.01584461, 0.00980356, 0.01218029])
    mysemi.stiffener_spacing_base = np.array([1.09512893, 0.67001459, 1.60080836, 1.27068546, 0.2687786 ])
    mysemi.permanent_ballast_height_base = 5.34047386
    mysemi.stiffener_web_height_ballast = np.array([0.04750412, 0.03926778, 0.04484479, 0.04255339, 0.05903525])
    mysemi.stiffener_web_thickness_ballast = np.array([0.00197299, 0.00162998, 0.00186254, 0.00176738, 0.00245192])
    mysemi.stiffener_flange_width_ballast = np.array([0.01176864, 0.01018018, 0.01062256, 0.01119399, 0.01023957])
    mysemi.stiffener_flange_thickness_ballast = np.array([0.00182314, 0.00428608, 0.01616793, 0.0109717 , 0.00814284])
    mysemi.stiffener_spacing_ballast = np.array([0.88934305, 0.19623501, 0.29410086, 0.30762027, 0.4208429 ])
    mysemi.permanent_ballast_height_ballast = 0.10007504

    mysemi.evaluate('psqp')
    mysemi.visualize('semi-psqp.jpg')
    return mysemi
    
    '''
OrderedDict([('fairlead.x', array([22.2366002])), ('fairlead_offset_from_shell.x', array([4.99949523])), ('radius_to_ballast_cylinder.x', array([26.79698385])), ('freeboard_base.x', array([4.97159308])), ('section_height_base.x', array([6.72946378, 5.97993104, 5.47072089, 5.71437475, 5.44290777])), ('outer_radius_base.x', array([2.0179943 , 2.21979373, 2.4417731 , 2.68595041, 2.95454545,
       3.25      ])), ('wall_thickness_base.x', array([0.01100738, 0.00722966, 0.00910002, 0.01033024, 0.00639292,
       0.00560714])), ('freeboard_ballast.x', array([-1.14370386e-20])), ('section_height_ballast.x', array([1.44382195, 2.71433629, 6.1047888 , 5.14428218, 6.82937098])), ('outer_radius_ballast.x', array([2.57228724, 2.82647421, 3.10005118, 3.40594536, 3.74653989,
       4.12119389])), ('wall_thickness_ballast.x', array([0.01558312, 0.005     , 0.005     , 0.005     , 0.005     ,
       0.005     ])), ('outer_pontoon_radius.x', array([0.92428188])), ('inner_pontoon_radius.x', array([0.88909984])), ('scope_ratio.x', array([4.71531904])), ('anchor_radius.x', array([837.58954811])), ('mooring_diameter.x', array([0.36574595])), ('stiffener_web_height_base.x', array([0.01625364, 0.04807025, 0.07466081, 0.0529478 , 0.03003529])), ('stiffener_web_thickness_base.x', array([0.00263325, 0.00191218, 0.00404707, 0.00495706, 0.00137335])), ('stiffener_flange_width_base.x', array([0.0100722 , 0.06406752, 0.01342377, 0.07119415, 0.01102604])), ('stiffener_flange_thickness_base.x', array([0.06126737, 0.00481305, 0.01584461, 0.00980356, 0.01218029])), ('stiffener_spacing_base.x', array([1.09512893, 0.67001459, 1.60080836, 1.27068546, 0.2687786 ])), ('permanent_ballast_height_base.x', array([5.34047386])), ('stiffener_web_height_ballast.x', array([0.04750412, 0.03926778, 0.04484479, 0.04255339, 0.05903525])), ('stiffener_web_thickness_ballast.x', array([0.00197299, 0.00162998, 0.00186254, 0.00176738, 0.00245192])), ('stiffener_flange_width_ballast.x', array([0.01176864, 0.01018018, 0.01062256, 0.01119399, 0.01023957])), ('stiffener_flange_thickness_ballast.x', array([0.00182314, 0.00428608, 0.01616793, 0.0109717 , 0.00814284])), ('stiffener_spacing_ballast.x', array([0.88934305, 0.19623501, 0.29410086, 0.30762027, 0.4208429 ])), ('permanent_ballast_height_ballast.x', array([0.10007504]))])
    '''
    
if __name__ == '__main__':
    #psqp_optimal()
    example_semi()
